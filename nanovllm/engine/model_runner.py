import pickle
import gc
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.linear import MarlinInt8Linear, _int8_per_channel_cublas
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.rwkv7 import RWKV7ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.use_state_cache = config.use_state_cache
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        if self.use_state_cache:
            # Handle both torch.dtype and string representations
            dtype = hf_config.torch_dtype
            if isinstance(dtype, str):
                assert dtype == "float16" or dtype == "torch.float16"
            else:
                assert dtype == torch.float16

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        default_dtype = torch.get_default_dtype()
        # Handle torch_dtype that might be a string
        dtype = hf_config.torch_dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""))
        torch.set_default_dtype(dtype)
        if self.use_state_cache:
            # RWKV replaces most large parameter storages during load_pth().
            # Build the module skeleton on CPU to avoid preallocating dead CUDA buffers.
            torch.set_default_device("cpu")
            self.model = RWKV7ForCausalLM(hf_config)
            # RWKV post-load quantization and sizing need runtime config knobs
            # (e.g. rwkv_int8_*), not just the HF model config.
            self.model.config = config
            torch.set_default_device("cuda")
        else:
            torch.set_default_device("cuda")
            self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # Allocate cache before warmup for RWKV (state cache is required for forward)
        if self.use_state_cache:
            self.allocate_state_cache()
        else:
            self.warmup_model()
            self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        try:
            if self.world_size > 1 and hasattr(self, "shm"):
                try:
                    self.shm.close()
                except Exception:
                    pass
                if dist.is_available() and dist.is_initialized():
                    try:
                        dist.barrier()
                    except Exception:
                        pass
                if self.rank == 0:
                    try:
                        self.shm.unlink()
                    except Exception:
                        pass

            if not self.enforce_eager:
                for attr in ("graphs", "graph_pool"):
                    if hasattr(self, attr):
                        try:
                            delattr(self, attr)
                        except Exception:
                            pass

            for attr in ("state_cache", "token_shift_cache", "kv_cache", "model", "sampler"):
                if hasattr(self, attr):
                    try:
                        delattr(self, attr)
                    except Exception:
                        pass

            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        finally:
            if dist.is_available() and dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def allocate_state_cache(self):
        config = self.config
        hf_config = config.hf_config
        num_heads = hf_config.num_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_heads)
        block_bytes = hf_config.num_hidden_layers * (head_dim + 2) * num_heads * head_dim * hf_config.torch_dtype.itemsize
        self.warmup_int8_kernels()
        prefill_probe_bytes = self.measure_state_prefill_probe_bytes(
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=1,
            prompt_len=config.rwkv_prefill_token_budget,
        )

        def compute_num_state_blocks():
            free, total = torch.cuda.mem_get_info()
            reserve = total * (1 - config.gpu_memory_utilization)
            available = free - reserve - prefill_probe_bytes
            return int(available) // block_bytes

        config.num_state_blocks = compute_num_state_blocks()
        if config.num_state_blocks <= 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            config.num_state_blocks = compute_num_state_blocks()
        if config.num_state_blocks <= 0:
            raise RuntimeError(
                f"Unable to allocate RWKV state cache: computed num_state_blocks={config.num_state_blocks}. "
                "Try lowering model memory pressure or increasing gpu_memory_utilization."
            )
        self.state_cache = torch.zeros(hf_config.num_hidden_layers, config.num_state_blocks, num_heads, head_dim, head_dim)
        self.token_shift_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_state_blocks, hf_config.hidden_size)
        self.bind_state_cache_modules(self.state_cache, self.token_shift_cache)
        target_model = getattr(self.model, "model", self.model)
        if hasattr(target_model, "decode_tokenshift_scratch"):
            target_model.decode_tokenshift_scratch = torch.empty(
                config.max_num_seqs,
                hf_config.hidden_size,
                dtype=hf_config.torch_dtype,
                device=self.state_cache.device,
            )

    def warmup_int8_kernels(self):
        if not self.use_state_cache or not torch.cuda.is_available():
            return
        dtype = self.config.hf_config.torch_dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""))
        warmed = False
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, MarlinInt8Linear):
                    x = torch.zeros((1, module.input_size), device=module.qweight.device, dtype=dtype)
                    _ = module(x)
                    warmed = True
            lm_head = getattr(self.model, "lm_head", None)
            if lm_head is not None and getattr(lm_head, "use_int8", False):
                in_features = lm_head.qweight.shape[1]
                x = torch.zeros((1, in_features), device=lm_head.qweight.device, dtype=dtype)
                _ = _int8_per_channel_cublas(x, lm_head.qweight, lm_head.scales, lm_head.scales_fp16, None)
                warmed = True
        if warmed:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

    def bind_state_cache_modules(self, state_cache: torch.Tensor, token_shift_cache: torch.Tensor):
        # Use sets to track which layers have been assigned
        att_assigned = set()
        ffn_assigned = set()
        for name, module in self.model.named_modules():
            # Try to get layer_id from layer_idx attribute or parse from module name
            layer_id = None
            if hasattr(module, "layer_idx"):
                layer_id = module.layer_idx
            else:
                # Parse from name like "blocks.0.att" or "model.blocks.5.ffn"
                import re
                match = re.search(r'\.blocks?\.(\d+)\.', name)
                if match:
                    layer_id = int(match.group(1))

            if layer_id is not None:
                if hasattr(module, "att_tokenshift_cache") and hasattr(module, "state_cache"):
                    if layer_id not in att_assigned:
                        module.att_tokenshift_cache = token_shift_cache[0, layer_id]
                        module.state_cache = state_cache[layer_id]
                        att_assigned.add(layer_id)
                if hasattr(module, "ffn_tokenshift_cache"):
                    if layer_id not in ffn_assigned:
                        module.ffn_tokenshift_cache = token_shift_cache[1, layer_id]
                        ffn_assigned.add(layer_id)

    def measure_state_prefill_probe_bytes(self, num_heads: int, head_dim: int, batch_size: int, prompt_len: int) -> int:
        hf_config = self.config.hf_config
        probe_state_cache = torch.zeros(hf_config.num_hidden_layers, batch_size, num_heads, head_dim, head_dim)
        probe_token_shift_cache = torch.zeros(2, hf_config.num_hidden_layers, batch_size, hf_config.hidden_size)
        self.bind_state_cache_modules(probe_state_cache, probe_token_shift_cache)

        seqs = []
        for block_id in range(batch_size):
            seq = Sequence([0] * prompt_len)
            seq.block_table = [block_id]
            seqs.append(seq)
        input_ids, positions = self.prepare_prefill(seqs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
        _ = self.run_model(input_ids, positions, True)
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated()
        reset_context()

        del input_ids, positions
        del probe_state_cache, probe_token_shift_cache
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        return max(0, peak_alloc - base_alloc)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        if self.use_state_cache:
            input_rows = []
            position_rows = []
            slot_mapping_in = []
            slot_mapping_out = []
            context_lens = []
            max_seqlen = max(len(seq) - seq.num_cached_tokens for seq in seqs)
            for seq in seqs:
                new_token_ids = seq[seq.num_cached_tokens:]
                seqlen = len(new_token_ids)
                pad_len = max_seqlen - seqlen
                input_rows.append([0] * pad_len + new_token_ids)
                position_rows.append([0] * pad_len + list(range(seqlen)))
                context_lens.append(seqlen)
                # Warmup may bypass scheduler allocation.
                block_id = seq.block_table[0] if seq.block_table else 0
                slot_mapping_in.append(block_id)
                slot_mapping_out.append(block_id)
            input_ids = torch.tensor(input_rows, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            positions = torch.tensor(position_rows, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            slot_mapping_in = torch.tensor(slot_mapping_in, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            slot_mapping_out = torch.tensor(slot_mapping_out, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            set_context(True, slot_mapping_in=slot_mapping_in, slot_mapping_out=slot_mapping_out, context_lens=context_lens)
            return input_ids, positions

        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        if self.use_state_cache:
            input_ids = []
            positions = []
            slot_mapping_in = []
            slot_mapping_out = []
            context_lens = []
            for seq in seqs:
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                context_lens.append(len(seq))
                block_id = seq.block_table[0]
                slot_mapping_in.append(block_id)
                slot_mapping_out.append(block_id)
            input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            slot_mapping_in = torch.tensor(slot_mapping_in, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            slot_mapping_out = torch.tensor(slot_mapping_out, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            set_context(False, context_lens=context_lens, slot_mapping_in=slot_mapping_in, slot_mapping_out=slot_mapping_out)
            return input_ids, positions

        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if self.use_state_cache:
            return self.model.forward_logits(input_ids, positions)
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        if self.use_state_cache:
            return
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
