import pickle
import gc

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.linear import MarlinInt8Linear
from nanovllm.models.rwkv7 import RWKV7ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        model_config = config.model_config
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.eos = config.eos
        self._bs1_decode_tensors = None
        self._bs1_temperature = None
        self._bs1_decode_graphs = {}
        self._bs1_decode_graph_pool = None
        self._bs1_decode_logits = None
        self._bs1_next_token = None
        self._bs1_decode_graph_attempted = set()

        # Handle both torch.dtype and string representations
        dtype = model_config.torch_dtype
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
        dtype = model_config.torch_dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""))
        torch.set_default_dtype(dtype)
        # RWKV replaces most large parameter storages during load_pth().
        # Build the module skeleton on CPU to avoid preallocating dead CUDA buffers.
        torch.set_default_device("cpu")
        self.model = RWKV7ForCausalLM(model_config)
        # RWKV post-load quantization and sizing need runtime config knobs
        # (e.g. rwkv_int8_*), not just the shape config.
        self.model.config = config
        torch.set_default_device("cuda")
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # Allocate cache before warmup for RWKV (state cache is required for forward)
        self.allocate_state_cache()
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

            for attr in ("state_cache", "token_shift_cache", "model", "sampler"):
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

    def allocate_state_cache(self):
        config = self.config
        model_config = config.model_config
        num_heads = model_config.num_heads // self.world_size
        head_dim = getattr(model_config, "head_dim", model_config.hidden_size // model_config.num_heads)
        block_bytes = model_config.num_hidden_layers * (head_dim + 2) * num_heads * head_dim * model_config.torch_dtype.itemsize
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
        self.state_cache = torch.zeros(model_config.num_hidden_layers, config.num_state_blocks, num_heads, head_dim, head_dim)
        self.token_shift_cache = torch.zeros(2, model_config.num_hidden_layers, config.num_state_blocks, model_config.hidden_size)
        self.bind_state_cache_modules(self.state_cache, self.token_shift_cache)
        target_model = getattr(self.model, "model", self.model)
        if hasattr(target_model, "decode_tokenshift_scratch"):
            target_model.decode_tokenshift_scratch = torch.empty(
                config.max_num_seqs,
                model_config.hidden_size,
                dtype=model_config.torch_dtype,
                device=self.state_cache.device,
            )

    def warmup_int8_kernels(self):
        if not torch.cuda.is_available():
            return
        dtype = self.config.model_config.torch_dtype
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
                in_features = self.config.model_config.hidden_size
                x = torch.zeros((1, in_features), device=lm_head.qweight.device, dtype=dtype)
                _ = lm_head(x)
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
        model_config = self.config.model_config
        probe_state_cache = torch.zeros(model_config.num_hidden_layers, batch_size, num_heads, head_dim, head_dim)
        probe_token_shift_cache = torch.zeros(2, model_config.num_hidden_layers, batch_size, model_config.hidden_size)
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

    def _reset_state_cache_slots_for_prefill(self, seqs: list[Sequence]) -> None:
        if not seqs:
            return
        fresh_slots = sorted({
            int(seq.block_table[0])
            for seq in seqs
            if seq.num_cached_tokens == 0 and seq.block_table
        })
        if not fresh_slots:
            return
        blocks = getattr(getattr(self.model, "model", None), "blocks", None)
        if blocks is None or len(blocks) == 0:
            return
        slot_ids = torch.tensor(
            fresh_slots,
            dtype=torch.int64,
            device=blocks[0].att.state_cache.device,
        )
        for block in blocks:
            block.att.state_cache.index_fill_(0, slot_ids, 0)
            block.att.att_tokenshift_cache.index_fill_(0, slot_ids, 0)
            block.ffn.ffn_tokenshift_cache.index_fill_(0, slot_ids, 0)

    def prepare_prefill(self, seqs: list[Sequence]):
        self._reset_state_cache_slots_for_prefill(seqs)
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

    def prepare_decode(self, seqs: list[Sequence]):
        if len(seqs) == 1:
            seq = seqs[0]
            if self._bs1_decode_tensors is None:
                self._bs1_decode_tensors = dict(
                    input_ids=torch.empty(1, dtype=torch.int64, device="cuda"),
                    positions=torch.empty(1, dtype=torch.int64, device="cuda"),
                    slot_mapping_in=torch.empty(1, dtype=torch.int32, device="cuda"),
                    slot_mapping_out=torch.empty(1, dtype=torch.int32, device="cuda"),
                    context_lens=torch.empty(1, dtype=torch.int32, device="cuda"),
                )
            cached = self._bs1_decode_tensors
            cached["input_ids"][0] = seq.last_token
            cached["positions"][0] = len(seq) - 1
            cached["slot_mapping_in"][0] = seq.block_table[0]
            cached["slot_mapping_out"][0] = seq.block_table[0]
            cached["context_lens"][0] = len(seq)
            set_context(
                False,
                context_lens=cached["context_lens"],
                slot_mapping_in=cached["slot_mapping_in"],
                slot_mapping_out=cached["slot_mapping_out"],
            )
            return cached["input_ids"], cached["positions"]
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

    def prepare_decode_single(self, seq: Sequence):
        input_ids, positions = self.prepare_decode([seq])
        temperatures = self.prepare_sample([seq]) if self.rank == 0 else None
        self._ensure_bs1_decode_graph(input_ids, positions, temperatures is None)
        return input_ids, positions, temperatures

    def _prepare_decode_single_slots(
        self,
        last_token: int,
        position: int,
        context_len: int,
        slot_in: int,
        slot_out: int,
        temperature: float = 0.0,
    ):
        if slot_in != slot_out:
            # For bs=1, preserving the prefill state only matters on the first
            # decode step. Copy once to the output slot, then keep decoding
            # in-place on that slot instead of maintaining a true ping-pong path.
            self._copy_bs1_decode_state(slot_in, slot_out)
            slot_in = slot_out
        if self._bs1_decode_tensors is None:
            self._bs1_decode_tensors = dict(
                input_ids=torch.empty(1, dtype=torch.int64, device="cuda"),
                positions=torch.empty(1, dtype=torch.int64, device="cuda"),
                slot_mapping_in=torch.empty(1, dtype=torch.int32, device="cuda"),
                slot_mapping_out=torch.empty(1, dtype=torch.int32, device="cuda"),
                context_lens=torch.empty(1, dtype=torch.int32, device="cuda"),
            )
        cached = self._bs1_decode_tensors
        cached["input_ids"][0] = last_token
        cached["positions"][0] = position
        cached["slot_mapping_in"][0] = slot_in
        cached["slot_mapping_out"][0] = slot_out
        cached["context_lens"][0] = context_len
        set_context(
            False,
            context_lens=cached["context_lens"],
            slot_mapping_in=cached["slot_mapping_in"],
            slot_mapping_out=cached["slot_mapping_out"],
        )
        temperatures = None
        if self.rank == 0 and temperature > 1e-10:
            if self._bs1_temperature is None:
                self._bs1_temperature = torch.empty(1, dtype=torch.float32, device="cuda")
            self._bs1_temperature[0] = temperature
            temperatures = self._bs1_temperature
        self._ensure_bs1_decode_graph(cached["input_ids"], cached["positions"], temperatures is None)
        return cached["input_ids"], cached["positions"], temperatures

    def _snapshot_bs1_decode_state(self, slot_in: int, slot_out: int):
        slots = (slot_in,) if slot_in == slot_out else (slot_in, slot_out)
        blocks = getattr(getattr(self.model, "model", None), "blocks", None)
        if blocks is None:
            return None
        snapshots = []
        for block in blocks:
            att_snap = {slot: block.att.att_tokenshift_cache[slot].clone() for slot in slots}
            state_snap = {slot: block.att.state_cache[slot].clone() for slot in slots}
            ffn_snap = {slot: block.ffn.ffn_tokenshift_cache[slot].clone() for slot in slots}
            snapshots.append((att_snap, state_snap, ffn_snap))
        return snapshots

    def _restore_bs1_decode_state(self, snapshots):
        if snapshots is None:
            return
        for block, (att_snap, state_snap, ffn_snap) in zip(self.model.model.blocks, snapshots):
            for slot, value in att_snap.items():
                block.att.att_tokenshift_cache[slot].copy_(value)
            for slot, value in state_snap.items():
                block.att.state_cache[slot].copy_(value)
            for slot, value in ffn_snap.items():
                block.ffn.ffn_tokenshift_cache[slot].copy_(value)

    def _copy_bs1_decode_state(self, slot_in: int, slot_out: int):
        if slot_in == slot_out:
            return
        self.state_cache[:, slot_out].copy_(self.state_cache[:, slot_in])
        self.token_shift_cache[:, :, slot_out].copy_(self.token_shift_cache[:, :, slot_in])

    def _ensure_bs1_decode_graph(self, input_ids: torch.Tensor, positions: torch.Tensor, greedy_only: bool):
        if self.world_size != 1:
            return
        cached = self._bs1_decode_tensors
        if cached is None:
            return
        slot_in = int(cached["slot_mapping_in"][0].item())
        slot_out = int(cached["slot_mapping_out"][0].item())
        if greedy_only and (slot_in, slot_out, False) in self._bs1_decode_graphs:
            return
        key = (slot_in, slot_out, greedy_only)
        if key in self._bs1_decode_graphs or key in self._bs1_decode_graph_attempted:
            return
        self._bs1_decode_graph_attempted.add(key)
        state_snapshot = self._snapshot_bs1_decode_state(slot_in, slot_out)
        with torch.inference_mode():
            set_context(
                False,
                force_contiguous_decode=True,
                contiguous_decode_slot_in_start=slot_in,
                contiguous_decode_slot_out_start=slot_out,
                contiguous_decode_slot_count=int(cached["slot_mapping_in"].numel()),
                context_lens=cached["context_lens"],
                slot_mapping_in=cached["slot_mapping_in"],
                slot_mapping_out=cached["slot_mapping_out"],
            )
            try:
                logits = self.model.forward_one_logits(input_ids, positions)
                self._restore_bs1_decode_state(state_snapshot)
                if self._bs1_next_token is None:
                    self._bs1_next_token = torch.empty(1, dtype=torch.int64, device=input_ids.device)
                if not greedy_only:
                    self._bs1_decode_logits = logits.clone()
                self._bs1_next_token.copy_(logits.argmax(dim=-1))
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self._bs1_decode_graph_pool):
                    logits = self.model.forward_one_logits(input_ids, positions)
                    if not greedy_only:
                        self._bs1_decode_logits.copy_(logits)
                    self._bs1_next_token.copy_(logits.argmax(dim=-1))
                self._bs1_decode_graphs[key] = graph
                if self._bs1_decode_graph_pool is None:
                    self._bs1_decode_graph_pool = graph.pool()
                self._restore_bs1_decode_state(state_snapshot)
            except Exception:
                self._bs1_decode_graphs.pop(key, None)
                self._restore_bs1_decode_state(state_snapshot)
            finally:
                set_context(
                    False,
                    context_lens=cached["context_lens"],
                    slot_mapping_in=cached["slot_mapping_in"],
                    slot_mapping_out=cached["slot_mapping_out"],
                )

    def decode_single_step(
        self,
        seq: Sequence,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        temperatures: torch.Tensor | None,
        record_sequence: bool = True,
    ):
        cached = self._bs1_decode_tensors
        slot_in = slot_out = None
        if cached is not None:
            slot_in = int(cached["slot_mapping_in"][0].item())
            slot_out = int(cached["slot_mapping_out"][0].item())
        graph_key = None
        if slot_in is not None and slot_out is not None:
            if temperatures is None:
                if (slot_in, slot_out, True) in self._bs1_decode_graphs:
                    graph_key = (slot_in, slot_out, True)
                elif (slot_in, slot_out, False) in self._bs1_decode_graphs:
                    graph_key = (slot_in, slot_out, False)
            else:
                if (slot_in, slot_out, False) in self._bs1_decode_graphs:
                    graph_key = (slot_in, slot_out, False)
        use_bs1_graph = graph_key is not None
        if use_bs1_graph:
            self._bs1_decode_graphs[graph_key].replay()
            logits = None if graph_key[2] and temperatures is None else self._bs1_decode_logits
        else:
            logits = self.model.forward_one_logits(input_ids, positions)
        if self.rank == 0:
            if temperatures is None:
                token = self._bs1_next_token if self._bs1_next_token is not None else logits.argmax(dim=-1)
            else:
                token = self.sampler(logits, temperatures)
        else:
            token = None
        if self.rank == 0 and record_sequence:
            token_id = int(token.item())
            seq.append_token(token_id)
            return token_id
        return token

    def run_decode_only_single(self, seq: Sequence, decode_steps: int) -> int:
        input_ids, positions, temperatures = self.prepare_decode_single(seq)
        next_token = torch.tensor([seq.last_token], device=input_ids.device, dtype=input_ids.dtype)
        context_lens = self._bs1_decode_tensors["context_lens"]
        steps = 0
        while steps < decode_steps:
            input_ids[0] = next_token[0]
            next_token = self.decode_single_step(seq, input_ids, positions, temperatures, record_sequence=False)
            positions.add_(1)
            context_lens.add_(1)
            steps += 1
        reset_context()
        return steps

    def prepare_sample(self, seqs: list[Sequence]):
        if len(seqs) == 1:
            if seqs[0].temperature <= 1e-10:
                return None
            if self._bs1_temperature is None:
                self._bs1_temperature = torch.empty(1, dtype=torch.float32, device="cuda")
            self._bs1_temperature[0] = seqs[0].temperature
            return self._bs1_temperature
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        return self.model.forward_logits(input_ids, positions)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if not is_prefill and len(seqs) == 1:
            seq = seqs[0]
            input_ids, positions, temperatures = self.prepare_decode_single(seq)
            token = self.decode_single_step(seq, input_ids, positions, temperatures, record_sequence=False)
            reset_context()
            if self.rank == 0:
                return [int(token.item())]
            return None
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids
