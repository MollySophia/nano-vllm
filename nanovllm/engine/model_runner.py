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
from nanovllm.layers.sampler import GREEDY_TEMPERATURE_EPS, Sampler
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
        self.state_slot_manager = None
        self.prefix_index = None

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
        self.sampler = Sampler(
            temperature_bucket_resolution=config.sampling_bucket_temperature_resolution,
            top_p_bucket_resolution=config.sampling_bucket_top_p_resolution,
        )
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

            for attr in ("state_cache", "token_shift_cache", "slot_last_hidden", "slot_last_hidden_valid", "model", "sampler"):
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

    def attach_state_cache(self, slot_manager, prefix_index):
        self.state_slot_manager = slot_manager
        self.prefix_index = prefix_index

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

        def compute_total_state_slots():
            free, total = torch.cuda.mem_get_info()
            reserve = total * (1 - config.gpu_memory_utilization)
            available = free - reserve - prefill_probe_bytes
            num_blocks = int(available) // block_bytes
            if config.max_state_slots != -1:
                num_blocks = min(num_blocks, config.max_state_slots)
            return num_blocks

        total_slots = compute_total_state_slots()
        if total_slots <= 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            total_slots = compute_total_state_slots()
        if total_slots <= 0:
            raise RuntimeError(
                f"Unable to allocate RWKV state cache: computed total_slots={total_slots}. "
                "Try lowering model memory pressure or increasing gpu_memory_utilization."
            )
        config.num_state_slots_total = total_slots
        if self.world_size == 1 and total_slots > 1 and not config.enforce_eager:
            config.bs1_graph_slot = total_slots - 1
            config.num_state_blocks = total_slots - 1
        else:
            config.bs1_graph_slot = -1
            config.num_state_blocks = total_slots
        self.state_cache = torch.zeros(model_config.num_hidden_layers, config.num_state_slots_total, num_heads, head_dim, head_dim)
        self.token_shift_cache = torch.zeros(2, model_config.num_hidden_layers, config.num_state_slots_total, model_config.hidden_size)
        if config.rwkv_state_cache_enable:
            self.slot_last_hidden = torch.zeros(config.num_state_slots_total, model_config.hidden_size)
            self.slot_last_hidden_valid = torch.zeros(config.num_state_slots_total, dtype=torch.bool)
        else:
            for attr in ("slot_last_hidden", "slot_last_hidden_valid"):
                if hasattr(self, attr):
                    delattr(self, attr)
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
            if self.config.rwkv_state_cache_enable:
                seq.prompt_cache_slot = block_id
                seq.cached_prefix_len = 0
            else:
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
        if self.config.rwkv_state_cache_enable:
            fresh_slots = sorted({
                int(seq.prompt_cache_slot)
                for seq in seqs
                if seq.cached_prefix_len == 0 and seq.prompt_cache_slot is not None
            })
        else:
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
        if hasattr(self, "slot_last_hidden"):
            self.slot_last_hidden.index_fill_(0, slot_ids, 0)
        if hasattr(self, "slot_last_hidden_valid"):
            self.slot_last_hidden_valid.index_fill_(0, slot_ids, False)

    def _seq_slot_for_decode(self, seq: Sequence) -> int:
        if seq.active_state_slot is not None:
            return int(seq.active_state_slot)
        if self.config.rwkv_state_cache_enable:
            assert seq.state_slot is not None
            return int(seq.state_slot)
        assert seq.block_table
        return int(seq.block_table[0])

    def _shared_bs1_graph_slot(self) -> int | None:
        if self.config.enforce_eager:
            return None
        slot = getattr(self.config, "bs1_graph_slot", -1)
        if self.world_size != 1 or slot is None or int(slot) < 0:
            return None
        return int(slot)

    def prepare_prefill(self, seqs: list[Sequence]):
        self._reset_state_cache_slots_for_prefill(seqs)
        input_rows = []
        position_rows = []
        slot_mapping_in = []
        slot_mapping_out = []
        context_lens = []
        if self.config.rwkv_state_cache_enable:
            max_seqlen = max(seq.num_prompt_tokens - seq.cached_prefix_len for seq in seqs)
        else:
            max_seqlen = max(len(seq) - seq.num_cached_tokens for seq in seqs)
        for seq in seqs:
            if self.config.rwkv_state_cache_enable:
                new_token_ids = seq.prompt_token_ids[seq.cached_prefix_len:]
                start_pos = seq.cached_prefix_len
                slot_in = seq.cache_hit_slot if seq.cache_hit_slot is not None else seq.prompt_cache_slot
                slot_out = seq.prompt_cache_slot
            else:
                new_token_ids = seq[seq.num_cached_tokens:]
                start_pos = 0
                block_id = seq.block_table[0] if seq.block_table else 0
                slot_in = block_id
                slot_out = block_id
            seqlen = len(new_token_ids)
            pad_len = max_seqlen - seqlen
            input_rows.append([0] * pad_len + new_token_ids)
            position_rows.append([0] * pad_len + list(range(start_pos, start_pos + seqlen)))
            context_lens.append(seqlen)
            slot_mapping_in.append(slot_in)
            slot_mapping_out.append(slot_out)
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
            self._ensure_bs1_decode_tensors_mutable()
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
            if self.config.rwkv_state_cache_enable and not seq.state_slot_materialized:
                assert seq.prompt_cache_slot is not None and seq.state_slot is not None
                cached["slot_mapping_in"][0] = int(seq.prompt_cache_slot)
                cached["slot_mapping_out"][0] = int(seq.state_slot)
            else:
                slot_id = self._seq_slot_for_decode(seq)
                cached["slot_mapping_in"][0] = slot_id
                cached["slot_mapping_out"][0] = slot_id
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
            if self.config.rwkv_state_cache_enable and not seq.state_slot_materialized:
                assert seq.prompt_cache_slot is not None and seq.state_slot is not None
                slot_mapping_in.append(int(seq.prompt_cache_slot))
                slot_mapping_out.append(int(seq.state_slot))
            else:
                slot_id = self._seq_slot_for_decode(seq)
                slot_mapping_in.append(slot_id)
                slot_mapping_out.append(slot_id)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_in = torch.tensor(slot_mapping_in, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_out = torch.tensor(slot_mapping_out, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(False, context_lens=context_lens, slot_mapping_in=slot_mapping_in, slot_mapping_out=slot_mapping_out)
        return input_ids, positions

    def prepare_decode_single(self, seq: Sequence):
        graph_slot = self._shared_bs1_graph_slot()
        if graph_slot is not None and seq.active_state_slot is None:
            if self.config.rwkv_state_cache_enable and not seq.state_slot_materialized:
                assert seq.prompt_cache_slot is not None
                source_slot = int(seq.prompt_cache_slot)
            else:
                source_slot = self._seq_slot_for_decode(seq)
            return self._prepare_decode_single_slots(
                last_token=seq.last_token,
                position=len(seq) - 1,
                context_len=len(seq),
                slot_in=source_slot,
                slot_out=graph_slot,
                temperature=seq.temperature,
                copy_input_state=(source_slot != graph_slot),
                prepare_graph=True,
            )
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
        copy_input_state: bool = True,
        prepare_graph: bool = True,
    ):
        if copy_input_state and slot_in != slot_out:
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
        if self.rank == 0 and temperature > GREEDY_TEMPERATURE_EPS:
            if self._bs1_temperature is None:
                self._bs1_temperature = torch.empty(1, dtype=torch.float32, device="cuda")
            self._bs1_temperature[0] = temperature
            temperatures = self._bs1_temperature
        if prepare_graph:
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

    def _ensure_bs1_decode_tensors_mutable(self) -> None:
        if self._bs1_decode_tensors is None:
            return
        sample = next(iter(self._bs1_decode_tensors.values()))
        is_inference = getattr(sample, "is_inference", None)
        if callable(is_inference) and is_inference():
            self._bs1_decode_tensors = {name: tensor.clone() for name, tensor in self._bs1_decode_tensors.items()}

    def _invalidate_bs1_slot_graphs(self, slot_id: int) -> None:
        keys = [key for key in self._bs1_decode_graphs if key[0] == slot_id and key[1] == slot_id]
        for key in keys:
            self._bs1_decode_graphs.pop(key, None)
        attempted = [key for key in self._bs1_decode_graph_attempted if key[0] == slot_id and key[1] == slot_id]
        for key in attempted:
            self._bs1_decode_graph_attempted.discard(key)

    def _copy_slot_states(self, slot_ins: list[int], slot_outs: list[int], copy_last_hidden: bool = False):
        if not slot_ins:
            return
        if len(slot_ins) == 1:
            src = int(slot_ins[0])
            dst = int(slot_outs[0])
            if src != dst:
                self.state_cache[:, dst].copy_(self.state_cache[:, src])
                self.token_shift_cache[:, :, dst].copy_(self.token_shift_cache[:, :, src])
                if copy_last_hidden:
                    self.slot_last_hidden[dst].copy_(self.slot_last_hidden[src])
                    self.slot_last_hidden_valid[dst] = self.slot_last_hidden_valid[src]
            return
        src_index = torch.tensor(slot_ins, dtype=torch.int64, device=self.state_cache.device)
        dst_index = torch.tensor(slot_outs, dtype=torch.int64, device=self.state_cache.device)
        state = self.state_cache.index_select(1, src_index)
        token_shift = self.token_shift_cache.index_select(2, src_index)
        self.state_cache.index_copy_(1, dst_index, state)
        self.token_shift_cache.index_copy_(2, dst_index, token_shift)
        if copy_last_hidden:
            last_hidden = self.slot_last_hidden.index_select(0, src_index)
            last_hidden_valid = self.slot_last_hidden_valid.index_select(0, src_index)
            self.slot_last_hidden.index_copy_(0, dst_index, last_hidden)
            self.slot_last_hidden_valid.index_copy_(0, dst_index, last_hidden_valid)

    def _store_slot_last_hidden(self, slot_id: int, hidden: torch.Tensor) -> None:
        self.slot_last_hidden[slot_id].copy_(hidden)
        self.slot_last_hidden_valid[slot_id] = True

    def _publish_cached_slot(self, slot_id: int, token_ids: list[int], prefix_len: int, hidden: torch.Tensor) -> None:
        self._store_slot_last_hidden(slot_id, hidden)
        if self.rank != 0 or self.state_slot_manager is None or self.prefix_index is None:
            return
        cache_key = self.prefix_index.insert(token_ids, prefix_len, slot_id)
        self.state_slot_manager.mark_cached(slot_id, cache_key, prefix_len)

    def _would_finish_after_token(self, seq: Sequence, token_id: int) -> bool:
        return ((not seq.ignore_eos and token_id == self.eos) or (seq.num_completion_tokens + 1 == seq.max_tokens))

    def _forward_hidden_one_token(
        self,
        token_id: int,
        position: int,
        context_len: int,
        slot_in: int,
        slot_out: int | None = None,
    ) -> torch.Tensor:
        if slot_out is None:
            slot_out = slot_in
        input_ids, positions, _ = self._prepare_decode_single_slots(
            last_token=token_id,
            position=position,
            context_len=context_len,
            slot_in=slot_in,
            slot_out=slot_out,
            copy_input_state=False,
            prepare_graph=False,
        )
        try:
            hidden = self.model.model.forward_one(input_ids, positions)
        finally:
            reset_context()
        return hidden

    def _finalize_finished_sequence_cache(self, seq: Sequence, token_id: int) -> None:
        if seq.state_slot is None:
            return
        if seq.active_state_slot is not None:
            slot_in = int(seq.active_state_slot)
            slot_out = int(seq.state_slot)
        elif seq.state_slot_materialized:
            slot_in = int(seq.state_slot)
            slot_out = int(seq.state_slot)
        else:
            assert seq.prompt_cache_slot is not None
            slot_in = int(seq.prompt_cache_slot)
            slot_out = int(seq.state_slot)
        hidden = self._forward_hidden_one_token(token_id, len(seq), len(seq) + 1, slot_in, slot_out)
        if not seq.state_slot_materialized:
            if self.rank == 0 and self.state_slot_manager is not None and seq.prompt_cache_slot is not None:
                self.state_slot_manager.unpin_cached(seq.prompt_cache_slot)
            seq.state_slot_materialized = True
        self._publish_cached_slot(
            seq.state_slot,
            seq.token_ids + [token_id],
            len(seq) + 1,
            hidden[0],
        )
        seq.final_cache_published = True

    def _after_bs1_decode_step(self, seq: Sequence) -> None:
        graph_slot = self._shared_bs1_graph_slot()
        if graph_slot is not None and seq.active_state_slot is None:
            seq.active_state_slot = graph_slot
        if not self.config.rwkv_state_cache_enable or seq.state_slot_materialized:
            return
        if self.rank == 0 and self.state_slot_manager is not None and seq.prompt_cache_slot is not None:
            self.state_slot_manager.unpin_cached(seq.prompt_cache_slot)
        if seq.state_slot is not None:
            self._invalidate_bs1_slot_graphs(int(seq.state_slot))
            if seq.active_state_slot is None:
                seq.active_state_slot = int(seq.state_slot)
        seq.state_slot_materialized = True

    def _ensure_bs1_decode_graph(self, input_ids: torch.Tensor, positions: torch.Tensor, greedy_only: bool):
        if self.config.enforce_eager:
            return
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
                token = self.sampler(logits, [seq])
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
            if seqs[0].temperature <= GREEDY_TEMPERATURE_EPS:
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

    @torch.inference_mode()
    def _compute_prefill_logits_with_state_cache(self, seqs: list[Sequence]) -> torch.Tensor:
        seq_to_row: dict[int, torch.Tensor] = {}
        exact_seqs = [seq for seq in seqs if seq.exact_cache_hit]
        prefill_seqs = [seq for seq in seqs if not seq.exact_cache_hit]

        if prefill_seqs:
            input_ids, positions = self.prepare_prefill(prefill_seqs)
            hidden_states = self.model(input_ids, positions)
            logits = self.model.compute_logits(hidden_states)
            last_hidden = hidden_states[:, -1, :]
            for row, seq in enumerate(prefill_seqs):
                self._publish_cached_slot(seq.prompt_cache_slot, seq.prompt_token_ids, seq.num_prompt_tokens, last_hidden[row])
                seq_to_row[seq.seq_id] = logits[row]
                seq.state_slot_materialized = False
                if self.rank == 0 and self.state_slot_manager is not None and seq.prompt_cache_slot is not None:
                    self.state_slot_manager.pin_cached(seq.prompt_cache_slot)
                if self.rank == 0 and self.state_slot_manager is not None and seq.cache_hit_slot is not None:
                    self.state_slot_manager.unpin_cached(seq.cache_hit_slot)
                    seq.cache_hit_slot = None

        if exact_seqs:
            valid = self.slot_last_hidden_valid.index_select(
                0,
                torch.tensor([int(seq.prompt_cache_slot) for seq in exact_seqs], dtype=torch.int64, device=self.state_cache.device),
            )
            if not bool(valid.all().item()):
                raise RuntimeError("Exact RWKV cache hit is missing slot_last_hidden.")
            hidden = self.slot_last_hidden.index_select(
                0,
                torch.tensor([int(seq.prompt_cache_slot) for seq in exact_seqs], dtype=torch.int64, device=self.state_cache.device),
            )
            logits = self.model.compute_logits(hidden)
            for row, seq in enumerate(exact_seqs):
                seq_to_row[seq.seq_id] = logits[row]
                seq.state_slot_materialized = False

        ordered_logits = torch.stack([seq_to_row[seq.seq_id] for seq in seqs], dim=0)
        reset_context()
        return ordered_logits

    @torch.inference_mode()
    def _compute_decode_logits_with_state_cache(self, seqs: list[Sequence]) -> torch.Tensor:
        if len(seqs) == 1:
            seq = seqs[0]
            if not seq.state_slot_materialized:
                assert seq.prompt_cache_slot is not None and seq.state_slot is not None
                input_ids, positions, _ = self._prepare_decode_single_slots(
                    last_token=seq.last_token,
                    position=len(seq) - 1,
                    context_len=len(seq),
                    slot_in=int(seq.prompt_cache_slot),
                    slot_out=int(seq.state_slot),
                    copy_input_state=False,
                    prepare_graph=False,
                )
            else:
                input_ids, positions = self.prepare_decode([seq])
            logits = self.model.forward_one_logits(input_ids, positions)
            if not seq.state_slot_materialized:
                if self.rank == 0 and self.state_slot_manager is not None and seq.prompt_cache_slot is not None:
                    self.state_slot_manager.unpin_cached(seq.prompt_cache_slot)
                self._invalidate_bs1_slot_graphs(int(seq.state_slot))
                seq.state_slot_materialized = True
            reset_context()
            return logits
        input_ids, positions = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, positions, False)
        for seq in seqs:
            if seq.state_slot_materialized:
                continue
            if self.rank == 0 and self.state_slot_manager is not None and seq.prompt_cache_slot is not None:
                self.state_slot_manager.unpin_cached(seq.prompt_cache_slot)
            self._invalidate_bs1_slot_graphs(int(seq.state_slot))
            seq.state_slot_materialized = True
        reset_context()
        return logits

    @torch.inference_mode()
    def run_logits(self, seqs: list[Sequence], is_prefill: bool):
        if self.config.rwkv_state_cache_enable:
            if is_prefill:
                return self._compute_prefill_logits_with_state_cache(seqs)
            return self._compute_decode_logits_with_state_cache(seqs)
        if not is_prefill and len(seqs) == 1:
            seq = seqs[0]
            input_ids, positions = self.prepare_decode([seq])
            logits = self.model.forward_one_logits(input_ids, positions)
            reset_context()
            return logits
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        reset_context()
        return logits

    def prepare_postprocess(self, seqs: list[Sequence], token_ids: list[int] | None) -> None:
        if token_ids is None or self.rank != 0 or not self.config.rwkv_state_cache_enable:
            return
        for seq, token_id in zip(seqs, token_ids):
            if self._would_finish_after_token(seq, token_id):
                self._finalize_finished_sequence_cache(seq, token_id)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if not is_prefill and len(seqs) == 1:
            seq = seqs[0]
            input_ids, positions, temperatures = self.prepare_decode_single(seq)
            token = self.decode_single_step(seq, input_ids, positions, temperatures, record_sequence=False)
            self._after_bs1_decode_step(seq)
            reset_context()
            if self.rank == 0:
                token_ids = [int(token.item())]
            else:
                token_ids = None
            self.prepare_postprocess(seqs, token_ids)
            return token_ids
        logits = self.run_logits(seqs, is_prefill)
        token_ids = self.sampler(logits, seqs).tolist() if self.rank == 0 else None
        self.prepare_postprocess(seqs, token_ids)
        return token_ids
