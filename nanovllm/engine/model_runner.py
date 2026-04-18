import pickle
import gc
import os
import time

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.linear import MarlinInt8Linear
from nanovllm.models.rwkv7 import RWKV7ForCausalLM
from nanovllm.layers.sampler import GREEDY_TEMPERATURE_EPS, Sampler
from nanovllm.utils.context import get_context, set_context, reset_context
from nanovllm.utils.loader import load_model


def _bs1_requires_sequence_sampler(*, temperature: float, presence_penalty: float, repetition_penalty: float) -> bool:
    return (
        temperature > GREEDY_TEMPERATURE_EPS
        or presence_penalty != 0.0
        or repetition_penalty != 0.0
    )


class ModelRunProfiler:
    def __init__(self, label: str = ""):
        self.label = label or "default"
        self.started_at = time.perf_counter()
        self.step_counts = {"decode": 0, "prefill": 0}
        self.seq_totals = {"decode": 0, "prefill": 0}
        self.total_s = {"decode": 0.0, "prefill": 0.0}
        self.prepare_s = {"decode": 0.0, "prefill": 0.0}
        self.forward_s = {"decode": 0.0, "prefill": 0.0}
        self.sample_s = {"decode": 0.0, "prefill": 0.0}
        self.post_s = {"decode": 0.0, "prefill": 0.0}
        self.prefill_exec_batches = 0
        self.prefill_logical_tokens = 0
        self.prefill_flat_padded_tokens = 0
        self.prefill_bucketed_padded_tokens = 0

    def record_step(
        self,
        *,
        kind: str,
        seq_count: int,
        total_s: float,
        prepare_s: float,
        forward_s: float,
        sample_s: float,
        post_s: float,
        prefill_exec_batches: int = 0,
        prefill_logical_tokens: int = 0,
        prefill_flat_padded_tokens: int = 0,
        prefill_bucketed_padded_tokens: int = 0,
    ) -> None:
        self.step_counts[kind] += 1
        self.seq_totals[kind] += seq_count
        self.total_s[kind] += total_s
        self.prepare_s[kind] += prepare_s
        self.forward_s[kind] += forward_s
        self.sample_s[kind] += sample_s
        self.post_s[kind] += post_s
        if kind == "prefill":
            self.prefill_exec_batches += prefill_exec_batches
            self.prefill_logical_tokens += prefill_logical_tokens
            self.prefill_flat_padded_tokens += prefill_flat_padded_tokens
            self.prefill_bucketed_padded_tokens += prefill_bucketed_padded_tokens

    def emit_report(self) -> None:
        def _avg_ms(total_s: float, count: int) -> float:
            if count <= 0:
                return 0.0
            return total_s * 1000.0 / count

        def _avg_bsz(kind: str) -> float:
            count = self.step_counts[kind]
            if count <= 0:
                return 0.0
            return self.seq_totals[kind] / count

        wall_s = time.perf_counter() - self.started_at
        print(
            f"[model-run-profile] label={self.label} wall_s={wall_s:.3f}",
            flush=True,
        )
        for kind in ("decode", "prefill"):
            count = self.step_counts[kind]
            other_s = self.total_s[kind] - self.prepare_s[kind] - self.forward_s[kind] - self.sample_s[kind] - self.post_s[kind]
            extra = ""
            if kind == "prefill":
                flat_amp = self.prefill_flat_padded_tokens / max(1, self.prefill_logical_tokens)
                bucketed_amp = self.prefill_bucketed_padded_tokens / max(1, self.prefill_logical_tokens)
                exec_batches_per_step = self.prefill_exec_batches / count if count > 0 else 0.0
                logical_tokens_per_step = self.prefill_logical_tokens / count if count > 0 else 0.0
                flat_padded_tokens_per_step = self.prefill_flat_padded_tokens / count if count > 0 else 0.0
                bucketed_padded_tokens_per_step = self.prefill_bucketed_padded_tokens / count if count > 0 else 0.0
                extra = (
                    f" prefill_exec_batches_per_step={exec_batches_per_step:.2f} "
                    f"prefill_logical_tokens_per_step={logical_tokens_per_step:.2f} "
                    f"prefill_flat_padded_tokens_per_step={flat_padded_tokens_per_step:.2f} "
                    f"prefill_bucketed_padded_tokens_per_step={bucketed_padded_tokens_per_step:.2f} "
                    f"prefill_flat_padding_amp={flat_amp:.3f} "
                    f"prefill_bucketed_padding_amp={bucketed_amp:.3f}"
                )
            print(
                "[model-run-profile] "
                f"{kind}_steps count={count} avg_bsz={_avg_bsz(kind):.2f} "
                f"total_ms_per_step={_avg_ms(self.total_s[kind], count):.3f} "
                f"prepare_ms_per_step={_avg_ms(self.prepare_s[kind], count):.3f} "
                f"forward_ms_per_step={_avg_ms(self.forward_s[kind], count):.3f} "
                f"sample_ms_per_step={_avg_ms(self.sample_s[kind], count):.3f} "
                f"post_ms_per_step={_avg_ms(self.post_s[kind], count):.3f} "
                f"other_ms_per_step={_avg_ms(other_s, count):.3f}"
                f"{extra}",
                flush=True,
            )


def _build_prefill_bucket_plan(step_lengths: list[int]) -> tuple[list[list[int]], int, int, int]:
    if not step_lengths:
        return [], 0, 0, 0
    buckets_by_length: dict[int, list[int]] = {}
    for index, step_tokens in enumerate(step_lengths):
        if step_tokens <= 0:
            continue
        buckets_by_length.setdefault(step_tokens, []).append(index)
    buckets = [buckets_by_length[length] for length in sorted(buckets_by_length, reverse=True)]
    positive_lengths = [step_tokens for step_tokens in step_lengths if step_tokens > 0]
    logical_tokens = sum(step_lengths)
    flat_padded_tokens = (max(positive_lengths) * len(positive_lengths)) if positive_lengths else 0
    bucketed_padded_tokens = sum(len(bucket) * step_lengths[bucket[0]] for bucket in buckets)
    return buckets, logical_tokens, flat_padded_tokens, bucketed_padded_tokens


def _resolve_state_slot_layout(
    *,
    total_slots_capacity: int,
    requested_max_num_seqs: int,
    rwkv_state_cache_enable: bool,
    world_size: int,
    enforce_eager: bool,
) -> tuple[int, int, int, int]:
    if total_slots_capacity <= 0:
        raise ValueError("total_slots_capacity must be positive.")

    total_slots = total_slots_capacity
    if not rwkv_state_cache_enable and requested_max_num_seqs != -1:
        target_active_slots = min(requested_max_num_seqs, total_slots_capacity)
        reserve_graph_slot = (
            world_size == 1
            and not enforce_eager
            and total_slots_capacity > target_active_slots
        )
        total_slots = target_active_slots + int(reserve_graph_slot)

    if world_size == 1 and total_slots > 1 and not enforce_eager:
        bs1_graph_slot = total_slots - 1
        num_state_blocks = total_slots - 1
    else:
        bs1_graph_slot = -1
        num_state_blocks = total_slots

    if requested_max_num_seqs == -1:
        effective_max_num_seqs = num_state_blocks
    else:
        effective_max_num_seqs = min(requested_max_num_seqs, num_state_blocks)

    return total_slots, num_state_blocks, bs1_graph_slot, effective_max_num_seqs


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        model_config = config.model_config
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.eos = config.eos
        self.stop_token_seqs = tuple(tuple(seq) for seq in getattr(config, "stop_token_seqs", ()) if seq)
        self._run_profile = self._create_run_profiler()
        self._run_profile_prepare_s = 0.0
        self._run_profile_forward_s = 0.0
        self._run_profile_prefill_exec_batches = 0
        self._run_profile_prefill_logical_tokens = 0
        self._run_profile_prefill_flat_padded_tokens = 0
        self._run_profile_prefill_bucketed_padded_tokens = 0
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

    def _create_run_profiler(self) -> ModelRunProfiler | None:
        if self.rank != 0:
            return None
        raw = os.getenv("NANOVLLM_MODEL_RUN_PROFILE", os.getenv("NANOVLLM_BATCHER_PROFILE", ""))
        if raw.lower() in ("", "0", "false", "off", "no"):
            return None
        return ModelRunProfiler(label=os.getenv("NANOVLLM_MODEL_RUN_PROFILE_LABEL", os.getenv("NANOVLLM_BATCHER_PROFILE_LABEL", "")))

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
            if self._run_profile is not None:
                self._run_profile.emit_report()
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
            if config.rwkv_state_cache_safety_reserve_slots > 0:
                num_blocks -= config.rwkv_state_cache_safety_reserve_slots
            return num_blocks

        total_slots_capacity = compute_total_state_slots()
        if total_slots_capacity <= 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            total_slots_capacity = compute_total_state_slots()
        if total_slots_capacity <= 0:
            raise RuntimeError(
                f"Unable to allocate RWKV state cache: computed total_slots={total_slots_capacity}. "
                "Try lowering model memory pressure or increasing gpu_memory_utilization."
            )
        (
            total_slots,
            num_state_blocks,
            bs1_graph_slot,
            effective_max_num_seqs,
        ) = _resolve_state_slot_layout(
            total_slots_capacity=total_slots_capacity,
            requested_max_num_seqs=config.max_num_seqs,
            rwkv_state_cache_enable=config.rwkv_state_cache_enable,
            world_size=self.world_size,
            enforce_eager=config.enforce_eager,
        )
        config.num_state_slots_total = total_slots
        config.num_state_blocks = num_state_blocks
        config.bs1_graph_slot = bs1_graph_slot
        config.max_num_seqs = effective_max_num_seqs
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
                if seq.num_cached_tokens == 0 and seq.prompt_cache_slot is not None
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

    def _prefill_step_tokens(self, seq: Sequence) -> int:
        return seq.prefill_step_tokens(self.config.rwkv_prefill_chunk_size)

    def _prefill_bucket_plan(self, seqs: list[Sequence]) -> tuple[list[list[int]], int, int, int]:
        return _build_prefill_bucket_plan([self._prefill_step_tokens(seq) for seq in seqs])

    def prepare_prefill(self, seqs: list[Sequence]):
        self._reset_state_cache_slots_for_prefill(seqs)
        input_rows = []
        position_rows = []
        slot_mapping_in = []
        slot_mapping_out = []
        context_lens = []
        max_seqlen = max(self._prefill_step_tokens(seq) for seq in seqs)
        for seq in seqs:
            step_tokens = self._prefill_step_tokens(seq)
            if self.config.rwkv_state_cache_enable:
                chunk_end = seq.num_cached_tokens + step_tokens
                new_token_ids = seq.prompt_token_ids[seq.num_cached_tokens:chunk_end]
                start_pos = seq.num_cached_tokens
                if seq.num_cached_tokens == seq.cached_prefix_len and seq.cache_hit_slot is not None:
                    slot_in = seq.cache_hit_slot
                else:
                    slot_in = seq.prompt_cache_slot
                slot_out = seq.prompt_cache_slot
            else:
                chunk_end = seq.num_cached_tokens + step_tokens
                new_token_ids = seq.prompt_token_ids[seq.num_cached_tokens:chunk_end]
                start_pos = seq.num_cached_tokens
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
        use_sequence_sampler = _bs1_requires_sequence_sampler(
            temperature=seq.temperature,
            presence_penalty=seq.presence_penalty,
            repetition_penalty=seq.repetition_penalty,
        )
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
                force_sampler=use_sequence_sampler,
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
        force_sampler: bool = False,
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
        if self.rank == 0 and (force_sampler or temperature > GREEDY_TEMPERATURE_EPS):
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

    def _matches_stop_token_seq_after_token(self, seq: Sequence, token_id: int) -> bool:
        if not self.stop_token_seqs:
            return False
        next_completion_tokens = seq.num_raw_completion_tokens + 1
        for stop_seq in self.stop_token_seqs:
            stop_len = len(stop_seq)
            if next_completion_tokens < stop_len:
                continue
            if stop_len == 1:
                if token_id == stop_seq[0]:
                    return True
                continue
            if tuple(seq.token_ids[-(stop_len - 1):]) == stop_seq[:-1] and token_id == stop_seq[-1]:
                return True
        return False

    def _is_hidden_finalize_batch(self, seqs: list[Sequence]) -> bool:
        return self.config.rwkv_state_cache_enable and bool(seqs) and all(seq.pending_hidden_finalize for seq in seqs)

    def _store_decode_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)
        if not hasattr(self, "slot_last_hidden"):
            return hidden_states
        slot_mapping_out = get_context().slot_mapping_out
        if slot_mapping_out is None:
            return hidden_states
        slot_index = slot_mapping_out.to(dtype=torch.int64)
        self.slot_last_hidden.index_copy_(0, slot_index, hidden_states)
        self.slot_last_hidden_valid.index_fill_(0, slot_index, True)
        return hidden_states

    def _mark_state_slot_materialized(self, seq: Sequence, current_output_slot: int | None = None) -> None:
        if not self.config.rwkv_state_cache_enable or seq.state_slot_materialized:
            return
        if current_output_slot is not None and seq.active_state_slot is None:
            seq.active_state_slot = int(current_output_slot)
        if self.rank == 0 and self.state_slot_manager is not None and seq.prompt_cache_slot is not None:
            self.state_slot_manager.unpin_cached(seq.prompt_cache_slot)
        if seq.state_slot is not None:
            self._invalidate_bs1_slot_graphs(int(seq.state_slot))
            if seq.active_state_slot is None:
                seq.active_state_slot = int(seq.state_slot)
        seq.state_slot_materialized = True

    def _publish_hidden_finalize_sequence(self, seq: Sequence) -> None:
        if seq.state_slot is None:
            return
        if not seq.state_slot_materialized:
            self._mark_state_slot_materialized(seq)
        current_slot = self._seq_slot_for_decode(seq)
        target_slot = int(seq.state_slot)
        if current_slot != target_slot:
            self._copy_slot_states([current_slot], [target_slot], copy_last_hidden=True)
            current_slot = target_slot
        if not bool(self.slot_last_hidden_valid[current_slot].item()):
            raise RuntimeError("Hidden finalize is missing slot_last_hidden.")
        self._publish_cached_slot(
            target_slot,
            seq.token_ids,
            len(seq),
            self.slot_last_hidden[current_slot],
        )
        seq.final_cache_published = True

    def _after_bs1_decode_step(self, seq: Sequence) -> None:
        graph_slot = self._shared_bs1_graph_slot()
        if graph_slot is not None and seq.active_state_slot is None:
            seq.active_state_slot = graph_slot
        if not self.config.rwkv_state_cache_enable:
            return
        self._mark_state_slot_materialized(seq, seq.active_state_slot)

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
            seq = seqs[0]
            if not _bs1_requires_sequence_sampler(
                temperature=seq.temperature,
                presence_penalty=seq.presence_penalty,
                repetition_penalty=seq.repetition_penalty,
            ):
                return None
            if self._bs1_temperature is None:
                self._bs1_temperature = torch.empty(1, dtype=torch.float32, device="cuda")
            self._bs1_temperature[0] = seq.temperature
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
    def _compute_prefill_logits_with_state_cache(
        self,
        seqs: list[Sequence],
    ) -> tuple[torch.Tensor, float, float, int, int, int, int]:
        ordered_rows: list[torch.Tensor | None] = [None] * len(seqs)
        exact_items = [(index, seq) for index, seq in enumerate(seqs) if seq.exact_cache_hit]
        prefill_items = [(index, seq) for index, seq in enumerate(seqs) if not seq.exact_cache_hit]
        prepare_s = 0.0
        forward_s = 0.0
        prefill_exec_batches = 0
        prefill_logical_tokens = 0
        prefill_flat_padded_tokens = 0
        prefill_bucketed_padded_tokens = 0

        if prefill_items:
            prefill_indices = [index for index, _ in prefill_items]
            prefill_seqs = [seq for _, seq in prefill_items]
            (
                bucket_plan,
                prefill_logical_tokens,
                prefill_flat_padded_tokens,
                prefill_bucketed_padded_tokens,
            ) = self._prefill_bucket_plan(prefill_seqs)
            prefill_exec_batches = len(bucket_plan)
            for bucket in bucket_plan:
                bucket_indices = [prefill_indices[index] for index in bucket]
                bucket_seqs = [prefill_seqs[index] for index in bucket]
                prepare_started_at = time.perf_counter()
                input_ids, positions = self.prepare_prefill(bucket_seqs)
                prepare_s += time.perf_counter() - prepare_started_at
                try:
                    forward_started_at = time.perf_counter()
                    hidden_states = self.model(input_ids, positions)
                    logits = self.model.compute_logits(hidden_states)
                    forward_s += time.perf_counter() - forward_started_at
                    last_hidden = hidden_states[:, -1, :]
                    for row, (original_index, seq) in enumerate(zip(bucket_indices, bucket_seqs, strict=True)):
                        processed_prefix_len = seq.num_cached_tokens + self._prefill_step_tokens(seq)
                        self._publish_cached_slot(
                            seq.prompt_cache_slot,
                            seq.prompt_token_ids,
                            processed_prefix_len,
                            last_hidden[row],
                        )
                        ordered_rows[original_index] = logits[row]
                        seq.state_slot_materialized = False
                        if self.rank == 0 and self.state_slot_manager is not None and seq.prompt_cache_slot is not None:
                            self.state_slot_manager.pin_cached(seq.prompt_cache_slot)
                        if self.rank == 0 and self.state_slot_manager is not None and seq.cache_hit_slot is not None:
                            self.state_slot_manager.unpin_cached(seq.cache_hit_slot)
                            seq.cache_hit_slot = None
                finally:
                    reset_context()

        if exact_items:
            exact_indices = [index for index, _ in exact_items]
            exact_seqs = [seq for _, seq in exact_items]
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
            forward_started_at = time.perf_counter()
            logits = self.model.compute_logits(hidden)
            forward_s += time.perf_counter() - forward_started_at
            for row, (original_index, seq) in enumerate(zip(exact_indices, exact_seqs, strict=True)):
                ordered_rows[original_index] = logits[row]
                seq.state_slot_materialized = False

        ordered_logits = torch.stack(ordered_rows, dim=0)
        return (
            ordered_logits,
            prepare_s,
            forward_s,
            prefill_logical_tokens,
            prefill_flat_padded_tokens,
            prefill_bucketed_padded_tokens,
            prefill_exec_batches,
        )

    @torch.inference_mode()
    def _compute_prefill_logits_bucketed(
        self,
        seqs: list[Sequence],
    ) -> tuple[torch.Tensor, float, float, int, int, int, int]:
        (
            bucket_plan,
            prefill_logical_tokens,
            prefill_flat_padded_tokens,
            prefill_bucketed_padded_tokens,
        ) = self._prefill_bucket_plan(seqs)
        prepare_s = 0.0
        forward_s = 0.0
        if len(bucket_plan) == 1 and bucket_plan[0] == list(range(len(seqs))):
            prepare_started_at = time.perf_counter()
            input_ids, positions = self.prepare_prefill(seqs)
            prepare_s = time.perf_counter() - prepare_started_at
            try:
                forward_started_at = time.perf_counter()
                logits = self.run_model(input_ids, positions, True)
                forward_s = time.perf_counter() - forward_started_at
            finally:
                reset_context()
            return (
                logits,
                prepare_s,
                forward_s,
                prefill_logical_tokens,
                prefill_flat_padded_tokens,
                prefill_bucketed_padded_tokens,
                1,
            )

        ordered_rows: list[torch.Tensor | None] = [None] * len(seqs)
        for bucket in bucket_plan:
            bucket_seqs = [seqs[index] for index in bucket]
            prepare_started_at = time.perf_counter()
            input_ids, positions = self.prepare_prefill(bucket_seqs)
            prepare_s += time.perf_counter() - prepare_started_at
            try:
                forward_started_at = time.perf_counter()
                logits = self.run_model(input_ids, positions, True)
                forward_s += time.perf_counter() - forward_started_at
                for row, original_index in enumerate(bucket):
                    ordered_rows[original_index] = logits[row]
            finally:
                reset_context()
        return (
            torch.stack(ordered_rows, dim=0),
            prepare_s,
            forward_s,
            prefill_logical_tokens,
            prefill_flat_padded_tokens,
            prefill_bucketed_padded_tokens,
            len(bucket_plan),
        )

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
            hidden_states = self.model.model.forward_one(input_ids, positions)
            hidden_states = self._store_decode_hidden(hidden_states)
            logits = self.model.compute_logits(hidden_states)
            slot_mapping_out = get_context().slot_mapping_out
            current_output_slot = None
            if slot_mapping_out is not None and slot_mapping_out.numel() > 0:
                current_output_slot = int(slot_mapping_out[0].item())
            self._mark_state_slot_materialized(seq, current_output_slot)
            reset_context()
            return logits
        input_ids, positions = self.prepare_decode(seqs)
        hidden_states = self.model(input_ids, positions)
        hidden_states = self._store_decode_hidden(hidden_states)
        logits = self.model.compute_logits(hidden_states)
        slot_mapping_out = get_context().slot_mapping_out
        for row, seq in enumerate(seqs):
            current_output_slot = None
            if slot_mapping_out is not None:
                current_output_slot = int(slot_mapping_out[row].item())
            self._mark_state_slot_materialized(seq, current_output_slot)
        reset_context()
        return logits

    @torch.inference_mode()
    def run_logits(self, seqs: list[Sequence], is_prefill: bool):
        prepare_s = 0.0
        forward_s = 0.0
        prefill_exec_batches = 0
        prefill_logical_tokens = 0
        prefill_flat_padded_tokens = 0
        prefill_bucketed_padded_tokens = 0
        if self.config.rwkv_state_cache_enable:
            if is_prefill:
                (
                    logits,
                    prepare_s,
                    forward_s,
                    prefill_logical_tokens,
                    prefill_flat_padded_tokens,
                    prefill_bucketed_padded_tokens,
                    prefill_exec_batches,
                ) = self._compute_prefill_logits_with_state_cache(seqs)
            else:
                started_at = time.perf_counter()
                logits = self._compute_decode_logits_with_state_cache(seqs)
                forward_s = time.perf_counter() - started_at
            if self._run_profile is not None:
                self._run_profile_prepare_s = prepare_s
                self._run_profile_forward_s = forward_s
                self._run_profile_prefill_exec_batches = prefill_exec_batches
                self._run_profile_prefill_logical_tokens = prefill_logical_tokens
                self._run_profile_prefill_flat_padded_tokens = prefill_flat_padded_tokens
                self._run_profile_prefill_bucketed_padded_tokens = prefill_bucketed_padded_tokens
            return logits
        if not is_prefill and len(seqs) == 1:
            seq = seqs[0]
            prepare_started_at = time.perf_counter()
            input_ids, positions = self.prepare_decode([seq])
            prepare_s = time.perf_counter() - prepare_started_at
            forward_started_at = time.perf_counter()
            logits = self.model.forward_one_logits(input_ids, positions)
            forward_s = time.perf_counter() - forward_started_at
            reset_context()
            if self._run_profile is not None:
                self._run_profile_prepare_s = prepare_s
                self._run_profile_forward_s = forward_s
                self._run_profile_prefill_exec_batches = 0
                self._run_profile_prefill_logical_tokens = 0
                self._run_profile_prefill_flat_padded_tokens = 0
                self._run_profile_prefill_bucketed_padded_tokens = 0
            return logits
        if is_prefill:
            (
                logits,
                prepare_s,
                forward_s,
                prefill_logical_tokens,
                prefill_flat_padded_tokens,
                prefill_bucketed_padded_tokens,
                prefill_exec_batches,
            ) = self._compute_prefill_logits_bucketed(seqs)
        else:
            prepare_started_at = time.perf_counter()
            input_ids, positions = self.prepare_decode(seqs)
            prepare_s = time.perf_counter() - prepare_started_at
            try:
                forward_started_at = time.perf_counter()
                logits = self.run_model(input_ids, positions, is_prefill)
                forward_s = time.perf_counter() - forward_started_at
            finally:
                reset_context()
        if self._run_profile is not None:
            self._run_profile_prepare_s = prepare_s
            self._run_profile_forward_s = forward_s
            self._run_profile_prefill_exec_batches = prefill_exec_batches
            self._run_profile_prefill_logical_tokens = prefill_logical_tokens
            self._run_profile_prefill_flat_padded_tokens = prefill_flat_padded_tokens
            self._run_profile_prefill_bucketed_padded_tokens = prefill_bucketed_padded_tokens
        return logits

    def prepare_postprocess(self, seqs: list[Sequence], token_ids: list[int | None] | None) -> None:
        if self.rank != 0:
            return
        for seq in seqs:
            if seq.num_cached_tokens < seq.num_prompt_tokens:
                seq.num_cached_tokens = min(
                    seq.num_prompt_tokens,
                    seq.num_cached_tokens + self._prefill_step_tokens(seq),
                )
        if not self.config.rwkv_state_cache_enable:
            return
        for seq in seqs:
            if seq.pending_hidden_finalize:
                self._publish_hidden_finalize_sequence(seq)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int | None]:
        step_started_at = time.perf_counter()
        if not is_prefill and self._is_hidden_finalize_batch(seqs):
            _ = self.run_logits(seqs, False)
            if self.rank == 0:
                token_ids = [None] * len(seqs)
            else:
                token_ids = None
            post_started_at = time.perf_counter()
            self.prepare_postprocess(seqs, token_ids)
            post_s = time.perf_counter() - post_started_at
            if self._run_profile is not None:
                self._run_profile.record_step(
                    kind="decode",
                    seq_count=len(seqs),
                    total_s=time.perf_counter() - step_started_at,
                    prepare_s=self._run_profile_prepare_s,
                    forward_s=self._run_profile_forward_s,
                    sample_s=0.0,
                    post_s=post_s,
                )
            return token_ids
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
            post_started_at = time.perf_counter()
            self.prepare_postprocess(seqs, token_ids)
            post_s = time.perf_counter() - post_started_at
            if self._run_profile is not None:
                self._run_profile.record_step(
                    kind="decode",
                    seq_count=len(seqs),
                    total_s=time.perf_counter() - step_started_at,
                    prepare_s=0.0,
                    forward_s=0.0,
                    sample_s=0.0,
                    post_s=post_s,
                )
            return token_ids
        logits = self.run_logits(seqs, is_prefill)
        sample_started_at = time.perf_counter()
        if self.rank == 0:
            if is_prefill:
                sample_indices = [
                    index
                    for index, seq in enumerate(seqs)
                    if seq.num_cached_tokens + self._prefill_step_tokens(seq) >= seq.num_prompt_tokens
                ]
                if not sample_indices:
                    token_ids = [None] * len(seqs)
                elif len(sample_indices) == len(seqs):
                    token_ids = self.sampler(logits, seqs).tolist()
                else:
                    sample_logits = logits.index_select(
                        0,
                        torch.tensor(sample_indices, dtype=torch.int64, device=logits.device),
                    )
                    sampled = self.sampler(sample_logits, [seqs[index] for index in sample_indices]).tolist()
                    token_ids = [None] * len(seqs)
                    for index, token_id in zip(sample_indices, sampled, strict=True):
                        token_ids[index] = token_id
            else:
                sample_indices = [index for index, seq in enumerate(seqs) if not seq.pending_hidden_finalize]
                if not sample_indices:
                    token_ids = [None] * len(seqs)
                elif len(sample_indices) == len(seqs):
                    token_ids = self.sampler(logits, seqs).tolist()
                else:
                    sample_logits = logits.index_select(
                        0,
                        torch.tensor(sample_indices, dtype=torch.int64, device=logits.device),
                    )
                    sampled = self.sampler(sample_logits, [seqs[index] for index in sample_indices]).tolist()
                    token_ids = [None] * len(seqs)
                    for index, token_id in zip(sample_indices, sampled, strict=True):
                        token_ids[index] = token_id
        else:
            token_ids = None
        sample_s = time.perf_counter() - sample_started_at
        post_started_at = time.perf_counter()
        self.prepare_postprocess(seqs, token_ids)
        post_s = time.perf_counter() - post_started_at
        if self._run_profile is not None:
            self._run_profile.record_step(
                kind="prefill" if is_prefill else "decode",
                seq_count=len(seqs),
                total_s=time.perf_counter() - step_started_at,
                prepare_s=self._run_profile_prepare_s,
                forward_s=self._run_profile_forward_s,
                sample_s=sample_s,
                post_s=post_s,
                prefill_exec_batches=self._run_profile_prefill_exec_batches if is_prefill else 0,
                prefill_logical_tokens=self._run_profile_prefill_logical_tokens if is_prefill else 0,
                prefill_flat_padded_tokens=self._run_profile_prefill_flat_padded_tokens if is_prefill else 0,
                prefill_bucketed_padded_tokens=self._run_profile_prefill_bucketed_padded_tokens if is_prefill else 0,
            )
        return token_ids
