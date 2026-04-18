import os
from dataclasses import dataclass

from nanovllm.models.configuration_rwkv7 import RWKV7Config
from nanovllm.utils.loader import resolve_model_pth
from nanovllm.utils.rwkv_int8 import normalize_rwkv_int8_lm_head_flags


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    rwkv_prefill_token_budget: int = 2048
    rwkv_prefill_max_batch_size: int = 128
    rwkv_prefill_chunk_size: int = -1
    rwkv_state_cache_enable: bool = False
    max_state_slots: int = -1
    rwkv_state_cache_safety_reserve_slots: int = 0
    sampling_bucket_temperature_resolution: float = 0.0
    sampling_bucket_top_p_resolution: float = 0.0
    rwkv_quant_int8: bool = False
    rwkv_int8_fp16_lm_head: bool = False
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    model_config: RWKV7Config | None = None
    eos: int = -1
    num_state_blocks: int = -1
    num_state_slots_total: int = -1
    bs1_graph_slot: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model) or os.path.isfile(self.model)
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.max_num_seqs == -1 or self.max_num_seqs > 0, "max_num_seqs must be -1 or a positive integer."
        assert self.rwkv_prefill_token_budget > 0
        assert self.rwkv_prefill_chunk_size == -1 or self.rwkv_prefill_chunk_size > 0, (
            "rwkv_prefill_chunk_size must be -1 or a positive integer."
        )
        assert self.max_state_slots == -1 or self.max_state_slots > 0, "max_state_slots must be -1 or a positive integer."
        assert self.rwkv_state_cache_safety_reserve_slots >= 0, (
            "rwkv_state_cache_safety_reserve_slots must be non-negative."
        )
        assert self.sampling_bucket_temperature_resolution >= 0.0, "sampling_bucket_temperature_resolution must be non-negative."
        assert self.sampling_bucket_top_p_resolution >= 0.0, "sampling_bucket_top_p_resolution must be non-negative."
        if self.rwkv_state_cache_enable:
            assert self.tensor_parallel_size == 1, "RWKV state cache is currently only wired for tensor_parallel_size=1."
        (
            self.rwkv_quant_int8_lm_head,
            self.rwkv_quant_int8_lm_head_marlin,
        ) = normalize_rwkv_int8_lm_head_flags(
            rwkv_quant_int8=self.rwkv_quant_int8,
            rwkv_int8_fp16_lm_head=self.rwkv_int8_fp16_lm_head,
        )
        model_pth = resolve_model_pth(self.model)
        self.model_config = RWKV7Config.from_pth(model_pth)

        self.max_model_len = min(self.max_model_len, self.model_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
