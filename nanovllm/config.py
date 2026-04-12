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
    rwkv_quant_int8: bool = False
    rwkv_int8_fp16_lm_head: bool = False
    rwkv_quant_int8_lm_head: bool = False
    rwkv_quant_int8_lm_head_marlin: bool = False
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    model_config: RWKV7Config | None = None
    eos: int = -1
    num_state_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model) or os.path.isfile(self.model)
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.rwkv_prefill_token_budget > 0
        (
            self.rwkv_quant_int8_lm_head,
            self.rwkv_quant_int8_lm_head_marlin,
        ) = normalize_rwkv_int8_lm_head_flags(
            rwkv_quant_int8=self.rwkv_quant_int8,
            rwkv_int8_fp16_lm_head=self.rwkv_int8_fp16_lm_head,
            rwkv_int8_lm_head=self.rwkv_quant_int8_lm_head,
            rwkv_int8_lm_head_marlin=self.rwkv_quant_int8_lm_head_marlin,
        )
        model_pth = resolve_model_pth(self.model)
        self.model_config = RWKV7Config.from_pth(model_pth)

        default_gpu_memory_utilization = type(self).gpu_memory_utilization
        if self.gpu_memory_utilization == default_gpu_memory_utilization:
            self.gpu_memory_utilization = 0.97

        self.max_model_len = min(self.max_model_len, self.model_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
