import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    rwkv_prefill_token_budget: int = 2048
    rwkv_prefill_max_batch_size: int = 128
    rwkv_quant_int8: bool = False
    rwkv_quant_int8_lm_head: bool = False
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    num_state_blocks: int = -1
    use_state_cache: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model) or (os.path.isfile(self.model) and self.model.endswith(".pth"))
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.rwkv_prefill_token_budget > 0
        default_gpu_memory_utilization = type(self).gpu_memory_utilization

        # Check for RWKV pth file
        import glob
        if os.path.isfile(self.model) and self.model.endswith(".pth"):
            pth_files = [self.model]
        else:
            pth_files = glob.glob(os.path.join(self.model, "*.pth"))
        if pth_files:
            # RWKV model - create config from pth filename
            self.use_state_cache = True
            from nanovllm.models.configuration_rwkv7 import RWKV7Config
            self.hf_config = RWKV7Config.from_pth(pth_files[0])
        else:
            self.hf_config = AutoConfig.from_pretrained(self.model)
            self.use_state_cache = getattr(self.hf_config, "model_type", "") == "rwkv7"

        if self.use_state_cache and self.gpu_memory_utilization == default_gpu_memory_utilization:
            self.gpu_memory_utilization = 0.97

        if not self.use_state_cache:
            assert self.kvcache_block_size % 256 == 0
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
