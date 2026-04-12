import torch
import torch.nn as nn

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import LayerNorm
from nanovllm.models.rwkv7_decode import rwkv7_forward_decode, rwkv7_forward_one
from nanovllm.models.rwkv7_layers import RWKV7Block
from nanovllm.models.rwkv7_weights import (
    apply_rwkv7_post_load_quantization,
    load_rwkv7_pth_into_model,
)
from nanovllm.utils.context import get_context


class RWKV7Model(nn.Module):
    """RWKV-7 model."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        intermediate_size: int,
        vocab_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers

        self.emb = VocabParallelEmbedding(vocab_size, hidden_size)
        self.decode_tokenshift_scratch = None
        self.blocks = nn.ModuleList([
            RWKV7Block(idx, hidden_size, num_heads, head_dim, intermediate_size)
            for idx in range(num_layers)
        ])
        self.ln_out = LayerNorm(hidden_size)

        self.z = {}
        self._decode_elapsed_cache: dict[int, torch.Tensor] = {}

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        context = get_context()
        x = self.emb(input_ids)

        if not context.is_prefill:
            return rwkv7_forward_decode(self, x, positions)

        att_mask = None
        if x.dim() == 3:
            _, seq_len, _ = x.shape
            att_mask = (
                torch.arange(seq_len, device=x.device, dtype=torch.int32).unsqueeze(0) <
                (seq_len - context.context_lens).unsqueeze(1)
            ).unsqueeze(2)

        v_first = None
        for block in self.blocks:
            x, v_first = block(x, positions, True, v_first, att_mask)

        return self.ln_out(x)

    def forward_one(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden = rwkv7_forward_one(self, input_ids, positions)
        if hidden is None:
            return self.forward(input_ids, positions)
        return hidden

    def load_pth(self, pth_path: str):
        load_rwkv7_pth_into_model(self, pth_path)


class RWKV7ForCausalLM(nn.Module):
    """RWKV-7 for causal language modeling."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        hidden_size = config.hidden_size
        head_dim = getattr(config, "head_dim", 64)
        num_heads = getattr(config, "num_heads", hidden_size // head_dim)
        num_layers = config.num_hidden_layers
        intermediate_size = config.intermediate_size
        vocab_size = config.vocab_size

        self.model = RWKV7Model(hidden_size, num_heads, head_dim, num_layers, intermediate_size, vocab_size)
        self.lm_head = ParallelLMHead(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def forward_logits(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        if self.lm_head.tp_size == 1 and hidden_states.dim() == 2:
            if self.lm_head.use_int8:
                return self.lm_head(hidden_states)
            return torch.matmul(hidden_states, self.lm_head.weight)
        return self.lm_head(hidden_states)

    def forward_one_logits(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.forward_one(input_ids, positions)
        if self.lm_head.tp_size == 1 and hidden_states.dim() == 2:
            if self.lm_head.use_int8:
                return self.lm_head(hidden_states)
            return torch.matmul(hidden_states, self.lm_head.weight)
        return self.lm_head(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def apply_post_load_quantization(self):
        apply_rwkv7_post_load_quantization(self)

    def load_pth(self, pth_path: str):
        self.model.load_pth(pth_path)
        self.apply_post_load_quantization()
