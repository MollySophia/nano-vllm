# modified from flash-linear-attention
import re
from pathlib import Path

import torch


class RWKV7Config:
    model_type = "rwkv7"

    def __init__(
        self,
        hidden_size: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 24,
        head_dim: int | None = 64,
        num_heads: int | None = None,
        decay_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        a_low_rank_dim: int = 64,
        v_low_rank_dim: int = 16,
        hidden_act: str = "sqrelu",
        max_position_embeddings: int = 2048,
        norm_first: bool = True,
        norm_bias: bool = True,
        norm_eps: float = 1e-5,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        vocab_size: int = 65536,
        value_dim: int | list[int] | None = None,
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.hidden_ratio = hidden_ratio
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio)
        self.intermediate_size = intermediate_size
        self.norm_first = norm_first
        self.num_hidden_layers = num_hidden_layers

        if head_dim is None and num_heads is not None:
            head_dim = int(hidden_size // num_heads)
        elif head_dim is not None and num_heads is None:
            num_heads = int(hidden_size // head_dim)

        if value_dim is None:
            value_dim = [hidden_size] * num_hidden_layers
        elif isinstance(value_dim, int):
            assert value_dim >= hidden_size, "value_dim must be greater than hidden_size"
            assert value_dim % hidden_size == 0, "value_dim must be divisible by hidden_size"
            value_dim = [value_dim] * num_hidden_layers
        else:
            assert len(value_dim) == num_hidden_layers, "value_dim must have the same length as num_hidden_layers"
            for v in value_dim:
                assert v >= hidden_size, "value_dim must be greater than hidden_size"
                assert v % hidden_size == 0, "value_dim must be divisible by hidden_size"

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.value_dim = value_dim

        self.decay_low_rank_dim = decay_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.hidden_act = hidden_act
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.torch_dtype = torch_dtype
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_pth(cls, pth_path: str, ctx_len: int | None = None):
        """Create config from RWKV pth file."""
        z = torch.load(pth_path, map_location="cpu")
        if ctx_len is None:
            # Prefer context length encoded in filename like "...-ctx8192.pth".
            m = re.search(r"ctx(\d+)", Path(pth_path).name.lower())
            ctx_len = int(m.group(1)) if m else 4096

        # Infer dimensions from weight keys
        # blocks.0.att.r_k shape: [num_heads, head_dim]
        n_head, head_size = z["blocks.0.att.r_k"].shape
        n_embd = n_head * head_size

        # Count layers
        max_layer = -1
        for k in z.keys():
            kk = k.split(".")
            if kk[0] == "blocks":
                max_layer = max(max_layer, int(kk[1]))
        n_layer = max_layer + 1

        # Get vocab size from embedding
        vocab_size = z["emb.weight"].shape[0]

        # Infer intermediate size from ffn key weight
        intermediate_size = z["blocks.0.ffn.key.weight"].shape[0]

        return cls(
            hidden_size=n_embd,
            intermediate_size=intermediate_size,
            num_hidden_layers=n_layer,
            head_dim=head_size,
            num_heads=n_head,
            vocab_size=vocab_size,
            max_position_embeddings=ctx_len,
            torch_dtype=torch.float16,
        )
