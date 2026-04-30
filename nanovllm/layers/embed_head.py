import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context
from nanovllm.layers.linear import get_marlin_impl_or_raise


@torch.jit.ignore
def _all_reduce_(x: torch.Tensor):
    dist.all_reduce(x)


@torch.jit.ignore
def _gather_logits(logits: torch.Tensor, tp_size: int, tp_rank: int):
    all_logits = [torch.empty_like(logits) for _ in range(tp_size)] if tp_rank == 0 else None
    dist.gather(logits, all_logits, 0)
    return torch.cat(all_logits, -1) if tp_rank == 0 else None


def _tp_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _tp_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = _tp_rank()
        self.tp_size = _tp_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size == 1:
            return F.embedding(x, self.weight)
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        y = mask.unsqueeze(1) * y
        _all_reduce_(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty(embedding_dim, self.num_embeddings_per_partition))
        self.weight.weight_loader = self.weight_loader
        self.register_buffer("qweight", None)
        self.register_buffer("scales", None)
        self.register_buffer("workspace", None)
        self.group_size = 128
        self.use_int8 = False

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = self.num_embeddings_per_partition
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size).t().contiguous()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size == 1 and x.dim() == 2:
            if self.use_int8:
                return self._forward_marlin(x)
            return F.linear(x, self.weight.t())
        context = get_context()
        if context.is_prefill:
            if x.dim() == 3:
                x = x[:, -1, :].contiguous()
            else:
                last_indices = context.cu_seqlens_q[1:] - 1
                x = x[last_indices].contiguous()
        if self.use_int8:
            logits = self._forward_marlin(x)
        else:
            logits = F.linear(x, self.weight.t())
        if self.tp_size > 1:
            logits = _gather_logits(logits, self.tp_size, self.tp_rank)
        return logits

    def _forward_marlin(self, x: torch.Tensor) -> torch.Tensor:
        marlin = get_marlin_impl_or_raise()
        return marlin["apply_rtn_marlin_linear"](
            input=x,
            weight=self.qweight,
            weight_scale=self.scales,
            workspace=self.workspace,
            quant_type=marlin["scalar_types"].uint8b128,
            output_size_per_partition=self.num_embeddings_per_partition,
            input_size_per_partition=self.weight.shape[0] if self.weight is not None else x.shape[-1],
            bias=None,
        )

    @torch.no_grad()
    def quantize_weight_marlin_int8(self):
        marlin = get_marlin_impl_or_raise()
        weight = self.weight.data
        weight_row = weight.t().contiguous()
        marlin["verify_marlin_supports_shape"](
            self.num_embeddings_per_partition,
            weight_row.shape[1],
            weight_row.shape[1],
            self.group_size,
        )
        q_u8, scales = marlin["rtn_quantize"](weight_row, 8, self.group_size)
        q_packed, s_packed = marlin["repack_weights"](q_u8, scales, 8)
        self.qweight = q_packed.contiguous()
        self.scales = s_packed.to(weight.dtype if weight.dtype in (torch.float16, torch.bfloat16) else torch.float16).contiguous()
        self.workspace = marlin["marlin_make_workspace_new"](weight.device, 4)
        self.use_int8 = True
        if "weight" in self._parameters:
            del self._parameters["weight"]
        self.register_parameter("weight", None)
