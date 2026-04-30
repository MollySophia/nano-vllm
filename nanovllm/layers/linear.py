import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import lru_cache

from nanovllm.ops.marlin import ensure_loaded as ensure_marlin_loaded
from nanovllm.ops.marlin_utils import (
    apply_rtn_marlin_linear,
    marlin_make_workspace_new,
    repack_weights,
    rtn_quantize,
    scalar_types,
    verify_marlin_supports_shape,
)

def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


_MARLIN_IMPL_ERROR: Exception | None = None


@lru_cache(maxsize=1)
def _get_marlin_impl():
    global _MARLIN_IMPL_ERROR
    try:
        ensure_marlin_loaded()
        _MARLIN_IMPL_ERROR = None
        return {
            "rtn_quantize": rtn_quantize,
            "repack_weights": repack_weights,
            "apply_rtn_marlin_linear": apply_rtn_marlin_linear,
            "marlin_make_workspace_new": marlin_make_workspace_new,
            "scalar_types": scalar_types,
            "verify_marlin_supports_shape": verify_marlin_supports_shape,
        }
    except Exception as exc:
        _MARLIN_IMPL_ERROR = exc
        return None


def get_marlin_impl_or_raise():
    marlin = _get_marlin_impl()
    if marlin is not None:
        return marlin
    raise RuntimeError(
        "Local Marlin runtime is unavailable. "
        "rwkv_quant_int8 now requires the vendored Marlin sources to JIT build successfully. "
        f"Original error: {_MARLIN_IMPL_ERROR!r}"
    )


def _tp_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _tp_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = _tp_rank()
        self.tp_size = _tp_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MatmulLinear(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        weight_layout: str = "in_out",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        assert weight_layout in ("in_out", "out_in")
        self.weight_layout = weight_layout
        if weight_layout == "in_out":
            self.weight = nn.Parameter(torch.empty(input_size, output_size))
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_layout == "in_out":
            return F.linear(x, self.weight.t(), self.bias)
        return F.linear(x, self.weight, self.bias)


class MarlinInt8Linear(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        group_size: int = 128,
        scale_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.group_size = group_size
        self.scale_dtype = scale_dtype
        self.register_buffer("qweight", None)
        self.register_buffer("scales", None)
        self.register_buffer("workspace", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def quantize_from_weight(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        weight_layout: str = "auto",
    ):
        marlin = get_marlin_impl_or_raise()
        assert self.group_size == 128, "Minimal Marlin experiment only supports group_size=128."
        assert weight_layout in ("auto", "in_out", "out_in")
        assert weight.shape in (
            (self.input_size, self.output_size),
            (self.output_size, self.input_size),
        )
        # vLLM RTN/Marlin expects row-major [out, in]. Square projections make
        # shape-based inference ambiguous, so prefer the explicit module layout
        # when available.
        if weight_layout == "in_out":
            weight_oi = weight.t().contiguous()
        elif weight_layout == "out_in":
            weight_oi = weight.contiguous()
        else:
            weight_oi = weight if weight.shape == (self.output_size, self.input_size) else weight.t().contiguous()
        marlin["verify_marlin_supports_shape"](
            self.output_size,
            self.input_size,
            self.input_size,
            self.group_size,
        )
        q_u8, scales = marlin["rtn_quantize"](weight_oi, 8, self.group_size)
        q_packed, s_packed = marlin["repack_weights"](q_u8, scales, 8)
        self.qweight = q_packed.contiguous()
        self.scales = s_packed.to(self.scale_dtype).contiguous()
        self.workspace = marlin["marlin_make_workspace_new"](weight.device, 4)
        if self.bias is not None:
            target_dtype = (
                bias.dtype
                if bias is not None
                else (weight.dtype if weight.dtype in (torch.float16, torch.bfloat16) else torch.float16)
            )
            if self.bias.device != weight.device or self.bias.dtype != target_dtype:
                self.bias = nn.Parameter(
                    torch.empty(self.output_size, device=weight.device, dtype=target_dtype),
                    requires_grad=False,
                )
            if bias is not None:
                self.bias.data.copy_(bias.to(device=weight.device, dtype=target_dtype))
            else:
                self.bias.data.zero_()
        return self

    @classmethod
    @torch.no_grad()
    def from_float(cls, module: MatmulLinear):
        qmod = cls(
            module.input_size,
            module.output_size,
            bias=module.bias is not None,
            group_size=128,
            scale_dtype=module.weight.dtype if module.weight.dtype in (torch.float16, torch.bfloat16) else torch.float16,
        )
        bias = None if module.bias is None else module.bias.detach()
        qmod.quantize_from_weight(
            module.weight.detach(),
            bias,
            weight_layout=getattr(module, "weight_layout", "auto"),
        )
        return qmod

    @property
    def weight(self):
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        marlin = get_marlin_impl_or_raise()
        orig_shape = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        y = marlin["apply_rtn_marlin_linear"](
            input=x2,
            weight=self.qweight,
            weight_scale=self.scales,
            workspace=self.workspace,
            quant_type=marlin["scalar_types"].uint8b128,
            output_size_per_partition=self.output_size,
            input_size_per_partition=self.input_size,
            bias=self.bias,
        )
        return y.reshape(*orig_shape, self.output_size)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = _tp_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = _tp_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = _tp_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
