import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import sys
import os
import ctypes
import sysconfig
from pathlib import Path
from functools import lru_cache

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


@lru_cache(maxsize=1)
def _preload_env_cuda_libs() -> list[str]:
    purelib = Path(sysconfig.get_paths().get("purelib", ""))
    nvidia_root = purelib / "nvidia"
    if not nvidia_root.exists():
        return []

    loaded: list[str] = []
    candidates = [
        nvidia_root / "cuda_runtime" / "lib" / "libcudart.so.12",
        nvidia_root / "cuda_nvrtc" / "lib" / "libnvrtc.so.12",
        nvidia_root / "cublas" / "lib" / "libcublasLt.so.12",
        nvidia_root / "cublas" / "lib" / "libcublas.so.12",
        nvidia_root / "cusparselt" / "lib" / "libcusparseLt.so.0",
        nvidia_root / "cusparse" / "lib" / "libcusparse.so.12",
        nvidia_root / "cusolver" / "lib" / "libcusolver.so.11",
        nvidia_root / "nccl" / "lib" / "libnccl.so.2",
    ]
    for lib in candidates:
        if not lib.exists():
            continue
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        loaded.append(str(lib))
    return loaded


@lru_cache(maxsize=1)
def _get_marlin_impl():
    try:
        _preload_env_cuda_libs()
        from vllm.model_executor.layers.quantization.rtn import rtn_quantize, repack_weights
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_rtn_marlin_linear,
            marlin_make_workspace_new,
        )
        from vllm.scalar_type import scalar_types
        return {
            "rtn_quantize": rtn_quantize,
            "repack_weights": repack_weights,
            "apply_rtn_marlin_linear": apply_rtn_marlin_linear,
            "marlin_make_workspace_new": marlin_make_workspace_new,
            "scalar_types": scalar_types,
        }
    except Exception:
        return None


def get_marlin_impl_or_raise():
    marlin = _get_marlin_impl()
    if marlin is not None:
        return marlin
    current_python = Path(sys.executable).resolve()
    current_prefix = current_python.parent.parent
    site_packages = next((p for p in sys.path if "site-packages" in p), "")
    cuda_lib_root = Path(site_packages) / "nvidia" if site_packages else None
    cuda_libs = []
    if cuda_lib_root is not None and cuda_lib_root.exists():
        for rel in (
            "cuda_runtime/lib",
            "cu13/lib",
            "cuda_nvrtc/lib",
            "cublas/lib",
            "cusparse/lib",
            "cusparselt/lib",
            "cusolver/lib",
            "nccl/lib",
        ):
            lib_path = cuda_lib_root / rel
            if lib_path.exists():
                cuda_libs.append(str(lib_path))
    ld_hint = ":".join(cuda_libs) if cuda_libs else "<python-env>/lib/pythonX.Y/site-packages/nvidia/.../lib"
    raise RuntimeError(
        "Marlin runtime is unavailable. rwkv_quant_int8 now requires Marlin and no longer falls back. "
        f"Current python: {current_python}. Current prefix: {current_prefix}. "
        "Use the Python interpreter from the environment where vLLM/Marlin is installed. "
        "The loader already tries to preload CUDA libs from the active environment; if that still fails, set "
        f"LD_LIBRARY_PATH to include that environment's NVIDIA CUDA runtime libraries, e.g. {ld_hint}. "
        f"Current LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')!r}"
    )


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
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
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
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(input_size, output_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.matmul(x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y


def _choose_group_size(input_size: int, preferred: int = 128) -> int:
    for gs in (preferred, 64, 32, 16, 8, 4, 2, 1):
        if gs <= input_size and input_size % gs == 0:
            return gs
    return 1


def _maybe_compile_int8_helper(fn):
    return fn


def _int8_cublas_dequant_eager(
    y_int32: torch.Tensor,
    x_scale: torch.Tensor,
    scales_fp16: torch.Tensor,
) -> torch.Tensor:
    return y_int32.to(torch.float16) * x_scale.to(torch.float16) * scales_fp16


def _int8_cublas_quant_eager(
    x: torch.Tensor,
    act_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    x_scale = x_absmax / act_scale
    x_int8 = torch.clamp(torch.round(x / x_scale), -128, 127).to(torch.int8)
    return x_int8, x_scale


try:
    _int8_cublas_dequant = torch.compile(_int8_cublas_dequant_eager, dynamic=True)
except Exception:
    _int8_cublas_dequant = _int8_cublas_dequant_eager

try:
    _int8_cublas_quant = torch.compile(_int8_cublas_quant_eager, dynamic=True)
except Exception:
    _int8_cublas_quant = _int8_cublas_quant_eager


if _HAS_TRITON:
    @triton.jit
    def _int8_group128_matmul_kernel(
        x_ptr,
        q_ptr,
        s_ptr,
        b_ptr,
        y_ptr,
        M,
        N,
        NUM_GROUPS,
        stride_xm,
        stride_xk,
        stride_qg,
        stride_qk,
        stride_qn,
        stride_sg,
        stride_sn,
        stride_bn,
        stride_ym,
        stride_yn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, 128)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for g in range(NUM_GROUPS):
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (g * 128 + offs_k)[None, :] * stride_xk
            q_ptrs = q_ptr + g * stride_qg + offs_k[:, None] * stride_qk + offs_n[None, :] * stride_qn
            x_mask = offs_m[:, None] < M
            q_mask = offs_n[None, :] < N
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            q = tl.load(q_ptrs, mask=q_mask, other=0)
            q = tl.cast(q, x.dtype)
            group_acc = tl.dot(x, q, out_dtype=tl.float32)
            scales = tl.load(s_ptr + g * stride_sg + offs_n * stride_sn, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += group_acc * scales[None, :]

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n * stride_bn, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += bias[None, :]

        y = acc.to(tl.float16)
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(y_ptrs, y, mask=y_mask)

    @triton.jit
    def _int8_group_matmul_kernel(
        x_ptr,
        q_ptr,
        s_ptr,
        b_ptr,
        y_ptr,
        M,
        N,
        NUM_GROUPS,
        GROUP_SIZE,
        stride_xm,
        stride_xk,
        stride_qg,
        stride_qk,
        stride_qn,
        stride_sg,
        stride_sn,
        stride_bn,
        stride_ym,
        stride_yn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for g in range(NUM_GROUPS):
            group_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, GROUP_SIZE, BLOCK_K):
                x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (g * GROUP_SIZE + k0 + offs_k)[None, :] * stride_xk
                q_ptrs = q_ptr + g * stride_qg + (k0 + offs_k)[:, None] * stride_qk + offs_n[None, :] * stride_qn
                x_mask = (offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < GROUP_SIZE)
                q_mask = ((k0 + offs_k)[:, None] < GROUP_SIZE) & (offs_n[None, :] < N)
                x = tl.load(x_ptrs, mask=x_mask, other=0.0)
                q = tl.load(q_ptrs, mask=q_mask, other=0)
                q = tl.cast(q, x.dtype)
                group_acc += tl.dot(x, q, out_dtype=tl.float32)

            scales = tl.load(s_ptr + g * stride_sg + offs_n * stride_sn, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += group_acc * scales[None, :]

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n * stride_bn, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += bias[None, :]

        y = acc.to(tl.float16)
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(y_ptrs, y, mask=y_mask)


def _int8_group_matmul_triton(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert _HAS_TRITON
    assert x.is_cuda and qweight.is_cuda and scales.is_cuda
    assert x.dim() == 2
    m, k = x.shape
    num_groups, group_size, n = qweight.shape
    assert num_groups * group_size == k
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if group_size == 128:
        block_m = 64
        num_stages = None
        if n >= 16384:
            # FFN up-projection shapes prefer a medium-wide tile with fewer warps.
            block_n = 128
            num_warps = 4
            num_stages = 2
        else:
            # Attention / smaller-N paths prefer different tiles by batch size.
            if n == 4096 and m <= 640:
                block_n = 256
                num_warps = 8
                num_stages = 3
            elif n == 4096 and m <= 1024:
                block_n = 128
                num_warps = 4
                num_stages = 2
            else:
                block_n = 64
                num_warps = 4
        grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
        kwargs = dict(
            HAS_BIAS=bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        if num_stages is not None:
            kwargs["num_stages"] = num_stages
        _int8_group128_matmul_kernel[grid](
            x,
            qweight,
            scales,
            bias if bias is not None else qweight,
            y,
            m,
            n,
            num_groups,
            x.stride(0),
            x.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            qweight.stride(2),
            scales.stride(0),
            scales.stride(1),
            0 if bias is None else bias.stride(0),
            y.stride(0),
            y.stride(1),
            **kwargs,
        )
        return y
    grid = (triton.cdiv(m, 64), triton.cdiv(n, 64))
    _int8_group_matmul_kernel[grid](
        x,
        qweight,
        scales,
        bias if bias is not None else qweight,
        y,
        m,
        n,
        num_groups,
        group_size,
        x.stride(0),
        x.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        qweight.stride(2),
        scales.stride(0),
        scales.stride(1),
        0 if bias is None else bias.stride(0),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    return y


def _int8_matmul_impl(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    orig_shape = x.shape[:-1]
    k = x.shape[-1]
    num_groups = qweight.shape[0]
    assert qweight.shape[1] == group_size
    assert num_groups * group_size == k
    if _HAS_TRITON and x.is_cuda and qweight.is_cuda and scales.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
        x2 = x.reshape(-1, k)
        y2 = _int8_group_matmul_triton(x2, qweight, scales, bias)
        return y2.reshape(*orig_shape, qweight.shape[-1])
    xg = x.reshape(-1, num_groups, group_size).to(torch.float32)
    qg = qweight.to(torch.float32)
    yg = torch.einsum("mgk,gkn->mgn", xg, qg)
    y = (yg * scales.to(torch.float32).unsqueeze(0)).sum(dim=1)
    y = y.to(x.dtype).reshape(*orig_shape, qweight.shape[-1])
    if bias is not None:
        y = y + bias.to(x.dtype)
    return y


_int8_matmul = _maybe_compile_int8_helper(_int8_matmul_impl)


def _int8_per_channel_cublas_impl(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scales_fp16: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    act_scale: float = 127.0,
) -> torch.Tensor:
    orig_shape = x.shape[:-1]
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    assert qweight.dim() == 2
    n, k2 = qweight.shape
    assert k == k2
    if scales_fp16 is None:
        scales_fp16 = scales.to(torch.float16)
    if x2.shape[0] <= 16:
        weight_fp16 = (qweight.to(torch.float16) * scales_fp16[:, None]).t().contiguous()
        y = torch.matmul(x2, weight_fp16)
    else:
        x_int8, x_scale = _int8_cublas_quant(x2, act_scale)
        y_int32 = torch._int_mm(x_int8, qweight.t())
        y = _int8_cublas_dequant(y_int32, x_scale, scales_fp16)
    if bias is not None:
        y = y + bias.to(torch.float16)
    y = y.to(x.dtype)
    return y.reshape(*orig_shape, n)


def _int8_per_channel_cublas(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scales_fp16: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert hasattr(torch, "_int_mm")
    assert x.is_cuda and qweight.is_cuda and scales.is_cuda
    assert x.dtype in (torch.float16, torch.bfloat16)
    return _int8_per_channel_cublas_impl(x, qweight, scales, scales_fp16, bias)


class Int8MatmulLinear(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        group_size: int = 128,
        scale_dtype: torch.dtype = torch.float16,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.group_size = group_size
        self.scale_dtype = scale_dtype
        self.eps = eps
        assert input_size % group_size == 0
        self.num_groups = input_size // group_size
        self.register_buffer("qweight", torch.empty(self.num_groups, group_size, output_size, dtype=torch.int8))
        self.register_buffer("scales", torch.empty(self.num_groups, output_size, dtype=scale_dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def quantize_from_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        assert weight.shape == (self.input_size, self.output_size)
        weight_g = weight.view(self.num_groups, self.group_size, self.output_size)
        max_abs = weight_g.abs().amax(dim=1)
        scales = torch.clamp(max_abs / 127.0, min=self.eps)
        qweight = torch.round(weight_g / scales.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        self.qweight.copy_(qweight)
        self.scales.copy_(scales.to(self.scale_dtype))
        if self.bias is not None and bias is not None:
            self.bias.data.copy_(bias)
        elif self.bias is not None:
            self.bias.data.zero_()
        return self

    @classmethod
    @torch.no_grad()
    def from_float(cls, module: MatmulLinear):
        qmod = cls(
            module.input_size,
            module.output_size,
            bias=module.bias is not None,
            group_size=_choose_group_size(module.input_size),
            scale_dtype=module.weight.dtype if module.weight.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16,
        )
        bias = None if module.bias is None else module.bias.detach()
        qmod.quantize_from_weight(module.weight.detach(), bias)
        return qmod

    @property
    def weight(self):
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _int8_matmul(x, self.qweight, self.scales, self.group_size, self.bias)


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
    def quantize_from_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        marlin = get_marlin_impl_or_raise()
        assert self.group_size == 128, "Minimal Marlin experiment only supports group_size=128."
        assert weight.shape == (self.input_size, self.output_size)
        # vLLM RTN/Marlin expects row-major [out, in].
        weight_oi = weight.t().contiguous()
        q_u8, scales = marlin["rtn_quantize"](weight_oi, 8, self.group_size)
        q_packed, s_packed = marlin["repack_weights"](q_u8, scales, 8)
        self.qweight = q_packed.contiguous()
        self.scales = s_packed.to(self.scale_dtype).contiguous()
        self.workspace = marlin["marlin_make_workspace_new"](weight.device, 4)
        if self.bias is not None and bias is not None:
            self.bias.data.copy_(bias)
        elif self.bias is not None:
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
            scale_dtype=module.weight.dtype if module.weight.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16,
        )
        bias = None if module.bias is None else module.bias.detach()
        qmod.quantize_from_weight(module.weight.detach(), bias)
        return qmod

    @property
    def weight(self):
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        marlin = get_marlin_impl_or_raise()
        return marlin["apply_rtn_marlin_linear"](
            input=x,
            weight=self.qweight,
            weight_scale=self.scales,
            workspace=self.workspace,
            quant_type=marlin["scalar_types"].uint8b128,
            output_size_per_partition=self.output_size,
            input_size_per_partition=self.input_size,
            bias=self.bias,
        )


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
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
        tp_size = dist.get_world_size()
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
        tp_size = dist.get_world_size()
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
