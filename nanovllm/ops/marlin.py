from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from threading import Lock

import torch
from torch.utils.cpp_extension import load

from nanovllm.ops.marlin_scalar_type import ScalarType

_EXT_NAME = "nanovllm_marlin"
_LOCK = Lock()
_LOADED = False
_SCHEMA_LIB = None
_FAKES_REGISTERED = False


def _cuda_version_tuple() -> tuple[int, int]:
    version = torch.version.cuda or "0.0"
    major, minor, *_ = version.split(".")
    return int(major), int(minor)


def _arch_list_for_generation() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Marlin.")
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def _define_schemas():
    global _SCHEMA_LIB
    if _SCHEMA_LIB is not None:
        return
    lib = torch.library.Library(_EXT_NAME, "DEF")
    lib.define(
        "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
        "Tensor? b_bias_or_none, Tensor b_scales, Tensor? a_scales, "
        "Tensor? global_scale, Tensor? b_zeros_or_none, Tensor? g_idx_or_none, "
        "Tensor? perm_or_none, Tensor workspace, int b_type_id, "
        "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full=True, "
        "bool use_atomic_add=False, bool use_fp32_reduce=False, "
        "bool is_zp_float=False) -> Tensor"
    )
    _SCHEMA_LIB = lib


def _register_fake_impls():
    global _FAKES_REGISTERED
    if _FAKES_REGISTERED:
        return
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake

    @register_fake(f"{_EXT_NAME}::marlin_gemm")
    def _marlin_gemm_fake(
        a: torch.Tensor,
        c_or_none: torch.Tensor | None,
        b_q_weight: torch.Tensor,
        b_bias_or_none: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_zeros_or_none: torch.Tensor | None,
        g_idx_or_none: torch.Tensor | None,
        perm_or_none: torch.Tensor | None,
        workspace: torch.Tensor,
        b_type_id: int,
        size_m: torch.SymInt,
        size_n: torch.SymInt,
        size_k: torch.SymInt,
        is_k_full: bool = True,
        use_atomic_add: bool = False,
        use_fp32_reduce: bool = False,
        is_zp_float: bool = False,
    ) -> torch.Tensor:
        dtype = a.dtype if a.dtype in (torch.half, torch.bfloat16) else b_scales.dtype
        return torch.empty((size_m, size_n), device=a.device, dtype=dtype)

    _FAKES_REGISTERED = True


def _generate_kernels(marlin_dir: Path):
    script = marlin_dir / "generate_minimal_kernels.py"
    subprocess.check_call(
        [sys.executable, str(script), _arch_list_for_generation()],
        cwd=marlin_dir,
    )


def ensure_loaded():
    global _LOADED
    if _LOADED:
        return
    with _LOCK:
        if _LOADED:
            return

        _define_schemas()
        _register_fake_impls()

        cur = Path(__file__).resolve().parent
        cuda_root = cur / "cuda"
        marlin_dir = cuda_root / "marlin"
        _generate_kernels(marlin_dir)

        sources = [str(marlin_dir / "marlin.cu")]
        sources.extend(str(path) for path in sorted(marlin_dir.glob("sm*_kernel_*_u8b128_*.cu")))

        extra_cuda_cflags = [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-lineinfo",
        ]
        if _cuda_version_tuple() >= (12, 8):
            extra_cuda_cflags.append("-static-global-template-stub=false")

        load(
            name=_EXT_NAME,
            sources=sources,
            is_python_module=False,
            verbose=False,
            extra_include_paths=[str(cuda_root)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=extra_cuda_cflags,
        )
        _LOADED = True


def marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor | None,
    b_q_weight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    ensure_loaded()
    return torch.ops.nanovllm_marlin.marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias,
        b_scales,
        a_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )
