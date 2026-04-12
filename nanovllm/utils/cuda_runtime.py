from __future__ import annotations

import ctypes
import sysconfig
from functools import lru_cache
from pathlib import Path


def _iter_nvidia_lib_roots():
    seen: set[Path] = set()
    for key in ("purelib", "platlib"):
        root = sysconfig.get_paths().get(key)
        if not root:
            continue
        nvidia_root = (Path(root) / "nvidia").resolve()
        if nvidia_root in seen or not nvidia_root.exists():
            continue
        seen.add(nvidia_root)
        yield nvidia_root


@lru_cache(maxsize=1)
def preload_cuda_nvrtc_libs() -> tuple[str, ...]:
    loaded: list[str] = []
    seen: set[Path] = set()
    candidates = (
        ("cuda_runtime/lib", "libcudart.so*"),
        ("cuda_nvrtc/lib", "libnvrtc.so*"),
        ("cuda_nvrtc/lib", "libnvrtc-builtins.so*"),
        ("cu13/lib", "libcudart.so*"),
        ("cu13/lib", "libnvrtc.so*"),
        ("cu13/lib", "libnvrtc-builtins.so*"),
    )
    for nvidia_root in _iter_nvidia_lib_roots():
        for rel_dir, pattern in candidates:
            lib_dir = nvidia_root / rel_dir
            if not lib_dir.exists():
                continue
            for lib in sorted(lib_dir.glob(pattern)):
                lib = lib.resolve()
                if lib in seen:
                    continue
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                seen.add(lib)
                loaded.append(str(lib))
    return tuple(loaded)
