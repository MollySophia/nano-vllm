import os
from pathlib import Path
from threading import Lock

import torch
from torch.utils.cpp_extension import load

_EXT_NAME = "nanovllm_rapid_sampling"
_MODULE = None
_LOCK = Lock()


def ensure_loaded():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    with _LOCK:
        if _MODULE is not None:
            return _MODULE
        cur = Path(__file__).resolve().parent
        src_cpp = str(cur / "cuda" / "rapid_sampling.cpp")
        src_cu = str(cur / "cuda" / "rapid_sampling.cu")
        _MODULE = load(
            name=_EXT_NAME,
            sources=[src_cpp, src_cu],
            verbose=False,
            extra_cuda_cflags=[
                "-O3",
                "-res-usage",
                "--extra-device-vectorization",
            ] + (["-Xptxas", "-O3"] if os.name != "nt" else []),
        )
    return _MODULE


def setup_rand(seed: int, batch_size: int) -> torch.Tensor:
    return ensure_loaded().setup_rand(int(seed), int(batch_size))


def batch_sampling_temperature_topk_topp(
    logits: torch.Tensor,
    states: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    return ensure_loaded().batch_sampling_temperature_topk_topp(
        logits,
        states,
        float(temperature),
        int(top_k),
        float(top_p),
    )


def batch_sampling_repetition_temperature_topk_topp(
    logits: torch.Tensor,
    penalties: torch.Tensor,
    states: torch.Tensor,
    presence_penalty: float,
    repetition_penalty: float,
    penalty_decay: float,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    return ensure_loaded().batch_sampling_repetition_temperature_topk_topp(
        logits,
        penalties,
        states,
        float(presence_penalty),
        float(repetition_penalty),
        float(penalty_decay),
        float(temperature),
        int(top_k),
        float(top_p),
    )
