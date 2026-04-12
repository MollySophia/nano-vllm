import os
from glob import glob

import torch


def resolve_model_pth(path: str) -> str:
    if os.path.isfile(path):
        if not path.endswith(".pth"):
            raise ValueError(f"Only RWKV .pth checkpoints are supported: {path}")
        return path

    pth_files = sorted(glob(os.path.join(path, "*.pth")))
    if not pth_files:
        raise ValueError(f"No RWKV .pth checkpoint found under: {path}")
    return pth_files[0]


def load_model(model, path: str):
    if not hasattr(model, "load_pth"):
        raise ValueError("Only RWKV models with load_pth() are supported.")
    model.load_pth(resolve_model_pth(path))
    # `weight_loader` is only needed during checkpoint load. Leaving the bound
    # method attached on Parameters keeps module objects alive via a Python
    # reference cycle (`Parameter -> bound method -> module`), which in turn
    # pins GPU weights across sequential benchmark runs.
    for param in model.parameters():
        if hasattr(param, "weight_loader"):
            delattr(param, "weight_loader")
