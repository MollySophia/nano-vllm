import argparse


def add_rwkv_int8_cli_args(
    parser: argparse.ArgumentParser,
    *,
    include_enable_flag: bool = True,
) -> None:
    if include_enable_flag:
        parser.add_argument("--rwkv-quant-int8", action="store_true")
    parser.add_argument(
        "--rwkv-int8-fp16-lm-head",
        action="store_true",
        help="Keep the RWKV lm_head in fp16 even when --rwkv-quant-int8 is enabled.",
    )
    parser.add_argument(
        "--rwkv-int8-lm-head",
        action="store_true",
        help="Use the legacy cublas int8 lm_head path instead of the default Marlin lm_head.",
    )
    parser.add_argument(
        "--rwkv-int8-lm-head-marlin",
        action="store_true",
        help="Use the Marlin int8 lm_head path. This is the default when --rwkv-quant-int8 is enabled.",
    )


def resolve_rwkv_int8_lm_head_flags(
    *,
    rwkv_quant_int8: bool,
    rwkv_int8_fp16_lm_head: bool = False,
    rwkv_int8_lm_head: bool = False,
    rwkv_int8_lm_head_marlin: bool = False,
) -> tuple[bool, bool]:
    selected_modes = (
        int(rwkv_int8_fp16_lm_head)
        + int(rwkv_int8_lm_head)
        + int(rwkv_int8_lm_head_marlin)
    )
    if selected_modes > 1:
        raise ValueError(
            "RWKV int8 lm_head mode is ambiguous. Pick at most one of "
            "--rwkv-int8-fp16-lm-head / --rwkv-int8-lm-head / --rwkv-int8-lm-head-marlin."
        )
    if not rwkv_quant_int8:
        if selected_modes:
            raise ValueError("RWKV int8 lm_head flags require --rwkv-quant-int8.")
        return False, False
    if rwkv_int8_fp16_lm_head:
        return False, False
    if rwkv_int8_lm_head:
        return True, False
    return True, True


def normalize_rwkv_int8_lm_head_flags(
    *,
    rwkv_quant_int8: bool,
    rwkv_int8_fp16_lm_head: bool = False,
    rwkv_int8_lm_head: bool = False,
    rwkv_int8_lm_head_marlin: bool = False,
) -> tuple[bool, bool]:
    if not rwkv_quant_int8:
        if rwkv_int8_fp16_lm_head or rwkv_int8_lm_head or rwkv_int8_lm_head_marlin:
            raise ValueError("RWKV int8 lm_head flags require rwkv_quant_int8=True.")
        return False, False

    if rwkv_int8_fp16_lm_head:
        if rwkv_int8_lm_head or rwkv_int8_lm_head_marlin:
            raise ValueError("RWKV int8 fp16 lm_head is incompatible with int8 lm_head flags.")
        return False, False

    if rwkv_int8_lm_head_marlin:
        return True, True
    if rwkv_int8_lm_head:
        return True, False
    return True, True


def describe_rwkv_int8_mode(
    *,
    rwkv_quant_int8: bool,
    rwkv_quant_int8_lm_head: bool,
    rwkv_quant_int8_lm_head_marlin: bool,
) -> str:
    if not rwkv_quant_int8:
        return "fp16"
    if not rwkv_quant_int8_lm_head:
        return "int8_fp16_lm_head"
    if rwkv_quant_int8_lm_head_marlin:
        return "int8_marlin_lm_head"
    return "int8_cublas_lm_head"
