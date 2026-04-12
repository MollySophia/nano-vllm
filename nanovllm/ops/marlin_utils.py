from __future__ import annotations

import os

import numpy as np
import torch

from nanovllm.ops.marlin import marlin_gemm
from nanovllm.ops.marlin_scalar_type import ScalarType, scalar_types

GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
USE_FP32_REDUCE_DEFAULT = True


def verify_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> None:
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(
            "Weight output_size_per_partition = "
            f"{output_size_per_partition} is not divisible by "
            f"min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}."
        )
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(
            "Weight input_size_per_partition = "
            f"{input_size_per_partition} is not divisible by "
            f"min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}."
        )
    if group_size < input_size and input_size_per_partition % group_size != 0:
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition} "
            f"is not divisible by group_size = {group_size}."
        )


def marlin_make_workspace_new(
    device: torch.device, max_blocks_per_sm: int = 1
) -> torch.Tensor:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(
        sms * max_blocks_per_sm,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )


def rtn_quantize(
    tensor: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_present = tensor.dim() == 3
    if not batch_present:
        tensor = tensor.unsqueeze(0)

    q_range = 2**num_bits
    num_groups = (
        tensor.shape[1] * tensor.shape[2] // group_size
        if group_size != -1
        else tensor.shape[1]
    )
    input_flat = tensor.reshape(tensor.shape[0], num_groups, -1)
    input_min = torch.min(input_flat, dim=2, keepdim=True)[0]
    input_max = torch.max(input_flat, dim=2, keepdim=True)[0]
    input_max_abs = torch.max(input_min.abs(), input_max.abs())
    scale = input_max_abs * 2.0 / (q_range - 1)
    scaled_input = input_flat / scale
    scaled_input = scaled_input.round()
    scaled_input += q_range // 2
    scaled_input = scaled_input.clamp(0, q_range - 1)

    scale = scale.reshape(tensor.shape[0], tensor.shape[1], -1).contiguous()
    inputs_q = scaled_input.reshape(tensor.shape).to(torch.uint8).contiguous()

    if num_bits == 4:
        inputs_q = (inputs_q[:, :, 1::2] << 4) | (inputs_q[:, :, ::2] & 0xF)
        inputs_q = inputs_q.reshape(tensor.shape[0], tensor.shape[1] // 2, tensor.shape[2])
        inputs_q = inputs_q.contiguous()

    if not batch_present:
        inputs_q = inputs_q.squeeze(0)
        scale = scale.squeeze(0)

    return inputs_q, scale


def _get_perms():
    perm: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm_arr = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm_arr = perm_arr.reshape((-1, 8))[:, interleave].ravel()
    perm_tensor = torch.from_numpy(perm_arr)

    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm_tensor, scale_perm, scale_perm_single


_PERM, _SCALE_PERM, _SCALE_PERM_SINGLE = _get_perms()


def pack_for_marlin(
    weight: torch.Tensor,
    scale: torch.Tensor,
    qbits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = weight.shape[0]
    n = weight.size(1)
    k = weight.size(2)
    groupsize = k // scale.size(2)

    tile = 16
    s = scale.permute(0, 2, 1)
    w = weight.permute(0, 2, 1)
    if groupsize != k:
        w = w.reshape((batch, -1, groupsize, n))
        w = w.permute(0, 2, 1, 3)
        w = w.reshape((batch, groupsize, -1))
        s = s.reshape((batch, 1, -1))

    if groupsize != k:
        w = w.reshape((batch, groupsize, -1, n))
        w = w.permute(0, 2, 1, 3)
        w = w.reshape((batch, k, n)).contiguous()
        s = s.reshape((batch, -1, len(_SCALE_PERM)))[:, :, _SCALE_PERM]
    else:
        s = s.reshape((batch, -1, len(_SCALE_PERM_SINGLE)))[:, :, _SCALE_PERM_SINGLE]

    s = s.reshape((batch, -1, n)).contiguous()
    w = w.reshape((batch, k // tile, tile, n // tile, tile))
    w = w.permute((0, 1, 3, 2, 4))
    w = w.reshape((batch, k // tile, n * tile))
    perm = _PERM.to(device=w.device)
    res = w.reshape((batch, -1, perm.numel()))[:, :, perm].reshape(w.shape)

    if qbits == 4:
        q = torch.zeros(
            (batch, res.shape[1], res.shape[2] // 2),
            dtype=torch.int8,
            device=w.device,
        )
        for i in range(2):
            q |= res[:, :, i::2] << 4 * i
        q = q.reshape(batch, -1, n).contiguous()
    else:
        q = res.clone()
        q[:, :, 2::8] = res[:, :, 4::8]
        q[:, :, 3::8] = res[:, :, 5::8]
        q[:, :, 4::8] = res[:, :, 2::8]
        q[:, :, 5::8] = res[:, :, 3::8]
        q = q.reshape(batch, -1, n).to(torch.int8).contiguous()

    return q, s


def repack_8bit_into_32bit(input_tensor: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(
        (input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2] // 4),
        dtype=torch.int32,
        device=input_tensor.device,
    )
    for i in range(4):
        output |= (input_tensor[:, :, i::4] & 0xFF).to(torch.int32) << (8 * i)
    return output


def repack_weights(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    weight_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_present = qweight.dim() == 3
    if not batch_present:
        qweight = qweight.unsqueeze(0)
        scale = scale.unsqueeze(0)

    if weight_bits == 4:
        qweight_unpacked = torch.empty(
            (qweight.shape[0], qweight.shape[1] * 2, qweight.shape[2]),
            dtype=torch.uint8,
            device=qweight.device,
        )
        for i in range(2):
            qweight_unpacked[:, :, i::2] = ((qweight << (4 * (1 - i))) >> 4).reshape(
                qweight.shape[0], qweight.shape[1] * 2, qweight.shape[2] // 2
            )
    else:
        qweight_unpacked = qweight

    qweight_packed, scale_packed = pack_for_marlin(qweight_unpacked, scale, weight_bits)
    qweight_repacked = repack_8bit_into_32bit(qweight_packed.to(torch.uint8))
    qweight_reshaped = qweight_repacked.reshape(qweight.shape[0], qweight.shape[2] // 16, -1)

    if not batch_present:
        qweight_reshaped = qweight_reshaped.squeeze(0)
        scale_packed = scale_packed.squeeze(0)

    return qweight_reshaped, scale_packed


def _use_atomic_add_from_env() -> bool:
    value = os.environ.get(
        "NANOVLLM_MARLIN_USE_ATOMIC_ADD",
        os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", "0"),
    )
    return value.lower() in ("1", "true", "yes", "on")


def should_use_atomic_add_reduce(
    m: int,
    n: int,
    k: int,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    if n >= 2048 or k < 2048 or device.type != "cuda":
        return False
    if not _use_atomic_add_from_env():
        return False
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        return False
    return True


def apply_rtn_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    quant_type: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_global_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
    input_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if input_dtype is not None:
        raise NotImplementedError("The local Marlin path only supports fp16/bf16 activations.")

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)
    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )
    output = marlin_gemm(
        reshaped_x,
        None,
        weight,
        bias,
        weight_scale,
        None,
        None,
        None,
        None,
        None,
        workspace,
        quant_type,
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )
    return output.reshape(out_shape)
