import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovllm.layers.layernorm import LayerNorm
from nanovllm.layers.linear import RowParallelLinear, ColumnParallelLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import get_context
from nanovllm.ops.rwkv7_cuda import (
    ensure_loaded as ensure_rwkv7_cuda_loaded,
    wkv7_one as wkv7_one_cuda,
    wkv7_one_batch as wkv7_one_batch_cuda,
    wkv7_seq as wkv7_seq_cuda,
    wkv7_seq_batch as wkv7_seq_batch_cuda,
)


# Constants for w transformation (from Albatross CUDA kernel)
_NEXP_HALF_LOG2_E = -0.8750387749145276   # -log2(e) / 2 / something; ensures decay in (0.547, 1)
_NLOG2_E = -1.4426950408889634            # -log2(e)
_LN2 = math.log(2.0)
_NEXP_HALF = _NEXP_HALF_LOG2_E * _LN2
_NLOG2E_LN2 = _NLOG2_E * _LN2
_TWO_TO_NEG_41 = 4.547473508864641e-13
_RO1_I32 = -1640531527  # int32((int)2654435769) in CUDA code


def _maybe_compile_rwkv_helper(fn):
    if not hasattr(torch, "jit"):
        return fn
    try:
        return torch.jit.script(fn)
    except Exception:
        return fn


def _rwkv7_tmix_one_impl(
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    x: torch.Tensor,
    x_prev: torch.Tensor,
    v_first: torch.Tensor | None,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    r_k: torch.Tensor,
    receptance_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    output_weight: torch.Tensor,
    ln_x_weight: torch.Tensor,
    ln_x_bias: torch.Tensor,
):
    xx = x_prev - x
    xr = torch.addcmul(x, xx, x_r)
    xw = torch.addcmul(x, xx, x_w)
    xk = torch.addcmul(x, xx, x_k)
    xv = torch.addcmul(x, xx, x_v)
    xa = torch.addcmul(x, xx, x_a)
    xg = torch.addcmul(x, xx, x_g)

    r = F.linear(xr, receptance_weight)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, key_weight)
    v = F.linear(xv, value_weight)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(-1, num_heads, head_dim), dim=-1, p=2.0).view_as(k)
    k = k * (1 + (a - 1) * k_a)
    kka = kk * a

    if layer_idx == 0:
        v_first_out = v
    else:
        assert v_first is not None
        v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0))
        v_first_out = v_first

    return r, w, k, v, a, kk, kka, g, v_first_out, xx


def _rwkv7_tmix_one_post_impl(
    num_heads: int,
    head_dim: int,
    wkv_out: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    r_k: torch.Tensor,
    output_weight: torch.Tensor,
    ln_x_weight: torch.Tensor,
    ln_x_bias: torch.Tensor,
):
    y = F.group_norm(wkv_out.view_as(r), num_groups=num_heads, weight=ln_x_weight, bias=ln_x_bias, eps=64e-5)
    y = y + (
        ((r * k * r_k).view(-1, num_heads, head_dim).sum(dim=-1, keepdim=True) * v.view(-1, num_heads, head_dim)).view_as(r)
    )
    return F.linear(y * g, output_weight)


def _rwkv7_ffn_decode_impl(
    x: torch.Tensor,
    xx: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
):
    k = torch.addcmul(x, xx, x_k)
    k = torch.relu(F.linear(k, key_weight)) ** 2
    return k @ value_weight


def _rwkv7_tmix_seq_batch_impl(
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    x: torch.Tensor,
    x_prev: torch.Tensor,
    v_first: torch.Tensor | None,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    receptance_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
):
    bsz, seqlen, c = x.shape
    xx = x_prev - x
    xr = torch.addcmul(x, xx, x_r.view(1, 1, c))
    xw = torch.addcmul(x, xx, x_w.view(1, 1, c))
    xk = torch.addcmul(x, xx, x_k.view(1, 1, c))
    xv = torch.addcmul(x, xx, x_v.view(1, 1, c))
    xa = torch.addcmul(x, xx, x_a.view(1, 1, c))
    xg = torch.addcmul(x, xx, x_g.view(1, 1, c))
    r = F.linear(xr, receptance_weight)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, key_weight)
    v = F.linear(xv, value_weight)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))

    kk = F.normalize((k * k_k.view(1, 1, c)).view(bsz, seqlen, num_heads, head_dim), dim=-1, p=2.0).view(bsz, seqlen, c)
    k = k * (1 + (a - 1) * k_a.view(1, 1, c))
    kka = kk * a
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)

    if layer_idx == 0:
        v_first_out = v
    else:
        assert v_first is not None
        v_mix = torch.sigmoid(F.linear(F.linear(xv.view(bsz, seqlen, -1), v1), v2, bias=v0))
        v = v + (v_first - v) * v_mix
        v_first_out = v_first

    return r, w, k, v, kk, kka, g, v_first_out


_rwkv7_tmix_one = _maybe_compile_rwkv_helper(_rwkv7_tmix_one_impl)
_rwkv7_tmix_one_post = _maybe_compile_rwkv_helper(_rwkv7_tmix_one_post_impl)
_rwkv7_ffn_decode = _maybe_compile_rwkv_helper(_rwkv7_ffn_decode_impl)
_rwkv7_tmix_seq_batch = _maybe_compile_rwkv_helper(_rwkv7_tmix_seq_batch_impl)


@torch.jit.script
def _rwkv7_decode_block_batch_contiguous(
    x: torch.Tensor,
    att_tokenshift_cache: torch.Tensor,
    state_cache: torch.Tensor,
    ffn_tokenshift_cache: torch.Tensor,
    elapsed_t: torch.Tensor,
    v_first: torch.Tensor | None,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    r_k: torch.Tensor,
    receptance_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    output_weight: torch.Tensor,
    ln_x_weight: torch.Tensor,
    ln_x_bias: torch.Tensor,
    ln1_gamma: torch.Tensor,
    ln1_beta: torch.Tensor,
    ln1_eps: float,
    ln2_gamma: torch.Tensor,
    ln2_beta: torch.Tensor,
    ln2_eps: float,
    ffn_x_k: torch.Tensor,
    ffn_key_weight: torch.Tensor,
    ffn_value_weight: torch.Tensor,
):
    bsz, c = x.shape
    h = F.layer_norm(x, (c,), ln1_gamma, ln1_beta, ln1_eps)
    h3 = h.unsqueeze(1)
    x_prev = att_tokenshift_cache.unsqueeze(1)
    att_tokenshift_cache.copy_(h)
    r, w, k, v, kk, kka, g, v_first = _rwkv7_tmix_seq_batch(
        layer_idx,
        num_heads,
        head_dim,
        h3,
        x_prev,
        v_first,
        x_r,
        x_w,
        x_k,
        x_v,
        x_a,
        x_g,
        w0,
        w1,
        w2,
        a0,
        a1,
        a2,
        v0,
        v1,
        v2,
        g1,
        g2,
        k_k,
        k_a,
        receptance_weight,
        key_weight,
        value_weight,
    )
    y = torch.ops.nanovllm.rwkv7_seq_batch(
        state_cache,
        state_cache,
        r,
        w,
        k,
        v,
        -kk,
        kka,
        elapsed_t,
    )
    y = F.group_norm(y.view(bsz, c), num_groups=num_heads, weight=ln_x_weight, bias=ln_x_bias, eps=64e-5)
    y = y + (((r * k * r_k.view(1, 1, c)).view(bsz, 1, num_heads, head_dim).sum(dim=-1, keepdim=True) * v.view(bsz, 1, num_heads, head_dim)).view(bsz, c))
    y = F.linear(y * g.view(bsz, c), output_weight)
    x = x + y

    h2 = F.layer_norm(x, (c,), ln2_gamma, ln2_beta, ln2_eps)
    xx = ffn_tokenshift_cache - h2
    ffn_tokenshift_cache.copy_(h2)
    x = x + _rwkv7_ffn_decode(h2, xx, ffn_x_k, ffn_key_weight, ffn_value_weight)
    return x, v_first


def wkv7_one_step(state: torch.Tensor, r: torch.Tensor, w: torch.Tensor, k: torch.Tensor,
                  v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, position: int | torch.Tensor) -> torch.Tensor:
    """
    Single step WKV-7 computation with stable decay.
    Args:
        state: [H, N, N] - RNN state (modified in-place)
        r: [H, N] - receptance
        w: [H, N] - decay parameter
        k: [H, N] - key (k_mod)
        v: [H, N] - value
        a: [H, N] - mixing vector (in RWKV-7 callsite this is -kk)
        b: [H, N] - mixing vector (in RWKV-7 callsite this is kk*a)
    Returns:
        y: [H, N] - output
    """
    # Transform w to stable decay factor in (0.547, 1.0)
    # This prevents state explosion; the CUDA kernel always keeps decay < 1
    # Use exp instead of exp2 to avoid NVRTC dependency in some environments.
    decay = torch.exp(_NEXP_HALF / (1.0 + torch.exp(_NLOG2E_LN2 * w)))  # [H, N], values in (0.547, 1.0)

    # RWKV-7 CUDA kernel equivalent:
    # sa_i = sum_j a_j * state_{i,j}
    # state_{i,j} = state_{i,j} * (decay_j + rotator1(pos)) + k_j * v_i + sa_i * b_j
    # y_i = sum_j state_{i,j} * r_j   (computed after state update)
    out_dtype = r.dtype
    a = a.to(state.dtype)
    b = b.to(state.dtype)
    k = k.to(state.dtype)
    v = v.to(state.dtype)
    r = r.to(state.dtype)
    decay = decay.to(state.dtype)
    pos_i32 = torch.as_tensor(position, device=state.device, dtype=torch.int32)
    rot = (pos_i32 * _RO1_I32).to(torch.float32) * _TWO_TO_NEG_41
    decay = decay + rot.to(decay.dtype)

    sa = torch.einsum('hmn,hn->hm', state, a)  # [H, N]
    state.mul_(decay.unsqueeze(-2))  # decay applies on column dimension j
    state.add_(v.unsqueeze(-1) * k.unsqueeze(-2))  # outer(v, k)
    state.add_(sa.unsqueeze(-1) * b.unsqueeze(-2))  # outer(sa, b)
    y = torch.einsum('hmn,hn->hm', state, r)
    return y.to(out_dtype)


def wkv7_sequence(state: torch.Tensor, r: torch.Tensor, w: torch.Tensor, k: torch.Tensor,
                  v: torch.Tensor, kk: torch.Tensor, kka: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Sequence WKV-7 computation (prefill).
    Args:
        state: [num_heads, head_dim, head_dim] - initial state, will be updated in-place
        r: [T, num_heads, head_dim] - receptance
        w: [T, num_heads, head_dim] - decay
        k: [T, num_heads, head_dim] - key (k_mod)
        v: [T, num_heads, head_dim] - value
        kk: [T, num_heads, head_dim] - normalized key for state erase
        kka: [T, num_heads, head_dim] - kk * a for state write
    Returns:
        output: [T, num_heads, head_dim]
    """
    T, H, N = r.shape
    outputs = []

    for t in range(T):
        y = wkv7_one_step(state, r[t], w[t], k[t], v[t], -kk[t], kka[t], positions[t])
        outputs.append(y)

    return torch.stack(outputs, dim=0)


def _build_slot_runs(slots: torch.Tensor):
    order = torch.argsort(slots)
    sorted_slots = slots[order]
    sorted_slots_list = sorted_slots.tolist()
    runs = []
    start = 0
    n = len(sorted_slots_list)
    while start < n:
        end = start + 1
        while end < n and sorted_slots_list[end] == sorted_slots_list[end - 1] + 1:
            end += 1
        runs.append((start, end, sorted_slots_list[start], sorted_slots_list[end - 1] + 1))
        start = end
    has_duplicates = any(
        sorted_slots_list[i] == sorted_slots_list[i - 1]
        for i in range(1, n)
    )
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device, dtype=order.dtype)
    return order, inverse, runs, has_duplicates


def _is_contiguous_in_order(slots: torch.Tensor) -> bool:
    if slots.numel() <= 1:
        return True
    expected = torch.arange(
        int(slots[0].item()),
        int(slots[0].item()) + slots.numel(),
        device=slots.device,
        dtype=slots.dtype,
    )
    return torch.equal(slots, expected)


def _wkv7_one_batch_inplace_by_slot_runs(
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    kka: torch.Tensor,
    positions: torch.Tensor,
):
    if _is_contiguous_in_order(slot_mapping):
        slot_start = int(slot_mapping[0].item())
        slot_end = slot_start + slot_mapping.numel()
        return wkv7_one_batch_cuda(
            state_cache[slot_start:slot_end],
            state_cache[slot_start:slot_end],
            r,
            w,
            k,
            v,
            kk,
            kka,
            positions,
        )

    order, inverse, runs, has_duplicates = _build_slot_runs(slot_mapping)
    if has_duplicates:
        state = state_cache[slot_mapping].contiguous()
        y = wkv7_one_batch_cuda(state, state, r, w, k, v, kk, kka, positions)
        state_cache[slot_mapping] = state
        return y

    r_sorted = r[order]
    w_sorted = w[order]
    k_sorted = k[order]
    v_sorted = v[order]
    kk_sorted = kk[order]
    kka_sorted = kka[order]
    positions_sorted = positions[order]
    y_sorted = torch.empty_like(r_sorted)

    for start, end, slot_start, slot_end in runs:
        y_sorted[start:end] = wkv7_one_batch_cuda(
            state_cache[slot_start:slot_end],
            state_cache[slot_start:slot_end],
            r_sorted[start:end],
            w_sorted[start:end],
            k_sorted[start:end],
            v_sorted[start:end],
            kk_sorted[start:end],
            kka_sorted[start:end],
            positions_sorted[start:end],
        )

    return y_sorted[inverse]


def _wkv7_one_batch_out_by_slot_runs(
    state_cache: torch.Tensor,
    slot_mapping_in: torch.Tensor,
    slot_mapping_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    kka: torch.Tensor,
    positions: torch.Tensor,
):
    if _is_contiguous_in_order(slot_mapping_in) and _is_contiguous_in_order(slot_mapping_out):
        slot_in_start = int(slot_mapping_in[0].item())
        slot_out_start = int(slot_mapping_out[0].item())
        slot_count = slot_mapping_in.numel()
        return wkv7_one_batch_cuda(
            state_cache[slot_in_start:slot_in_start + slot_count],
            state_cache[slot_out_start:slot_out_start + slot_count],
            r,
            w,
            k,
            v,
            kk,
            kka,
            positions,
        )

    order, inverse, runs, has_duplicates = _build_slot_runs(slot_mapping_out)
    if has_duplicates:
        state_in = state_cache[slot_mapping_in].contiguous()
        state_out = state_cache[slot_mapping_out].contiguous()
        y = wkv7_one_batch_cuda(state_in, state_out, r, w, k, v, kk, kka, positions)
        state_cache[slot_mapping_out] = state_out
        return y

    slot_in_sorted = slot_mapping_in[order]
    state_in_sorted = state_cache[slot_in_sorted].contiguous()
    r_sorted = r[order]
    w_sorted = w[order]
    k_sorted = k[order]
    v_sorted = v[order]
    kk_sorted = kk[order]
    kka_sorted = kka[order]
    positions_sorted = positions[order]
    y_sorted = torch.empty_like(r_sorted)

    for start, end, slot_start, slot_end in runs:
        y_sorted[start:end] = wkv7_one_batch_cuda(
            state_in_sorted[start:end],
            state_cache[slot_start:slot_end],
            r_sorted[start:end],
            w_sorted[start:end],
            k_sorted[start:end],
            v_sorted[start:end],
            kk_sorted[start:end],
            kka_sorted[start:end],
            positions_sorted[start:end],
        )

    return y_sorted[inverse]


def _wkv7_seq_batch_inplace_by_slot_runs(
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    kka: torch.Tensor,
    elapsed_t: torch.Tensor,
):
    if _is_contiguous_in_order(slot_mapping):
        slot_start = int(slot_mapping[0].item())
        slot_end = slot_start + slot_mapping.numel()
        return wkv7_seq_batch_cuda(
            state_cache[slot_start:slot_end],
            state_cache[slot_start:slot_end],
            r,
            w,
            k,
            v,
            kk,
            kka,
            elapsed_t,
        )

    order, inverse, runs, has_duplicates = _build_slot_runs(slot_mapping)
    if has_duplicates:
        state = state_cache[slot_mapping].contiguous()
        y = wkv7_seq_batch_cuda(state, state, r, w, k, v, kk, kka, elapsed_t)
        state_cache[slot_mapping] = state
        return y

    r_sorted = r[order]
    w_sorted = w[order]
    k_sorted = k[order]
    v_sorted = v[order]
    kk_sorted = kk[order]
    kka_sorted = kka[order]
    elapsed_sorted = elapsed_t[order]
    y_sorted = torch.empty_like(r_sorted)

    for start, end, slot_start, slot_end in runs:
        y_sorted[start:end] = wkv7_seq_batch_cuda(
            state_cache[slot_start:slot_end],
            state_cache[slot_start:slot_end],
            r_sorted[start:end],
            w_sorted[start:end],
            k_sorted[start:end],
            v_sorted[start:end],
            kk_sorted[start:end],
            kka_sorted[start:end],
            elapsed_sorted[start:end],
        )

    return y_sorted[inverse]


def _wkv7_seq_batch_out_by_slot_runs(
    state_cache: torch.Tensor,
    slot_mapping_in: torch.Tensor,
    slot_mapping_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    kka: torch.Tensor,
    elapsed_t: torch.Tensor,
):
    if _is_contiguous_in_order(slot_mapping_in) and _is_contiguous_in_order(slot_mapping_out):
        slot_in_start = int(slot_mapping_in[0].item())
        slot_out_start = int(slot_mapping_out[0].item())
        slot_count = slot_mapping_in.numel()
        return wkv7_seq_batch_cuda(
            state_cache[slot_in_start:slot_in_start + slot_count],
            state_cache[slot_out_start:slot_out_start + slot_count],
            r,
            w,
            k,
            v,
            kk,
            kka,
            elapsed_t,
        )

    order, inverse, runs, has_duplicates = _build_slot_runs(slot_mapping_out)
    if has_duplicates:
        state_in = state_cache[slot_mapping_in].contiguous()
        state_out = state_cache[slot_mapping_out].contiguous()
        y = wkv7_seq_batch_cuda(state_in, state_out, r, w, k, v, kk, kka, elapsed_t)
        state_cache[slot_mapping_out] = state_out
        return y

    slot_in_sorted = slot_mapping_in[order]
    state_in_sorted = state_cache[slot_in_sorted].contiguous()
    r_sorted = r[order]
    w_sorted = w[order]
    k_sorted = k[order]
    v_sorted = v[order]
    kk_sorted = kk[order]
    kka_sorted = kka[order]
    elapsed_sorted = elapsed_t[order]
    y_sorted = torch.empty_like(r_sorted)

    for start, end, slot_start, slot_end in runs:
        y_sorted[start:end] = wkv7_seq_batch_cuda(
            state_in_sorted[start:end],
            state_cache[slot_start:slot_end],
            r_sorted[start:end],
            w_sorted[start:end],
            k_sorted[start:end],
            v_sorted[start:end],
            kk_sorted[start:end],
            kka_sorted[start:end],
            elapsed_sorted[start:end],
        )

    return y_sorted[inverse]


class RWKV7Attention(nn.Module):
    """RWKV-7 "Goose" linear attention implementation."""

    def __init__(self, layer_idx: int, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # State buffers (initialized in allocate_state_cache)
        self.att_tokenshift_cache = None  # [num_blocks, hidden_size]
        self.state_cache = None  # [num_blocks, num_heads, head_dim, head_dim]

        # TMix parameters will be registered as buffers in _load_weights
        # (not defined here to avoid conflict with register_buffer)
        self._cuda_kernel_ready = False
        self._cuda_kernel_attempted = False

    def _maybe_init_cuda_kernel(self):
        if self._cuda_kernel_attempted:
            return
        self._cuda_kernel_attempted = True
        try:
            ensure_rwkv7_cuda_loaded(self.head_dim)
            self._cuda_kernel_ready = True
        except Exception:
            self._cuda_kernel_ready = False

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        is_prefill: bool,
        v_first: torch.Tensor | None = None,
        att_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [total_tokens, hidden_size] for prefill, [batch, hidden_size] for decode
            positions: token positions
            is_prefill: whether this is prefill phase
            v_first: value from layer 0 for v-mixing
        Returns:
            (output tensor of same shape as x, v_first)
        """
        context = get_context()
        slot_mapping_in = context.slot_mapping_in
        slot_mapping_out = context.slot_mapping_out

        if is_prefill:
            return self._forward_prefill(x, positions, slot_mapping_in, slot_mapping_out, v_first, att_mask)
        else:
            return self._forward_decode(x, positions, slot_mapping_in, slot_mapping_out, v_first)

    def _forward_prefill(self, x: torch.Tensor, positions: torch.Tensor, slot_mapping_in: torch.Tensor, slot_mapping_out: torch.Tensor, v_first: torch.Tensor | None = None, att_mask: torch.Tensor | None = None):
        """Prefill."""
        if x.dim() == 3:
            if att_mask is None:
                return self._forward_prefill_batch_same_length(x, slot_mapping_in, slot_mapping_out, v_first)
            return self._forward_prefill_batch_right(x, slot_mapping_in, slot_mapping_out, v_first, att_mask)

        """Varlen prefill: x is [total_tokens, hidden_size]"""
        T, C = x.shape
        H, N = self.num_heads, self.head_dim

        # Get previous tokenshift values
        seq_starts = torch.cat([
            slot_mapping_in.new_ones(1, dtype=torch.bool),
            slot_mapping_in[1:] != slot_mapping_in[:-1],
        ])
        x_prev = x.clone()
        x_prev[1:] = x[:-1]
        x_prev[seq_starts] = self.att_tokenshift_cache[slot_mapping_in[seq_starts]].to(x.dtype)

        # Update tokenshift cache with last token of each sequence
        seq_ends = torch.cat([
            slot_mapping_in[:-1] != slot_mapping_in[1:],
            slot_mapping_in.new_ones(1, dtype=torch.bool),
        ])
        self.att_tokenshift_cache[slot_mapping_out[seq_ends]] = x[seq_ends].to(self.att_tokenshift_cache.dtype)

        # Compute xx = x_prev - x (token-shift difference)
        xx = x_prev - x

        # Compute shifted inputs
        mix = x + xx * self.x_r
        r = F.linear(mix, self.receptance_weight)

        mix = x + xx * self.x_w
        w = F.linear(torch.tanh(F.linear(mix, self.w1)), self.w2, bias=self.w0)

        mix = x + xx * self.x_k
        k = F.linear(mix, self.key_weight)

        xv = x + xx * self.x_v
        v = F.linear(xv, self.value_weight)

        mix = x + xx * self.x_a
        a = torch.sigmoid(F.linear(F.linear(mix, self.a1), self.a2, bias=self.a0))

        mix = x + xx * self.x_g
        g = F.linear(torch.sigmoid(F.linear(mix, self.g1)), self.g2)

        # Reshape for multi-head
        r = r.view(T, H, N)
        w = w.view(T, H, N)
        k = k.view(T, H, N)
        v = v.view(T, H, N)
        a = a.view(T, H, N)
        g = g.view(T, H, N)

        # Compute kk and kka
        # k_k is [hidden_size], need to broadcast with k [T, H, N]
        k_k = self.k_k.view(1, H, N)
        k_a = self.k_a.view(1, H, N)
        kk = F.normalize(k * k_k, dim=-1, p=2.0)
        k = k * (1 + (a - 1) * k_a)
        kka = kk * a

        # Handle v_first for layer 0+
        if self.layer_idx == 0:
            v_first = v
        else:
            assert v_first is not None
            v_mix = torch.sigmoid(F.linear(F.linear(xv.view(T, -1), self.v1), self.v2, bias=self.v0))
            v = v + (v_first - v) * v_mix.view(T, H, N)

        # WKV computation - process per sequence to handle state correctly
        # For varlen, we need to group by sequence
        # Preserve token order: process contiguous per-sequence spans in-place.
        self._maybe_init_cuda_kernel()
        y = torch.empty_like(r)
        seq_starts = torch.cat([
            slot_mapping_in.new_zeros(1, dtype=torch.bool).fill_(True),
            slot_mapping_in[1:] != slot_mapping_in[:-1],
        ])
        start_idx = torch.where(seq_starts)[0]
        end_idx = torch.cat([start_idx[1:], start_idx.new_tensor([T])])
        for start, end in zip(start_idx.tolist(), end_idx.tolist()):
            slot = slot_mapping_in[start].item()
            state = self.state_cache[slot]  # [H, N, N]
            if self._cuda_kernel_ready and x.is_cuda and x.dtype == torch.float16:
                c = H * N
                y_seq = wkv7_seq_cuda(
                    state,
                    state,
                    r[start:end].reshape(end - start, c),
                    w[start:end].reshape(end - start, c),
                    k[start:end].reshape(end - start, c),
                    v[start:end].reshape(end - start, c),
                    (-kk[start:end]).reshape(end - start, c),
                    kka[start:end].reshape(end - start, c),
                    int(positions[start].item()),
                )
                y[start:end] = y_seq.view(end - start, H, N)
            else:
                y[start:end] = wkv7_sequence(state, r[start:end], w[start:end], k[start:end], v[start:end], -kk[start:end], kka[start:end], positions[start:end])

        # Group norm
        y = F.group_norm(y.view(T, H * N), num_groups=H, weight=self.ln_x_weight, bias=self.ln_x_bias, eps=64e-5)
        y = y.view(T, H, N)

        # Add receptance term
        # r_k is [hidden_size], need to broadcast with r and k
        r_k = self.r_k.view(1, H, N)
        y = y + ((r * k * r_k).sum(dim=-1, keepdim=True) * v)

        # Apply gate and output projection
        y = y.view(T, H * N) * g.view(T, H * N)
        y = F.linear(y, self.output_weight)

        return y, v_first

    def _forward_prefill_batch_right(self, x: torch.Tensor, slot_mapping_in: torch.Tensor, slot_mapping_out: torch.Tensor, v_first: torch.Tensor | None = None, att_mask: torch.Tensor | None = None):
        """Right-padded batch prefill: x is [batch, seqlen, hidden_size]."""
        B, T, C = x.shape
        H, N = self.num_heads, self.head_dim
        context = get_context()
        context_lens = context.context_lens
        if att_mask is None:
            att_mask = (
                torch.arange(T, device=x.device, dtype=torch.int32).unsqueeze(0) <
                (T - context_lens).unsqueeze(1)
            ).unsqueeze(2)

        x_prev = torch.cat((self.att_tokenshift_cache[slot_mapping_in].to(x.dtype).unsqueeze(1), x[:, :-1, :]), dim=1)
        xx = x_prev - x
        self.att_tokenshift_cache[slot_mapping_out] = x[:, -1, :].to(self.att_tokenshift_cache.dtype)
        xr = torch.addcmul(x, xx, self.x_r.view(1, 1, C))
        xw = torch.addcmul(x, xx, self.x_w.view(1, 1, C))
        xk = torch.addcmul(x, xx, self.x_k.view(1, 1, C))
        xv = torch.addcmul(x, xx, self.x_v.view(1, 1, C))
        xa = torch.addcmul(x, xx, self.x_a.view(1, 1, C))
        r = F.linear(xr, self.receptance_weight)
        w = F.linear(
            torch.tanh(F.linear(xw, self.w1)),
            self.w2,
            bias=self.w0,
        )
        k = F.linear(xk, self.key_weight)
        v = F.linear(xv, self.value_weight)
        a = torch.sigmoid(
            F.linear(
                F.linear(xa, self.a1),
                self.a2,
                bias=self.a0,
            )
        )

        k_k = self.k_k.view(1, 1, C)
        k_a = self.k_a.view(1, 1, C)
        kk = F.normalize((k * k_k).view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        kk.masked_fill_(att_mask, 0)
        k = k * (1 + (a - 1) * k_a)
        kka = kk * a

        if self.layer_idx == 0:
            v_first = v
        else:
            assert v_first is not None
            v_mix = torch.sigmoid(
                F.linear(
                    F.linear(xv.view(B, T, -1), self.v1),
                    self.v2,
                    bias=self.v0,
                )
            )
            v = v + (v_first - v) * v_mix

        self._maybe_init_cuda_kernel()
        if self._cuda_kernel_ready and x.is_cuda and x.dtype == torch.float16:
            elapsed_t = context_lens - T
            if torch.equal(slot_mapping_in, slot_mapping_out):
                y = _wkv7_seq_batch_inplace_by_slot_runs(
                    self.state_cache,
                    slot_mapping_in,
                    r,
                    w,
                    k,
                    v,
                    -kk,
                    kka,
                    elapsed_t,
                )
            else:
                y = _wkv7_seq_batch_out_by_slot_runs(
                    self.state_cache,
                    slot_mapping_in,
                    slot_mapping_out,
                    r,
                    w,
                    k,
                    v,
                    -kk,
                    kka,
                    elapsed_t,
                )
        else:
            r_hn = r.view(B, T, H, N)
            w_hn = w.view(B, T, H, N)
            k_hn = k.view(B, T, H, N)
            v_hn = v.view(B, T, H, N)
            kk_hn = kk.view(B, T, H, N)
            kka_hn = kka.view(B, T, H, N)
            y = torch.zeros_like(r_hn)
            for i in range(B):
                seqlen = int(context_lens[i].item())
                start = T - seqlen
                slot = slot_mapping_in[i].item()
                state = self.state_cache[slot]
                y_i = wkv7_sequence(
                    state,
                    r_hn[i, start:],
                    w_hn[i, start:],
                    k_hn[i, start:],
                    v_hn[i, start:],
                    -kk_hn[i, start:],
                    kka_hn[i, start:],
                    torch.arange(seqlen, device=x.device, dtype=torch.int64),
                )
                y[i, start:] = y_i
            y = y.view(B, T, C)

        y = F.group_norm(y.view(B * T, C), num_groups=H, weight=self.ln_x_weight, bias=self.ln_x_bias, eps=64e-5).view(B, T, C)
        y = y + (
            (
                (r * k * self.r_k.view(1, 1, C)).view(B, T, H, N).sum(dim=-1, keepdim=True)
                * v.view(B, T, H, N)
            ).view(B, T, C)
        )
        g = F.linear(
            torch.sigmoid(F.linear(torch.addcmul(x, xx, self.x_g), self.g1)),
            self.g2,
        )
        y = F.linear(y * g, self.output_weight)
        return y, v_first

    def _forward_prefill_batch_same_length(self, x: torch.Tensor, slot_mapping_in: torch.Tensor, slot_mapping_out: torch.Tensor, v_first: torch.Tensor | None = None):
        """Same-length batch prefill: x is [batch, seqlen, hidden_size]."""
        B, T, C = x.shape
        H, N = self.num_heads, self.head_dim

        x_prev = torch.cat((self.att_tokenshift_cache[slot_mapping_in].to(x.dtype).unsqueeze(1), x[:, :-1, :]), dim=1)
        self.att_tokenshift_cache[slot_mapping_out] = x[:, -1, :].to(self.att_tokenshift_cache.dtype)
        r, w, k, v, kk, kka, g, v_first = _rwkv7_tmix_seq_batch(
            self.layer_idx,
            H,
            N,
            x,
            x_prev,
            v_first,
            self.x_r,
            self.x_w,
            self.x_k,
            self.x_v,
            self.x_a,
            self.x_g,
            self.w0,
            self.w1,
            self.w2,
            self.a0,
            self.a1,
            self.a2,
            self.v0,
            self.v1,
            self.v2,
            self.g1,
            self.g2,
            self.k_k,
            self.k_a,
            self.receptance_weight,
            self.key_weight,
            self.value_weight,
        )

        self._maybe_init_cuda_kernel()
        if self._cuda_kernel_ready and x.is_cuda and x.dtype == torch.float16:
            elapsed_t = torch.zeros(B, device=x.device, dtype=torch.int32)
            if torch.equal(slot_mapping_in, slot_mapping_out):
                y = _wkv7_seq_batch_inplace_by_slot_runs(
                    self.state_cache,
                    slot_mapping_in,
                    r,
                    w,
                    k,
                    v,
                    -kk,
                    kka,
                    elapsed_t,
                )
            else:
                y = _wkv7_seq_batch_out_by_slot_runs(
                    self.state_cache,
                    slot_mapping_in,
                    slot_mapping_out,
                    r,
                    w,
                    k,
                    v,
                    -kk,
                    kka,
                    elapsed_t,
                )
        else:
            y = torch.zeros_like(r)
            r_hn = r.view(B, T, H, N)
            w_hn = w.view(B, T, H, N)
            k_hn = k.view(B, T, H, N)
            v_hn = v.view(B, T, H, N)
            kk_hn = kk.view(B, T, H, N)
            kka_hn = kka.view(B, T, H, N)
            positions = torch.arange(T, device=x.device, dtype=torch.int64)
            for i in range(B):
                slot = slot_mapping_in[i].item()
                state = self.state_cache[slot]
                y_i = wkv7_sequence(
                    state,
                    r_hn[i],
                    w_hn[i],
                    k_hn[i],
                    v_hn[i],
                    -kk_hn[i],
                    kka_hn[i],
                    positions,
                )
                y[i] = y_i.view(T, C)

        y = F.group_norm(y.view(B * T, C), num_groups=H, weight=self.ln_x_weight, bias=self.ln_x_bias, eps=64e-5).view(B, T, C)
        y = y + (((r * k * self.r_k.view(1, 1, C)).view(B, T, H, N).sum(dim=-1, keepdim=True) * v.view(B, T, H, N)).view(B, T, C))
        y = F.linear(y * g, self.output_weight)
        return y, v_first

    def _forward_decode(self, x: torch.Tensor, positions: torch.Tensor, slot_mapping_in: torch.Tensor, slot_mapping_out: torch.Tensor, v_first: torch.Tensor | None = None):
        """Decode: x is [batch, hidden_size]"""
        if x.size(0) > 1:
            x3 = x.unsqueeze(1)
            v_first3 = None if v_first is None else v_first.unsqueeze(1)
            y3, v_first3 = self._forward_prefill_batch_right(x3, slot_mapping_in, slot_mapping_out, v_first3, None)
            return y3.squeeze(1), None if v_first3 is None else v_first3.squeeze(1)

        B, C = x.shape
        H, N = self.num_heads, self.head_dim

        # Get previous tokenshift and compute difference
        x_prev = self.att_tokenshift_cache[slot_mapping_in].to(x.dtype)

        # Update tokenshift cache
        self.att_tokenshift_cache[slot_mapping_out] = x.to(self.att_tokenshift_cache.dtype)
        r, w, k, v, _, kk, kka, g, v_first, _ = _rwkv7_tmix_one(
            self.layer_idx,
            H,
            N,
            x,
            x_prev,
            v_first,
            self.x_r,
            self.x_w,
            self.x_k,
            self.x_v,
            self.x_a,
            self.x_g,
            self.w0,
            self.w1,
            self.w2,
            self.a0,
            self.a1,
            self.a2,
            self.v0,
            self.v1,
            self.v2,
            self.g1,
            self.g2,
            self.k_k,
            self.k_a,
            self.r_k,
            self.receptance_weight,
            self.key_weight,
            self.value_weight,
            self.output_weight,
            self.ln_x_weight,
            self.ln_x_bias,
        )

        # Run the state update as a single batched kernel invocation.
        self._maybe_init_cuda_kernel()
        if self._cuda_kernel_ready and x.is_cuda and x.dtype == torch.float16:
            if torch.equal(slot_mapping_in, slot_mapping_out):
                y = _wkv7_one_batch_inplace_by_slot_runs(
                    self.state_cache,
                    slot_mapping_in,
                    r,
                    w,
                    k,
                    v,
                    -kk,
                    kka,
                    positions,
                )
            else:
                y = _wkv7_one_batch_out_by_slot_runs(
                    self.state_cache,
                    slot_mapping_in,
                    slot_mapping_out,
                    r,
                    w,
                    k,
                    v,
                    -kk,
                    kka,
                    positions,
                )
        else:
            r_hn = r.view(B, H, N)
            w_hn = w.view(B, H, N)
            k_hn = k.view(B, H, N)
            v_hn = v.view(B, H, N)
            kk_hn = kk.view(B, H, N)
            kka_hn = kka.view(B, H, N)
            outputs = []
            for i in range(B):
                slot = slot_mapping_in[i].item()
                state = self.state_cache[slot]
                y_i = wkv7_one_step(state, r_hn[i], w_hn[i], k_hn[i], v_hn[i], -kk_hn[i], kka_hn[i], positions[i])
                outputs.append(y_i)
            y = torch.stack(outputs, dim=0).view(B, C)

        return _rwkv7_tmix_one_post(
            H,
            N,
            y,
            r,
            k,
            v,
            g,
            self.r_k,
            self.output_weight,
            self.ln_x_weight,
            self.ln_x_bias,
        ), v_first


class RWKV7FeedForward(nn.Module):
    """RWKV-7 channel mixing (FFN)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # State buffer
        self.ffn_tokenshift_cache = None  # [num_blocks, hidden_size]

        # Parameters will be registered as buffers in _load_weights

    def forward(self, x: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        context = get_context()
        slot_mapping_in = context.slot_mapping_in
        slot_mapping_out = context.slot_mapping_out

        if is_prefill:
            return self._forward_prefill(x, slot_mapping_in, slot_mapping_out)
        else:
            return self._forward_decode(x, slot_mapping_in, slot_mapping_out)

    def _forward_prefill(self, x: torch.Tensor, slot_mapping_in: torch.Tensor, slot_mapping_out: torch.Tensor):
        """Varlen prefill"""
        if x.dim() == 3:
            B, T, _ = x.shape
            x_prev = torch.cat((self.ffn_tokenshift_cache[slot_mapping_in].to(x.dtype).unsqueeze(1), x[:, :-1, :]), dim=1)
            self.ffn_tokenshift_cache[slot_mapping_out] = x[:, -1, :].to(self.ffn_tokenshift_cache.dtype)
            xx = x_prev - x
            k = x + xx * self.x_k
            k = torch.relu(F.linear(k, self.key_weight)) ** 2
            return k @ self.value_weight

        seq_starts = torch.cat([
            slot_mapping_in.new_ones(1, dtype=torch.bool),
            slot_mapping_in[1:] != slot_mapping_in[:-1],
        ])
        x_prev = x.clone()
        x_prev[1:] = x[:-1]
        x_prev[seq_starts] = self.ffn_tokenshift_cache[slot_mapping_in[seq_starts]].to(x.dtype)

        seq_ends = torch.cat([
            slot_mapping_in[:-1] != slot_mapping_in[1:],
            slot_mapping_in.new_ones(1, dtype=torch.bool),
        ])
        self.ffn_tokenshift_cache[slot_mapping_out[seq_ends]] = x[seq_ends].to(self.ffn_tokenshift_cache.dtype)

        xx = x_prev - x
        k = x + xx * self.x_k
        k = torch.relu(F.linear(k, self.key_weight)) ** 2
        return k @ self.value_weight

    def _forward_decode(self, x: torch.Tensor, slot_mapping_in: torch.Tensor, slot_mapping_out: torch.Tensor):
        """Decode"""
        if x.dim() == 2 and x.size(0) > 1:
            return self._forward_prefill(x.unsqueeze(1), slot_mapping_in, slot_mapping_out).squeeze(1)

        x_prev = self.ffn_tokenshift_cache[slot_mapping_in].to(x.dtype)
        xx = x_prev - x
        self.ffn_tokenshift_cache[slot_mapping_out] = x.to(self.ffn_tokenshift_cache.dtype)
        return _rwkv7_ffn_decode(x, xx, self.x_k, self.key_weight, self.value_weight)


class RWKV7Block(nn.Module):
    """RWKV-7 transformer block."""

    def __init__(self, layer_idx: int, hidden_size: int, num_heads: int, head_dim: int, intermediate_size: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.ln1 = LayerNorm(hidden_size)
        self.att = RWKV7Attention(layer_idx, hidden_size, num_heads, head_dim)
        self.ln2 = LayerNorm(hidden_size)
        self.ffn = RWKV7FeedForward(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, is_prefill: bool, v_first: torch.Tensor | None = None, att_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (x, v_first) where v_first is the value from layer 0."""
        if not is_prefill:
            context = get_context()
            slot_mapping_in = context.slot_mapping_in
            slot_mapping_out = context.slot_mapping_out

            h = F.layer_norm(x, (self.ln1.hidden_size,), self.ln1.gamma, self.ln1.beta, self.ln1.eps)
            h, v_first = self.att._forward_decode(h, positions, slot_mapping_in, slot_mapping_out, v_first)
            x.add_(h)

            h = F.layer_norm(x, (self.ln2.hidden_size,), self.ln2.gamma, self.ln2.beta, self.ln2.eps)
            h = self.ffn._forward_decode(h, slot_mapping_in, slot_mapping_out)
            x.add_(h)
            return x, v_first

        # Attention with residual - apply ln1 first, then pass LN-normalized xx to attention
        xx = self.ln1(x)
        if att_mask is not None:
            xx.masked_fill_(att_mask, 0)
        h, v_first = self.att(positions, xx, is_prefill, v_first, att_mask)
        x.add_(h)
        if att_mask is not None:
            x.masked_fill_(att_mask, 0)

        # FFN with residual - uses ln2-normalized x
        h = self.ln2(x)
        if att_mask is not None:
            h.masked_fill_(att_mask, 0)
        h = self.ffn(h, is_prefill)
        x.add_(h)
        if att_mask is not None:
            x.masked_fill_(att_mask, 0)

        return x, v_first


class RWKV7Model(nn.Module):
    """RWKV-7 model."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, num_layers: int, intermediate_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers

        self.emb = VocabParallelEmbedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([
            RWKV7Block(i, hidden_size, num_heads, head_dim, intermediate_size)
            for i in range(num_layers)
        ])
        self.ln_out = LayerNorm(hidden_size)

        # Will hold loaded state dict from pth
        self.z = {}
        self._decode_elapsed_cache: dict[int, torch.Tensor] = {}

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        context = get_context()
        is_prefill = context.is_prefill

        x = self.emb(input_ids)
        if not is_prefill:
            slot_mapping_in = context.slot_mapping_in
            slot_mapping_out = context.slot_mapping_out
            v_first = None
            use_scripted_contiguous_decode = (
                x.dim() == 2
                and x.size(0) > 1
                and torch.equal(slot_mapping_in, slot_mapping_out)
                and _is_contiguous_in_order(slot_mapping_in)
            )
            if use_scripted_contiguous_decode:
                slot_start = int(slot_mapping_in[0].item())
                slot_end = slot_start + slot_mapping_in.numel()
                batch_size = slot_mapping_in.numel()
                elapsed_t = self._decode_elapsed_cache.get(batch_size)
                if elapsed_t is None or elapsed_t.device != x.device:
                    elapsed_t = torch.zeros(batch_size, device=x.device, dtype=torch.int32)
                    self._decode_elapsed_cache[batch_size] = elapsed_t
                v_first_seq = None
                for block in self.blocks:
                    x, v_first_seq = _rwkv7_decode_block_batch_contiguous(
                        x,
                        block.att.att_tokenshift_cache[slot_start:slot_end],
                        block.att.state_cache[slot_start:slot_end],
                        block.ffn.ffn_tokenshift_cache[slot_start:slot_end],
                        elapsed_t,
                        v_first_seq,
                        block.layer_idx,
                        block.att.num_heads,
                        block.att.head_dim,
                        block.att.x_r,
                        block.att.x_w,
                        block.att.x_k,
                        block.att.x_v,
                        block.att.x_a,
                        block.att.x_g,
                        block.att.w0,
                        block.att.w1,
                        block.att.w2,
                        block.att.a0,
                        block.att.a1,
                        block.att.a2,
                        block.att.v0,
                        block.att.v1,
                        block.att.v2,
                        block.att.g1,
                        block.att.g2,
                        block.att.k_k,
                        block.att.k_a,
                        block.att.r_k,
                        block.att.receptance_weight,
                        block.att.key_weight,
                        block.att.value_weight,
                        block.att.output_weight,
                        block.att.ln_x_weight,
                        block.att.ln_x_bias,
                        block.ln1.gamma,
                        block.ln1.beta,
                        block.ln1.eps,
                        block.ln2.gamma,
                        block.ln2.beta,
                        block.ln2.eps,
                        block.ffn.x_k,
                        block.ffn.key_weight,
                        block.ffn.value_weight,
                    )
                v_first = None if v_first_seq is None else v_first_seq.squeeze(1)
            else:
                for block in self.blocks:
                    h = F.layer_norm(x, (block.ln1.hidden_size,), block.ln1.gamma, block.ln1.beta, block.ln1.eps)
                    h, v_first = block.att._forward_decode(h, positions, slot_mapping_in, slot_mapping_out, v_first)
                    x.add_(h)

                    h = F.layer_norm(x, (block.ln2.hidden_size,), block.ln2.gamma, block.ln2.beta, block.ln2.eps)
                    h = block.ffn._forward_decode(h, slot_mapping_in, slot_mapping_out)
                    x.add_(h)

            return F.layer_norm(x, (self.ln_out.hidden_size,), self.ln_out.gamma, self.ln_out.beta, self.ln_out.eps)

        att_mask = None
        if is_prefill and x.dim() == 3:
            _, T, _ = x.shape
            att_mask = (
                torch.arange(T, device=x.device, dtype=torch.int32).unsqueeze(0) <
                (T - context.context_lens).unsqueeze(1)
            ).unsqueeze(2)
        v_first = None

        for block in self.blocks:
            x, v_first = block(x, positions, is_prefill, v_first, att_mask)

        x = self.ln_out(x)
        return x

    def load_pth(self, pth_path: str):
        """Load weights from RWKV pth file."""
        z = torch.load(pth_path, map_location='cpu')

        # Process keys similar to Albatross
        keys = list(z.keys())
        max_layer = -1

        for k in keys:
            kk = k.split('.')
            # Transpose certain weight matrices
            if 'att.g1' in k or 'att.g2' in k or 'att.a1' in k or 'att.a2' in k or \
               'att.w1' in k or 'att.w2' in k or 'att.v1' in k or 'att.v2' in k or \
               'ffn.value.weight' in k:
                z[k] = z[k].t()

            # Convert to model dtype and move to current device
            dtype = torch.get_default_dtype()
            if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
                device = torch.device(f'cuda:{torch.cuda.current_device()}')
            else:
                device = torch.device('cpu')
            z[k] = z[k].squeeze().to(dtype=dtype, device=device)

            if k.endswith('att.r_k'):
                z[k] = z[k].flatten()

            z[k] = z[k].contiguous()

            if kk[0] == 'blocks':
                max_layer = max(max_layer, int(kk[1]))

        # Pre-process embedding with ln0
        z['emb.weight'] = F.layer_norm(
            z['emb.weight'],
            (self.hidden_size,),
            weight=z['blocks.0.ln0.weight'],
            bias=z['blocks.0.ln0.bias']
        )

        # Dummy entries for layer 0 (ignored in actual computation)
        z['blocks.0.att.v0'] = z['blocks.0.att.a0']
        z['blocks.0.att.v1'] = z['blocks.0.att.a1']
        z['blocks.0.att.v2'] = z['blocks.0.att.a2']

        self.z = z

        # Now load into module parameters
        self._load_weights()

    def _load_weights(self):
        """Load processed weights into module parameters."""
        z = self.z

        # Embedding
        if self.emb.tp_size == 1:
            self.emb.weight.data = z['emb.weight']
        else:
            self.emb.weight.data.copy_(z['emb.weight'])

        for i, block in enumerate(self.blocks):
            bbb = f'blocks.{i}.'
            att = f'blocks.{i}.att.'
            ffn = f'blocks.{i}.ffn.'

            # Layer norms
            block.ln1.gamma.data = z[bbb + 'ln1.weight']
            block.ln1.beta.data = z[bbb + 'ln1.bias']
            block.ln2.gamma.data = z[bbb + 'ln2.weight']
            block.ln2.beta.data = z[bbb + 'ln2.bias']

            # FFN weights - register as buffers (already on correct device from load_pth)
            block.ffn.register_buffer('key_weight', z[ffn + 'key.weight'])
            block.ffn.register_buffer('value_weight', z[ffn + 'value.weight'])
            block.ffn.register_buffer('x_k', z[ffn + 'x_k'])

            # Attention parameters - register as buffers (already on correct device from load_pth)
            block.att.register_buffer('x_r', z[att + 'x_r'])
            block.att.register_buffer('x_w', z[att + 'x_w'])
            block.att.register_buffer('x_k', z[att + 'x_k'])
            block.att.register_buffer('x_v', z[att + 'x_v'])
            block.att.register_buffer('x_a', z[att + 'x_a'])
            block.att.register_buffer('x_g', z[att + 'x_g'])
            block.att.register_buffer('w0', z[att + 'w0'])
            block.att.register_buffer('w1', z[att + 'w1'])
            block.att.register_buffer('w2', z[att + 'w2'])
            block.att.register_buffer('a0', z[att + 'a0'])
            block.att.register_buffer('a1', z[att + 'a1'])
            block.att.register_buffer('a2', z[att + 'a2'])
            block.att.register_buffer('v0', z[att + 'v0'])
            block.att.register_buffer('v1', z[att + 'v1'])
            block.att.register_buffer('v2', z[att + 'v2'])
            block.att.register_buffer('g1', z[att + 'g1'])
            block.att.register_buffer('g2', z[att + 'g2'])
            block.att.register_buffer('k_k', z[att + 'k_k'].squeeze())  # [hidden_size]
            block.att.register_buffer('k_a', z[att + 'k_a'].squeeze())  # [hidden_size]
            block.att.register_buffer('r_k', z[att + 'r_k'].flatten())  # [hidden_size]
            block.att.register_buffer('receptance_weight', z[att + 'receptance.weight'])
            block.att.register_buffer('key_weight', z[att + 'key.weight'])
            block.att.register_buffer('value_weight', z[att + 'value.weight'])
            block.att.register_buffer('output_weight', z[att + 'output.weight'])
            block.att.register_buffer('ln_x_weight', z[att + 'ln_x.weight'])
            block.att.register_buffer('ln_x_bias', z[att + 'ln_x.bias'])

        # Output layer norm
        self.ln_out.gamma.data = z['ln_out.weight']
        self.ln_out.beta.data = z['ln_out.bias']


class RWKV7ForCausalLM(nn.Module):
    """RWKV-7 for causal language modeling."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', 64)
        num_heads = getattr(config, 'num_heads', hidden_size // head_dim)
        num_layers = config.num_hidden_layers
        intermediate_size = config.intermediate_size
        vocab_size = config.vocab_size

        self.model = RWKV7Model(hidden_size, num_heads, head_dim, num_layers, intermediate_size, vocab_size)
        self.lm_head = ParallelLMHead(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def forward_logits(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        if self.lm_head.tp_size == 1 and hidden_states.dim() == 2:
            return F.linear(hidden_states, self.lm_head.weight)
        return self.lm_head(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_pth(self, pth_path: str):
        """Load model weights from pth file."""
        self.model.load_pth(pth_path)
        # Load head weight
        if self.lm_head.tp_size == 1:
            self.lm_head.weight.data = self.model.z['head.weight']
        else:
            self.lm_head.weight.data.copy_(self.model.z['head.weight'])
