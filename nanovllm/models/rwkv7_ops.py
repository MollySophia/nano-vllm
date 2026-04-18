import math
import os
from typing import Optional

import torch
import torch.nn.functional as F

from nanovllm.layers.linear import MatmulLinear, MarlinInt8Linear
from nanovllm.ops.rwkv7_cuda import (
    ensure_loaded as ensure_rwkv7_cuda_loaded,
    cmix_one as rwkv7_cmix_one_cuda,
    rwkv7_one_op as rwkv7_one_cuda,
    wkv7_one_batch as wkv7_one_batch_cuda,
    wkv7_seq as wkv7_seq_cuda,
    wkv7_seq_batch as wkv7_seq_batch_cuda,
)
from nanovllm.utils.context import get_context


# Constants for w transformation (from Albatross CUDA kernel)
_NEXP_HALF_LOG2_E = -0.8750387749145276
_NLOG2_E = -1.4426950408889634
_LN2 = math.log(2.0)
_NEXP_HALF = _NEXP_HALF_LOG2_E * _LN2
_NLOG2E_LN2 = _NLOG2_E * _LN2
_TWO_TO_NEG_41 = 4.547473508864641e-13
_RO1_I32 = -1640531527
_MAX_NONCONTIGUOUS_STATE_GATHER_ROWS = max(
    1,
    int(os.getenv("NANOVLLM_MAX_NONCONTIGUOUS_STATE_GATHER_ROWS", "4")),
)


def _maybe_compile_rwkv_helper(fn):
    if not hasattr(torch, "jit"):
        return fn
    try:
        return torch.jit.script(fn)
    except Exception:
        return fn


def _matmul_linear_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
):
    if x.dim() == 1 and weight.dim() == 2:
        return F.linear(x, weight.t(), bias)
    y = torch.matmul(x, weight)
    if bias is not None:
        y = y + bias
    return y


_matmul_linear = _maybe_compile_rwkv_helper(_matmul_linear_impl)


def _linear_dispatch(
    x: torch.Tensor,
    proj_or_weight,
    bias: torch.Tensor | None = None,
):
    if isinstance(proj_or_weight, torch.Tensor):
        return _matmul_linear(x, proj_or_weight, bias=bias)
    if isinstance(proj_or_weight, MarlinInt8Linear):
        if bias is not None:
            raise RuntimeError("MarlinInt8Linear does not support runtime bias override.")
        return proj_or_weight(x)
    if isinstance(proj_or_weight, MatmulLinear):
        proj_bias = bias if bias is not None else proj_or_weight.bias
        if proj_or_weight.weight_layout == "out_in":
            return F.linear(x, proj_or_weight.weight, proj_bias)
        return _matmul_linear(x, proj_or_weight.weight, bias=proj_bias)
    proj_bias = bias if bias is not None else proj_or_weight.bias
    return _matmul_linear(x, proj_or_weight.weight, bias=proj_bias)


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
    x_prev.copy_(x)
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
    # Match the Albatross single-token path exactly to minimize fp16 tie flips.
    kk = F.normalize((k * k_k).view(num_heads, head_dim), dim=-1, p=2.0).view_as(k)
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
    y = F.group_norm(
        wkv_out.view(1, num_heads * head_dim),
        num_groups=num_heads,
        weight=ln_x_weight,
        bias=ln_x_bias,
        eps=64e-5,
    ).view(num_heads * head_dim)
    y = y + (
        ((r * k * r_k).view(num_heads, head_dim).sum(dim=-1, keepdim=True) * v.view(num_heads, head_dim)).view(num_heads * head_dim)
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
    k = torch.relu(_matmul_linear(k, key_weight)) ** 2
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
    w1_proj,
    w2_proj,
    a0: torch.Tensor,
    a1_proj,
    a2_proj,
    v0: torch.Tensor,
    v1_proj,
    v2_proj,
    g1_proj,
    g2_proj,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    receptance_proj,
    key_proj,
    value_proj,
):
    bsz, seqlen, c = x.shape
    xx = x_prev - x
    xr = torch.addcmul(x, xx, x_r.view(1, 1, c))
    xw = torch.addcmul(x, xx, x_w.view(1, 1, c))
    xk = torch.addcmul(x, xx, x_k.view(1, 1, c))
    xv = torch.addcmul(x, xx, x_v.view(1, 1, c))
    xa = torch.addcmul(x, xx, x_a.view(1, 1, c))
    xg = torch.addcmul(x, xx, x_g.view(1, 1, c))
    r = _linear_dispatch(xr, receptance_proj)
    w = _linear_dispatch(torch.tanh(_linear_dispatch(xw, w1_proj)), w2_proj) + w0
    k = _linear_dispatch(xk, key_proj)
    v = _linear_dispatch(xv, value_proj)
    a = torch.sigmoid(_linear_dispatch(_linear_dispatch(xa, a1_proj), a2_proj) + a0)

    kk = F.normalize((k * k_k.view(1, 1, c)).view(bsz, seqlen, num_heads, head_dim), dim=-1, p=2.0).view(bsz, seqlen, c)
    k = k * (1 + (a - 1) * k_a.view(1, 1, c))
    kka = kk * a
    g = _linear_dispatch(torch.sigmoid(_linear_dispatch(xg, g1_proj)), g2_proj)

    if layer_idx == 0:
        v_first_out = v
    else:
        assert v_first is not None
        v_mix = torch.sigmoid(_linear_dispatch(_linear_dispatch(xv.view(bsz, seqlen, -1), v1_proj), v2_proj, bias=v0))
        v = v + (v_first - v) * v_mix
        v_first_out = v_first

    return r, w, k, v, kk, kka, g, v_first_out


_rwkv7_tmix_one = _maybe_compile_rwkv_helper(_rwkv7_tmix_one_impl)
_rwkv7_tmix_one_post = _maybe_compile_rwkv_helper(_rwkv7_tmix_one_post_impl)
_rwkv7_ffn_decode = _maybe_compile_rwkv_helper(_rwkv7_ffn_decode_impl)
_rwkv7_tmix_seq_batch = _rwkv7_tmix_seq_batch_impl


def _rwkv7_decode_block_batch_contiguous(
    x: torch.Tensor,
    att_tokenshift_cache_in: torch.Tensor,
    att_tokenshift_cache_out: torch.Tensor,
    state_cache_in: torch.Tensor,
    state_cache_out: torch.Tensor,
    ffn_tokenshift_cache_in: torch.Tensor,
    ffn_tokenshift_cache_out: torch.Tensor,
    positions: torch.Tensor,
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
    w1_proj,
    w2_proj,
    a0: torch.Tensor,
    a1_proj,
    a2_proj,
    v0: torch.Tensor,
    v1_proj,
    v2_proj,
    g1_proj,
    g2_proj,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    r_k: torch.Tensor,
    receptance_proj,
    key_proj,
    value_proj,
    output_proj,
    ln_x_weight: torch.Tensor,
    ln_x_bias: torch.Tensor,
    ln1_gamma: torch.Tensor,
    ln1_beta: torch.Tensor,
    ln1_eps: float,
    ln2_gamma: torch.Tensor,
    ln2_beta: torch.Tensor,
    ln2_eps: float,
    ffn_x_k: torch.Tensor,
    ffn_key_proj,
    ffn_value_proj,
    decode_tokenshift_scratch: torch.Tensor,
):
    bsz, c = x.shape
    att_inplace = att_tokenshift_cache_in.data_ptr() == att_tokenshift_cache_out.data_ptr()
    ffn_inplace = ffn_tokenshift_cache_in.data_ptr() == ffn_tokenshift_cache_out.data_ptr()
    h = F.layer_norm(x, (c,), ln1_gamma, ln1_beta, ln1_eps)
    if att_inplace:
        xx = att_tokenshift_cache_in.to(dtype=h.dtype) - h
        att_tokenshift_cache_out.copy_(h)
    else:
        h_cache = h.to(att_tokenshift_cache_out.dtype)
        x_prev = decode_tokenshift_scratch[:bsz]
        x_prev.copy_(att_tokenshift_cache_in)
        att_tokenshift_cache_out.copy_(h_cache)
        xx = x_prev - h
    xr = torch.addcmul(h, xx, x_r)
    xw = torch.addcmul(h, xx, x_w)
    xk = torch.addcmul(h, xx, x_k)
    xv = torch.addcmul(h, xx, x_v)
    xa = torch.addcmul(h, xx, x_a)
    xg = torch.addcmul(h, xx, x_g)

    r = _linear_dispatch(xr, receptance_proj)
    w = _linear_dispatch(torch.tanh(_linear_dispatch(xw, w1_proj)), w2_proj, bias=w0)
    k = _linear_dispatch(xk, key_proj)
    v = _linear_dispatch(xv, value_proj)
    a = torch.sigmoid(_linear_dispatch(_linear_dispatch(xa, a1_proj), a2_proj, bias=a0))
    g = _linear_dispatch(torch.sigmoid(_linear_dispatch(xg, g1_proj)), g2_proj)

    kk = F.normalize((k * k_k.view(1, 1, c)).view(bsz, num_heads, head_dim), dim=-1, p=2.0).view(bsz, c)
    k = k * (1 + (a - 1) * k_a.view(1, c))
    kka = kk * a

    if layer_idx == 0:
        v_first = v
    else:
        assert v_first is not None
        v = v + (v_first - v) * torch.sigmoid(_linear_dispatch(_linear_dispatch(xv, v1_proj), v2_proj, bias=v0))

    y = wkv7_one_batch_cuda(
        state_cache_in,
        state_cache_out,
        r,
        w,
        k,
        v,
        -kk,
        kka,
        positions,
    )
    y = F.group_norm(y.view(bsz, c), num_groups=num_heads, weight=ln_x_weight, bias=ln_x_bias, eps=64e-5)
    y = y + (
        ((r * k * r_k.view(1, c)).view(bsz, num_heads, head_dim).sum(dim=-1, keepdim=True) * v.view(bsz, num_heads, head_dim)).view(bsz, c)
    )
    y = _linear_dispatch(y * g, output_proj)
    x.add_(y)

    h2 = F.layer_norm(x, (c,), ln2_gamma, ln2_beta, ln2_eps)
    if ffn_inplace:
        xx = ffn_tokenshift_cache_in.to(dtype=h2.dtype) - h2
        ffn_tokenshift_cache_out.copy_(h2)
    else:
        h2_cache = h2.to(ffn_tokenshift_cache_out.dtype)
        x_prev_ffn = decode_tokenshift_scratch[:bsz]
        x_prev_ffn.copy_(ffn_tokenshift_cache_in)
        ffn_tokenshift_cache_out.copy_(h2_cache)
        xx = x_prev_ffn - h2
    k_ffn = torch.addcmul(h2, xx, ffn_x_k)
    k_ffn = torch.relu(_linear_dispatch(k_ffn, ffn_key_proj)) ** 2
    x.add_(_linear_dispatch(k_ffn, ffn_value_proj))
    return x, v_first


def _rwkv7_decode_block_one_contiguous(
    x: torch.Tensor,
    att_tokenshift_cache_in: torch.Tensor,
    att_tokenshift_cache_out: torch.Tensor,
    state_cache_in: torch.Tensor,
    state_cache_out: torch.Tensor,
    ffn_tokenshift_cache_in: torch.Tensor,
    ffn_tokenshift_cache_out: torch.Tensor,
    positions: torch.Tensor,
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
    w1_proj,
    w2_proj,
    a0: torch.Tensor,
    a1_proj,
    a2_proj,
    v0: torch.Tensor,
    v1_proj,
    v2_proj,
    g1_proj,
    g2_proj,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    r_k: torch.Tensor,
    receptance_proj,
    key_proj,
    value_proj,
    output_proj,
    ln_x_weight: torch.Tensor,
    ln_x_bias: torch.Tensor,
    ln1_gamma: torch.Tensor,
    ln1_beta: torch.Tensor,
    ln1_eps: float,
    ln2_gamma: torch.Tensor,
    ln2_beta: torch.Tensor,
    ln2_eps: float,
    ffn_x_k: torch.Tensor,
    ffn_key_proj,
    ffn_value_proj,
    decode_tokenshift_scratch: torch.Tensor,
):
    use_fp16_tmix_helper = (
        isinstance(receptance_proj, MatmulLinear)
        and isinstance(w1_proj, MatmulLinear)
        and isinstance(w2_proj, MatmulLinear)
        and isinstance(key_proj, MatmulLinear)
        and isinstance(value_proj, MatmulLinear)
        and isinstance(a1_proj, MatmulLinear)
        and isinstance(a2_proj, MatmulLinear)
        and isinstance(v1_proj, MatmulLinear)
        and isinstance(v2_proj, MatmulLinear)
        and isinstance(g1_proj, MatmulLinear)
        and isinstance(g2_proj, MatmulLinear)
    )
    if use_fp16_tmix_helper:
        x0 = x[0]
        h = F.layer_norm(x0, (x0.shape[-1],), ln1_gamma, ln1_beta, ln1_eps)
        if att_tokenshift_cache_out.data_ptr() != att_tokenshift_cache_in.data_ptr():
            att_tokenshift_cache_out[0].copy_(att_tokenshift_cache_in[0])
        x_prev = att_tokenshift_cache_out[0]
        xx = x_prev - h
        x_prev.copy_(h)
        xr = torch.addcmul(h, xx, x_r)
        xw = torch.addcmul(h, xx, x_w)
        xk = torch.addcmul(h, xx, x_k)
        xv = torch.addcmul(h, xx, x_v)
        xa = torch.addcmul(h, xx, x_a)
        xg = torch.addcmul(h, xx, x_g)

        r = F.linear(xr, receptance_proj.weight)
        w = F.linear(torch.tanh(F.linear(xw, w1_proj.weight)), w2_proj.weight, bias=w0)
        k = F.linear(xk, key_proj.weight)
        v = F.linear(xv, value_proj.weight)
        a = torch.sigmoid(F.linear(F.linear(xa, a1_proj.weight), a2_proj.weight, bias=a0))
        g = F.linear(torch.sigmoid(F.linear(xg, g1_proj.weight)), g2_proj.weight)
        kk = F.normalize((k * k_k).view(num_heads, head_dim), dim=-1, p=2.0).view_as(k)
        k = k * (1 + (a - 1) * k_a)
        kka = kk * a
        if layer_idx == 0:
            v_first = v
        else:
            assert v_first is not None
            v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, v1_proj.weight), v2_proj.weight, bias=v0))
    else:
        x0 = x[0]
        h = F.layer_norm(x0, (x0.shape[-1],), ln1_gamma, ln1_beta, ln1_eps)
        h_cache = h.to(att_tokenshift_cache_out.dtype)
        x_prev = decode_tokenshift_scratch[0]
        x_prev.copy_(att_tokenshift_cache_in[0])
        xx = x_prev - h
        att_tokenshift_cache_out[0].copy_(h_cache)
        xr = torch.addcmul(h, xx, x_r)
        xw = torch.addcmul(h, xx, x_w)
        xk = torch.addcmul(h, xx, x_k)
        xv = torch.addcmul(h, xx, x_v)
        xa = torch.addcmul(h, xx, x_a)
        xg = torch.addcmul(h, xx, x_g)
        r = _linear_dispatch(xr, receptance_proj)
        w = _linear_dispatch(torch.tanh(_linear_dispatch(xw, w1_proj)), w2_proj, bias=w0)
        k = _linear_dispatch(xk, key_proj)
        v = _linear_dispatch(xv, value_proj)
        a = torch.sigmoid(_linear_dispatch(_linear_dispatch(xa, a1_proj), a2_proj, bias=a0))
        g = _linear_dispatch(torch.sigmoid(_linear_dispatch(xg, g1_proj)), g2_proj)
        kk = F.normalize((k * k_k).view(1, num_heads, head_dim), dim=-1, p=2.0).view_as(k)
        k = k * (1 + (a - 1) * k_a)
        kka = kk * a
        if layer_idx == 0:
            v_first = v
        else:
            assert v_first is not None
            v = v + (v_first - v) * torch.sigmoid(_linear_dispatch(_linear_dispatch(xv, v1_proj), v2_proj, bias=v0))
    y = rwkv7_one_cuda(
        state_cache_in[0],
        state_cache_out[0],
        r,
        w,
        k,
        v,
        -kk,
        kka,
        positions[0:1],
    )
    if use_fp16_tmix_helper and isinstance(output_proj, MatmulLinear):
        y = F.group_norm(
            y.view(1, num_heads * head_dim),
            num_groups=num_heads,
            weight=ln_x_weight,
            bias=ln_x_bias,
            eps=64e-5,
        ).view(num_heads * head_dim)
        y = y + (
            ((r * k * r_k).view(num_heads, head_dim).sum(dim=-1, keepdim=True) * v.view(num_heads, head_dim)).view(num_heads * head_dim)
        )
        y = F.linear(y * g, output_proj.weight)
    else:
        y = F.group_norm(y.view(1, -1), num_groups=num_heads, weight=ln_x_weight, bias=ln_x_bias, eps=64e-5).view(-1)
        y = y + ((r * k * r_k).view(1, num_heads, head_dim).sum(dim=-1, keepdim=True) * v.view(1, num_heads, head_dim)).view_as(r)
        y = _linear_dispatch(y * g, output_proj)
    x0.add_(y)

    h2 = F.layer_norm(x0, (x0.shape[-1],), ln2_gamma, ln2_beta, ln2_eps)
    if (
        isinstance(ffn_key_proj, MatmulLinear)
        and isinstance(ffn_value_proj, MatmulLinear)
        and ffn_key_proj.weight_layout == "out_in"
        and ffn_value_proj.weight_layout == "in_out"
    ):
        if ffn_tokenshift_cache_out.data_ptr() != ffn_tokenshift_cache_in.data_ptr():
            ffn_tokenshift_cache_out[0].copy_(ffn_tokenshift_cache_in[0])
        x_prev_ffn = ffn_tokenshift_cache_out[0]
        x0.add_(rwkv7_cmix_one_cuda(h2, x_prev_ffn, ffn_x_k, ffn_key_proj.weight, ffn_value_proj.weight))
    else:
        h2_cache = h2.to(ffn_tokenshift_cache_out.dtype)
        x_prev_ffn = decode_tokenshift_scratch[0]
        x_prev_ffn.copy_(ffn_tokenshift_cache_in[0])
        ffn_tokenshift_cache_out[0].copy_(h2_cache)
        xx = x_prev_ffn - h2
        if isinstance(ffn_key_proj, MatmulLinear) and isinstance(ffn_value_proj, MatmulLinear):
            x0.add_(_rwkv7_ffn_decode(h2, xx, ffn_x_k, ffn_key_proj.weight, ffn_value_proj.weight))
        else:
            k_ffn = torch.addcmul(h2, xx, ffn_x_k)
            k_ffn = torch.relu(_linear_dispatch(k_ffn, ffn_key_proj)) ** 2
            x0.add_(_linear_dispatch(k_ffn, ffn_value_proj))
    return x, v_first


def wkv7_one_step(
    state: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    position: int | torch.Tensor,
) -> torch.Tensor:
    """
    Single step WKV-7 computation with stable decay.
    """
    decay = torch.exp(_NEXP_HALF / (1.0 + torch.exp(_NLOG2E_LN2 * w)))

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

    sa = torch.einsum("hmn,hn->hm", state, a)
    state.mul_(decay.unsqueeze(-2))
    state.add_(v.unsqueeze(-1) * k.unsqueeze(-2))
    state.add_(sa.unsqueeze(-1) * b.unsqueeze(-2))
    y = torch.einsum("hmn,hn->hm", state, r)
    return y.to(out_dtype)


def wkv7_sequence(
    state: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    kka: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """
    Sequence WKV-7 computation (prefill).
    """
    t, _, _ = r.shape
    outputs = []

    for idx in range(t):
        y = wkv7_one_step(state, r[idx], w[idx], k[idx], v[idx], -kk[idx], kka[idx], positions[idx])
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


def _state_rows_for_sorted_run(state_cache: torch.Tensor, sorted_slots: torch.Tensor, start: int, end: int) -> torch.Tensor:
    run_slots = sorted_slots[start:end]
    if _is_contiguous_in_order(run_slots):
        slot_start = int(run_slots[0].item())
        return state_cache[slot_start:slot_start + (end - start)]
    return state_cache[run_slots].contiguous()


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
    context = get_context()
    if getattr(context, "force_regular_decode", False):
        return wkv7_one_batch_cuda(
            state_cache[slot_mapping],
            state_cache[slot_mapping],
            r,
            w,
            k,
            v,
            kk,
            kka,
            positions,
        )
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
    context = get_context()
    if getattr(context, "force_regular_decode", False):
        return wkv7_one_batch_cuda(
            state_cache[slot_mapping_in],
            state_cache[slot_mapping_out],
            r,
            w,
            k,
            v,
            kk,
            kka,
            positions,
        )
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
    r_sorted = r[order]
    w_sorted = w[order]
    k_sorted = k[order]
    v_sorted = v[order]
    kk_sorted = kk[order]
    kka_sorted = kka[order]
    positions_sorted = positions[order]
    y_sorted = torch.empty_like(r_sorted)

    for start, end, slot_start, slot_end in runs:
        run_slots_in = slot_in_sorted[start:end]
        if _is_contiguous_in_order(run_slots_in):
            y_sorted[start:end] = wkv7_one_batch_cuda(
                _state_rows_for_sorted_run(state_cache, slot_in_sorted, start, end),
                state_cache[slot_start:slot_end],
                r_sorted[start:end],
                w_sorted[start:end],
                k_sorted[start:end],
                v_sorted[start:end],
                kk_sorted[start:end],
                kka_sorted[start:end],
                positions_sorted[start:end],
            )
            continue
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + _MAX_NONCONTIGUOUS_STATE_GATHER_ROWS, end)
            out_slot_start = slot_start + (chunk_start - start)
            out_slot_end = out_slot_start + (chunk_end - chunk_start)
            y_sorted[chunk_start:chunk_end] = wkv7_one_batch_cuda(
                _state_rows_for_sorted_run(state_cache, slot_in_sorted, chunk_start, chunk_end),
                state_cache[out_slot_start:out_slot_end],
                r_sorted[chunk_start:chunk_end],
                w_sorted[chunk_start:chunk_end],
                k_sorted[chunk_start:chunk_end],
                v_sorted[chunk_start:chunk_end],
                kk_sorted[chunk_start:chunk_end],
                kka_sorted[chunk_start:chunk_end],
                positions_sorted[chunk_start:chunk_end],
            )
            chunk_start = chunk_end

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
    r_sorted = r[order]
    w_sorted = w[order]
    k_sorted = k[order]
    v_sorted = v[order]
    kk_sorted = kk[order]
    kka_sorted = kka[order]
    elapsed_sorted = elapsed_t[order]
    y_sorted = torch.empty_like(r_sorted)

    for start, end, slot_start, slot_end in runs:
        run_slots_in = slot_in_sorted[start:end]
        if _is_contiguous_in_order(run_slots_in):
            y_sorted[start:end] = wkv7_seq_batch_cuda(
                _state_rows_for_sorted_run(state_cache, slot_in_sorted, start, end),
                state_cache[slot_start:slot_end],
                r_sorted[start:end],
                w_sorted[start:end],
                k_sorted[start:end],
                v_sorted[start:end],
                kk_sorted[start:end],
                kka_sorted[start:end],
                elapsed_sorted[start:end],
            )
            continue
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + _MAX_NONCONTIGUOUS_STATE_GATHER_ROWS, end)
            out_slot_start = slot_start + (chunk_start - start)
            out_slot_end = out_slot_start + (chunk_end - chunk_start)
            y_sorted[chunk_start:chunk_end] = wkv7_seq_batch_cuda(
                _state_rows_for_sorted_run(state_cache, slot_in_sorted, chunk_start, chunk_end),
                state_cache[out_slot_start:out_slot_end],
                r_sorted[chunk_start:chunk_end],
                w_sorted[chunk_start:chunk_end],
                k_sorted[chunk_start:chunk_end],
                v_sorted[chunk_start:chunk_end],
                kk_sorted[chunk_start:chunk_end],
                kka_sorted[chunk_start:chunk_end],
                elapsed_sorted[chunk_start:chunk_end],
            )
            chunk_start = chunk_end

    return y_sorted[inverse]
