import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovllm.layers.layernorm import LayerNorm
from nanovllm.layers.linear import MatmulLinear
from nanovllm.models.rwkv7_ops import (
    _linear_dispatch,
    _rwkv7_ffn_decode,
    _rwkv7_tmix_one,
    _rwkv7_tmix_one_post,
    _rwkv7_tmix_seq_batch,
    _wkv7_one_batch_inplace_by_slot_runs,
    _wkv7_one_batch_out_by_slot_runs,
    _wkv7_seq_batch_inplace_by_slot_runs,
    _wkv7_seq_batch_out_by_slot_runs,
    ensure_rwkv7_cuda_loaded,
    rwkv7_cmix_one_cuda,
    rwkv7_one_cuda,
    wkv7_one_step,
    wkv7_sequence,
)
from nanovllm.utils.context import get_context


class RWKV7Attention(nn.Module):
    """RWKV-7 "Goose" linear attention implementation."""

    def __init__(self, layer_idx: int, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.att_tokenshift_cache = None
        self.state_cache = None

        self._cuda_kernel_ready = False
        self._cuda_kernel_attempted = False
        self.receptance_proj = MatmulLinear(hidden_size, hidden_size, weight_layout="out_in")
        self.key_proj = MatmulLinear(hidden_size, hidden_size, weight_layout="out_in")
        self.value_proj = MatmulLinear(hidden_size, hidden_size, weight_layout="out_in")
        self.output_proj = MatmulLinear(hidden_size, hidden_size, weight_layout="out_in")
        self.w1_proj = MatmulLinear(hidden_size, 128, weight_layout="out_in")
        self.w2_proj = MatmulLinear(128, hidden_size, weight_layout="out_in")
        self.a1_proj = MatmulLinear(hidden_size, 128, weight_layout="out_in")
        self.a2_proj = MatmulLinear(128, hidden_size, weight_layout="out_in")
        self.v1_proj = MatmulLinear(hidden_size, 96, weight_layout="out_in")
        self.v2_proj = MatmulLinear(96, hidden_size, weight_layout="out_in")
        self.g1_proj = MatmulLinear(hidden_size, 480, weight_layout="out_in")
        self.g2_proj = MatmulLinear(480, hidden_size, weight_layout="out_in")

    @property
    def receptance_weight(self):
        return self.receptance_proj.weight

    @property
    def key_weight(self):
        return self.key_proj.weight

    @property
    def value_weight(self):
        return self.value_proj.weight

    @property
    def output_weight(self):
        return self.output_proj.weight

    @property
    def w1(self):
        return self.w1_proj.weight

    @property
    def w2(self):
        return self.w2_proj.weight

    @property
    def a1(self):
        return self.a1_proj.weight

    @property
    def a2(self):
        return self.a2_proj.weight

    @property
    def v1(self):
        return self.v1_proj.weight

    @property
    def v2(self):
        return self.v2_proj.weight

    @property
    def g1(self):
        return self.g1_proj.weight

    @property
    def g2(self):
        return self.g2_proj.weight

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
        context = get_context()
        slot_mapping_in = context.slot_mapping_in
        slot_mapping_out = context.slot_mapping_out

        if is_prefill:
            return self._forward_prefill(x, positions, slot_mapping_in, slot_mapping_out, v_first, att_mask)
        return self._forward_decode(x, positions, slot_mapping_in, slot_mapping_out, v_first)

    def _forward_prefill(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping_in: torch.Tensor,
        slot_mapping_out: torch.Tensor,
        v_first: torch.Tensor | None = None,
        att_mask: torch.Tensor | None = None,
    ):
        if x.dim() != 3:
            raise NotImplementedError("RWKV state-cache path only supports batched prefill.")
        if att_mask is None:
            return self._forward_prefill_batch_same_length(x, slot_mapping_in, slot_mapping_out, v_first)
        return self._forward_prefill_batch_right(x, slot_mapping_in, slot_mapping_out, v_first, att_mask)

    def _forward_prefill_batch_right(
        self,
        x: torch.Tensor,
        slot_mapping_in: torch.Tensor,
        slot_mapping_out: torch.Tensor,
        v_first: torch.Tensor | None = None,
        att_mask: torch.Tensor | None = None,
    ):
        bsz, seqlen, hidden = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim
        context = get_context()
        context_lens = context.context_lens
        if att_mask is None:
            att_mask = (
                torch.arange(seqlen, device=x.device, dtype=torch.int32).unsqueeze(0) <
                (seqlen - context_lens).unsqueeze(1)
            ).unsqueeze(2)

        att_cache_in = self.att_tokenshift_cache[slot_mapping_in].to(x.dtype)
        x_prev = torch.cat((att_cache_in.unsqueeze(1), x[:, :-1, :]), dim=1)
        starts = (seqlen - context_lens).to(torch.long)
        x_prev[torch.arange(bsz, device=x.device), starts] = att_cache_in
        xx = x_prev - x
        self.att_tokenshift_cache[slot_mapping_out] = x[:, -1, :].to(self.att_tokenshift_cache.dtype)
        xr = torch.addcmul(x, xx, self.x_r.view(1, 1, hidden))
        xw = torch.addcmul(x, xx, self.x_w.view(1, 1, hidden))
        xk = torch.addcmul(x, xx, self.x_k.view(1, 1, hidden))
        xv = torch.addcmul(x, xx, self.x_v.view(1, 1, hidden))
        xa = torch.addcmul(x, xx, self.x_a.view(1, 1, hidden))
        r = _linear_dispatch(xr, self.receptance_proj)
        w = _linear_dispatch(
            torch.tanh(_linear_dispatch(xw, self.w1_proj)),
            self.w2_proj,
            bias=self.w0,
        )
        k = _linear_dispatch(xk, self.key_proj)
        v = _linear_dispatch(xv, self.value_proj)
        a = torch.sigmoid(
            _linear_dispatch(
                _linear_dispatch(xa, self.a1_proj),
                self.a2_proj,
                bias=self.a0,
            )
        )

        k_k = self.k_k.view(1, 1, hidden)
        k_a = self.k_a.view(1, 1, hidden)
        kk = F.normalize((k * k_k).view(bsz, seqlen, num_heads, head_dim), dim=-1, p=2.0).view(bsz, seqlen, hidden)
        kk.masked_fill_(att_mask, 0)
        k = k * (1 + (a - 1) * k_a)
        kka = kk * a

        if self.layer_idx == 0:
            v_first = v
        else:
            assert v_first is not None
            v_mix = torch.sigmoid(
                _linear_dispatch(
                    _linear_dispatch(xv.view(bsz, seqlen, -1), self.v1_proj),
                    self.v2_proj,
                    bias=self.v0,
                )
            )
            v = v + (v_first - v) * v_mix

        self._maybe_init_cuda_kernel()
        if self._cuda_kernel_ready and x.is_cuda and x.dtype == torch.float16:
            elapsed_t = context_lens - seqlen
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
            r_hn = r.view(bsz, seqlen, num_heads, head_dim)
            w_hn = w.view(bsz, seqlen, num_heads, head_dim)
            k_hn = k.view(bsz, seqlen, num_heads, head_dim)
            v_hn = v.view(bsz, seqlen, num_heads, head_dim)
            kk_hn = kk.view(bsz, seqlen, num_heads, head_dim)
            kka_hn = kka.view(bsz, seqlen, num_heads, head_dim)
            y = torch.zeros_like(r_hn)
            for idx in range(bsz):
                seq_len_i = int(context_lens[idx].item())
                start = seqlen - seq_len_i
                slot = slot_mapping_in[idx].item()
                state = self.state_cache[slot]
                y_i = wkv7_sequence(
                    state,
                    r_hn[idx, start:],
                    w_hn[idx, start:],
                    k_hn[idx, start:],
                    v_hn[idx, start:],
                    -kk_hn[idx, start:],
                    kka_hn[idx, start:],
                    torch.arange(seq_len_i, device=x.device, dtype=torch.int64),
                )
                y[idx, start:] = y_i
            y = y.view(bsz, seqlen, hidden)

        y = F.group_norm(y.view(bsz * seqlen, hidden), num_groups=num_heads, weight=self.ln_x_weight, bias=self.ln_x_bias, eps=64e-5).view(bsz, seqlen, hidden)
        y = y + (
            (
                (r * k * self.r_k.view(1, 1, hidden)).view(bsz, seqlen, num_heads, head_dim).sum(dim=-1, keepdim=True)
                * v.view(bsz, seqlen, num_heads, head_dim)
            ).view(bsz, seqlen, hidden)
        )
        g = _linear_dispatch(
            torch.sigmoid(_linear_dispatch(torch.addcmul(x, xx, self.x_g), self.g1_proj)),
            self.g2_proj,
        )
        y = _linear_dispatch(y * g, self.output_proj)
        return y, v_first

    def _forward_prefill_batch_same_length(
        self,
        x: torch.Tensor,
        slot_mapping_in: torch.Tensor,
        slot_mapping_out: torch.Tensor,
        v_first: torch.Tensor | None = None,
    ):
        bsz, seqlen, hidden = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim

        x_prev = torch.cat((self.att_tokenshift_cache[slot_mapping_in].to(x.dtype).unsqueeze(1), x[:, :-1, :]), dim=1)
        self.att_tokenshift_cache[slot_mapping_out] = x[:, -1, :].to(self.att_tokenshift_cache.dtype)
        r, w, k, v, kk, kka, g, v_first = _rwkv7_tmix_seq_batch(
            self.layer_idx,
            num_heads,
            head_dim,
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
            self.w1_proj,
            self.w2_proj,
            self.a0,
            self.a1_proj,
            self.a2_proj,
            self.v0,
            self.v1_proj,
            self.v2_proj,
            self.g1_proj,
            self.g2_proj,
            self.k_k,
            self.k_a,
            self.receptance_proj,
            self.key_proj,
            self.value_proj,
        )

        self._maybe_init_cuda_kernel()
        if self._cuda_kernel_ready and x.is_cuda and x.dtype == torch.float16:
            elapsed_t = torch.zeros(bsz, device=x.device, dtype=torch.int32)
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
            r_hn = r.view(bsz, seqlen, num_heads, head_dim)
            w_hn = w.view(bsz, seqlen, num_heads, head_dim)
            k_hn = k.view(bsz, seqlen, num_heads, head_dim)
            v_hn = v.view(bsz, seqlen, num_heads, head_dim)
            kk_hn = kk.view(bsz, seqlen, num_heads, head_dim)
            kka_hn = kka.view(bsz, seqlen, num_heads, head_dim)
            positions = torch.arange(seqlen, device=x.device, dtype=torch.int64)
            for idx in range(bsz):
                slot = slot_mapping_in[idx].item()
                state = self.state_cache[slot]
                y_i = wkv7_sequence(
                    state,
                    r_hn[idx],
                    w_hn[idx],
                    k_hn[idx],
                    v_hn[idx],
                    -kk_hn[idx],
                    kka_hn[idx],
                    positions,
                )
                y[idx] = y_i.view(seqlen, hidden)

        y = F.group_norm(y.view(bsz * seqlen, hidden), num_groups=num_heads, weight=self.ln_x_weight, bias=self.ln_x_bias, eps=64e-5).view(bsz, seqlen, hidden)
        y = y + (((r * k * self.r_k.view(1, 1, hidden)).view(bsz, seqlen, num_heads, head_dim).sum(dim=-1, keepdim=True) * v.view(bsz, seqlen, num_heads, head_dim)).view(bsz, seqlen, hidden))
        y = _linear_dispatch(y * g, self.output_proj)
        return y, v_first

    def _forward_decode(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping_in: torch.Tensor,
        slot_mapping_out: torch.Tensor,
        v_first: torch.Tensor | None = None,
    ):
        bsz, hidden = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim
        context = get_context()
        assume_equal_slots = getattr(context, "force_regular_decode", False)

        x_prev = self.att_tokenshift_cache[slot_mapping_in].to(x.dtype)
        self.att_tokenshift_cache[slot_mapping_out] = x.to(self.att_tokenshift_cache.dtype)
        xx = x_prev - x
        xr = torch.addcmul(x, xx, self.x_r)
        xw = torch.addcmul(x, xx, self.x_w)
        xk = torch.addcmul(x, xx, self.x_k)
        xv = torch.addcmul(x, xx, self.x_v)
        xa = torch.addcmul(x, xx, self.x_a)
        xg = torch.addcmul(x, xx, self.x_g)

        r = self.receptance_proj(xr)
        w = self.w2_proj(torch.tanh(self.w1_proj(xw))) + self.w0
        k = self.key_proj(xk)
        v = self.value_proj(xv)
        a = torch.sigmoid(self.a2_proj(self.a1_proj(xa)) + self.a0)
        g = self.g2_proj(torch.sigmoid(self.g1_proj(xg)))
        kk = F.normalize((k * self.k_k).view(bsz, num_heads, head_dim), dim=-1, p=2.0).view_as(k)
        k = k * (1 + (a - 1) * self.k_a)
        kka = kk * a

        if self.layer_idx == 0:
            v_first = v
        else:
            assert v_first is not None
            v = v + (v_first - v) * torch.sigmoid(self.v2_proj(self.v1_proj(xv)) + self.v0)

        self._maybe_init_cuda_kernel()
        if self._cuda_kernel_ready and x.is_cuda and x.dtype == torch.float16:
            if assume_equal_slots or torch.equal(slot_mapping_in, slot_mapping_out):
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
            r_hn = r.view(bsz, num_heads, head_dim)
            w_hn = w.view(bsz, num_heads, head_dim)
            k_hn = k.view(bsz, num_heads, head_dim)
            v_hn = v.view(bsz, num_heads, head_dim)
            kk_hn = kk.view(bsz, num_heads, head_dim)
            kka_hn = kka.view(bsz, num_heads, head_dim)
            outputs = []
            for idx in range(bsz):
                slot = slot_mapping_in[idx].item()
                state = self.state_cache[slot]
                y_i = wkv7_one_step(state, r_hn[idx], w_hn[idx], k_hn[idx], v_hn[idx], -kk_hn[idx], kka_hn[idx], positions[idx])
                outputs.append(y_i)
            y = torch.stack(outputs, dim=0).view(bsz, hidden)

        y = F.group_norm(y.view_as(r), num_groups=num_heads, weight=self.ln_x_weight, bias=self.ln_x_bias, eps=64e-5)
        y = y + (
            ((r * k * self.r_k).view(-1, num_heads, head_dim).sum(dim=-1, keepdim=True) * v.view(-1, num_heads, head_dim)).view_as(r)
        )
        return self.output_proj(y * g), v_first


class RWKV7FeedForward(nn.Module):
    """RWKV-7 channel mixing (FFN)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.ffn_tokenshift_cache = None
        self.key_proj = MatmulLinear(hidden_size, intermediate_size, weight_layout="out_in")
        self.value_proj = MatmulLinear(intermediate_size, hidden_size)

    @property
    def key_weight(self):
        return self.key_proj.weight

    @property
    def value_weight(self):
        return self.value_proj.weight

    def forward(self, x: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        context = get_context()
        slot_mapping_in = context.slot_mapping_in
        slot_mapping_out = context.slot_mapping_out

        if is_prefill:
            return self._forward_prefill(x, slot_mapping_in, slot_mapping_out)
        return self._forward_decode(x, slot_mapping_in, slot_mapping_out)

    def _forward_prefill(self, x: torch.Tensor, slot_mapping_in: torch.Tensor, slot_mapping_out: torch.Tensor):
        if x.dim() == 3:
            bsz, seqlen, _ = x.shape
            context_lens = get_context().context_lens
            ffn_cache_in = self.ffn_tokenshift_cache[slot_mapping_in].to(x.dtype)
            x_prev = torch.cat((ffn_cache_in.unsqueeze(1), x[:, :-1, :]), dim=1)
            starts = (seqlen - context_lens).to(torch.long)
            x_prev[torch.arange(bsz, device=x.device), starts] = ffn_cache_in
            self.ffn_tokenshift_cache[slot_mapping_out] = x[:, -1, :].to(self.ffn_tokenshift_cache.dtype)
            xx = x_prev - x
            k = x + xx * self.x_k
            k = torch.relu(_linear_dispatch(k, self.key_proj)) ** 2
            return _linear_dispatch(k, self.value_proj)

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
        k = torch.relu(self.key_proj(k)) ** 2
        return self.value_proj(k)

    def _forward_decode(self, x: torch.Tensor, slot_mapping_in: torch.Tensor, slot_mapping_out: torch.Tensor):
        x_prev = self.ffn_tokenshift_cache[slot_mapping_in].to(x.dtype)
        xx = x_prev - x
        self.ffn_tokenshift_cache[slot_mapping_out] = x.to(self.ffn_tokenshift_cache.dtype)
        k = torch.addcmul(x, xx, self.x_k)
        k = torch.relu(_linear_dispatch(k, self.key_proj)) ** 2
        return _linear_dispatch(k, self.value_proj)


class RWKV7Block(nn.Module):
    """RWKV-7 transformer block."""

    def __init__(self, layer_idx: int, hidden_size: int, num_heads: int, head_dim: int, intermediate_size: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.ln1 = LayerNorm(hidden_size)
        self.att = RWKV7Attention(layer_idx, hidden_size, num_heads, head_dim)
        self.ln2 = LayerNorm(hidden_size)
        self.ffn = RWKV7FeedForward(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        v_first: torch.Tensor | None = None,
        att_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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

        xx = self.ln1(x)
        if att_mask is not None:
            xx.masked_fill_(att_mask, 0)
        h, v_first = self.att(positions, xx, is_prefill, v_first, att_mask)
        x.add_(h)
        if att_mask is not None:
            x.masked_fill_(att_mask, 0)

        h = self.ln2(x)
        if att_mask is not None:
            h.masked_fill_(att_mask, 0)
        h = self.ffn(h, is_prefill)
        x.add_(h)
        if att_mask is not None:
            x.masked_fill_(att_mask, 0)

        return x, v_first
