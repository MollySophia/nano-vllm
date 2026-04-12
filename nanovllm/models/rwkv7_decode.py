import torch
import torch.nn.functional as F

from nanovllm.models.rwkv7_ops import (
    _is_contiguous_in_order,
    _rwkv7_decode_block_batch_contiguous,
    _rwkv7_decode_block_one_contiguous,
)
from nanovllm.utils.context import get_context


def _resolve_contiguous_decode_window(context, require_single: bool = False):
    if getattr(context, "force_contiguous_decode", False):
        slot_in_start = int(getattr(context, "contiguous_decode_slot_in_start"))
        slot_out_start = int(getattr(context, "contiguous_decode_slot_out_start"))
        slot_count = int(getattr(context, "contiguous_decode_slot_count"))
    else:
        slot_mapping_in = context.slot_mapping_in
        slot_mapping_out = context.slot_mapping_out
        if (
            slot_mapping_in is None
            or slot_mapping_out is None
            or slot_mapping_in.numel() != slot_mapping_out.numel()
            or not _is_contiguous_in_order(slot_mapping_in)
            or not _is_contiguous_in_order(slot_mapping_out)
        ):
            return None
        slot_in_start = int(slot_mapping_in[0].item())
        slot_out_start = int(slot_mapping_out[0].item())
        slot_count = slot_mapping_in.numel()
    if require_single and slot_count != 1:
        return None
    return slot_in_start, slot_out_start, slot_count


def _run_contiguous_decode_block(model, block, x: torch.Tensor, positions: torch.Tensor, v_first, slot_in_start: int, slot_out_start: int, slot_count: int):
    slot_in_end = slot_in_start + slot_count
    slot_out_end = slot_out_start + slot_count
    if slot_count == 1:
        return _rwkv7_decode_block_one_contiguous(
            x,
            block.att.att_tokenshift_cache[slot_in_start:slot_in_end],
            block.att.att_tokenshift_cache[slot_out_start:slot_out_end],
            block.att.state_cache[slot_in_start:slot_in_end],
            block.att.state_cache[slot_out_start:slot_out_end],
            block.ffn.ffn_tokenshift_cache[slot_in_start:slot_in_end],
            block.ffn.ffn_tokenshift_cache[slot_out_start:slot_out_end],
            positions,
            v_first,
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
            block.att.w1_proj,
            block.att.w2_proj,
            block.att.a0,
            block.att.a1_proj,
            block.att.a2_proj,
            block.att.v0,
            block.att.v1_proj,
            block.att.v2_proj,
            block.att.g1_proj,
            block.att.g2_proj,
            block.att.k_k,
            block.att.k_a,
            block.att.r_k,
            block.att.receptance_proj,
            block.att.key_proj,
            block.att.value_proj,
            block.att.output_proj,
            block.att.ln_x_weight,
            block.att.ln_x_bias,
            block.ln1.gamma,
            block.ln1.beta,
            block.ln1.eps,
            block.ln2.gamma,
            block.ln2.beta,
            block.ln2.eps,
            block.ffn.x_k,
            block.ffn.key_proj,
            block.ffn.value_proj,
            model.decode_tokenshift_scratch,
        )
    return _rwkv7_decode_block_batch_contiguous(
        x,
        block.att.att_tokenshift_cache[slot_in_start:slot_in_end],
        block.att.att_tokenshift_cache[slot_out_start:slot_out_end],
        block.att.state_cache[slot_in_start:slot_in_end],
        block.att.state_cache[slot_out_start:slot_out_end],
        block.ffn.ffn_tokenshift_cache[slot_in_start:slot_in_end],
        block.ffn.ffn_tokenshift_cache[slot_out_start:slot_out_end],
        positions,
        v_first,
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
        block.att.w1_proj,
        block.att.w2_proj,
        block.att.a0,
        block.att.a1_proj,
        block.att.a2_proj,
        block.att.v0,
        block.att.v1_proj,
        block.att.v2_proj,
        block.att.g1_proj,
        block.att.g2_proj,
        block.att.k_k,
        block.att.k_a,
        block.att.r_k,
        block.att.receptance_proj,
        block.att.key_proj,
        block.att.value_proj,
        block.att.output_proj,
        block.att.ln_x_weight,
        block.att.ln_x_bias,
        block.ln1.gamma,
        block.ln1.beta,
        block.ln1.eps,
        block.ln2.gamma,
        block.ln2.beta,
        block.ln2.eps,
        block.ffn.x_k,
        block.ffn.key_proj,
        block.ffn.value_proj,
        model.decode_tokenshift_scratch,
    )


def rwkv7_forward_decode(model, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    context = get_context()
    slot_mapping_in = context.slot_mapping_in
    slot_mapping_out = context.slot_mapping_out
    v_first = None
    force_contiguous_decode = getattr(context, "force_contiguous_decode", False)
    use_contiguous_decode = (
        x.dim() == 2
        and (
            force_contiguous_decode
            or (
                x.size(0) > 1
                and not context.force_regular_decode
                and _is_contiguous_in_order(slot_mapping_in)
                and _is_contiguous_in_order(slot_mapping_out)
                and slot_mapping_in.numel() == slot_mapping_out.numel()
            )
        )
    )
    if use_contiguous_decode:
        window = _resolve_contiguous_decode_window(context)
        assert window is not None
        slot_in_start, slot_out_start, slot_count = window
        for block in model.blocks:
            x, v_first = _run_contiguous_decode_block(
                model,
                block,
                x,
                positions,
                v_first,
                slot_in_start,
                slot_out_start,
                slot_count,
            )
    else:
        for block in model.blocks:
            h = F.layer_norm(x, (block.ln1.hidden_size,), block.ln1.gamma, block.ln1.beta, block.ln1.eps)
            h, v_first = block.att._forward_decode(h, positions, slot_mapping_in, slot_mapping_out, v_first)
            x.add_(h)

            h = F.layer_norm(x, (block.ln2.hidden_size,), block.ln2.gamma, block.ln2.beta, block.ln2.eps)
            h = block.ffn._forward_decode(h, slot_mapping_in, slot_mapping_out)
            x.add_(h)

    return F.layer_norm(x, (model.ln_out.hidden_size,), model.ln_out.gamma, model.ln_out.beta, model.ln_out.eps)


def rwkv7_forward_one(model, input_ids: torch.Tensor, positions: torch.Tensor):
    context = get_context()
    assert not context.is_prefill
    if input_ids.dim() != 1 or input_ids.numel() != 1 or positions.dim() != 1 or positions.numel() != 1:
        return None

    window = _resolve_contiguous_decode_window(context, require_single=True)
    if window is None:
        return None

    slot_in_start, slot_out_start, _ = window
    x = model.emb(input_ids)
    v_first = None
    for block in model.blocks:
        x, v_first = _run_contiguous_decode_block(
            model,
            block,
            x,
            positions,
            v_first,
            slot_in_start,
            slot_out_start,
            1,
        )
    return F.layer_norm(x, (model.ln_out.hidden_size,), model.ln_out.gamma, model.ln_out.beta, model.ln_out.eps)
