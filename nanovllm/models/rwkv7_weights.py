import gc

import torch
import torch.nn.functional as F

from nanovllm.layers.linear import MarlinInt8Linear, get_marlin_impl_or_raise


def load_rwkv7_pth_into_model(model, pth_path: str):
    z = torch.load(pth_path, map_location="cpu")
    keys = list(z.keys())

    for key in keys:
        if (
            key.endswith("att.w1")
            or key.endswith("att.w2")
            or key.endswith("att.a1")
            or key.endswith("att.a2")
            or key.endswith("att.v1")
            or key.endswith("att.v2")
            or key.endswith("att.g1")
            or key.endswith("att.g2")
            or key.endswith("ffn.value.weight")
        ):
            z[key] = z[key].t()

        dtype = torch.get_default_dtype()
        if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")
        z[key] = z[key].squeeze().to(dtype=dtype, device=device)

        if key.endswith("att.r_k"):
            z[key] = z[key].flatten()

        z[key] = z[key].contiguous()

    z["emb.weight"] = F.layer_norm(
        z["emb.weight"],
        (model.hidden_size,),
        weight=z["blocks.0.ln0.weight"],
        bias=z["blocks.0.ln0.bias"],
    )

    z["blocks.0.att.v0"] = z["blocks.0.att.a0"]
    z["blocks.0.att.v1"] = z["blocks.0.att.a1"]
    z["blocks.0.att.v2"] = z["blocks.0.att.a2"]

    model.z = z
    load_rwkv7_state_into_modules(model)


def load_rwkv7_state_into_modules(model):
    z = model.z

    if model.emb.tp_size == 1:
        model.emb.weight.data = z["emb.weight"]
    else:
        model.emb.weight.data.copy_(z["emb.weight"])

    for idx, block in enumerate(model.blocks):
        bbb = f"blocks.{idx}."
        att = f"blocks.{idx}.att."
        ffn = f"blocks.{idx}.ffn."

        block.ln1.gamma.data = z[bbb + "ln1.weight"]
        block.ln1.beta.data = z[bbb + "ln1.bias"]
        block.ln2.gamma.data = z[bbb + "ln2.weight"]
        block.ln2.beta.data = z[bbb + "ln2.bias"]

        block.ffn.key_proj.weight.data = z[ffn + "key.weight"]
        block.ffn.value_proj.weight.data = z[ffn + "value.weight"]
        block.ffn.register_buffer("x_k", z[ffn + "x_k"])

        block.att.register_buffer("x_r", z[att + "x_r"])
        block.att.register_buffer("x_w", z[att + "x_w"])
        block.att.register_buffer("x_k", z[att + "x_k"])
        block.att.register_buffer("x_v", z[att + "x_v"])
        block.att.register_buffer("x_a", z[att + "x_a"])
        block.att.register_buffer("x_g", z[att + "x_g"])
        block.att.register_buffer("w0", z[att + "w0"])
        block.att.w1_proj.weight.data = z[att + "w1"]
        block.att.w2_proj.weight.data = z[att + "w2"]
        block.att.register_buffer("a0", z[att + "a0"])
        block.att.a1_proj.weight.data = z[att + "a1"]
        block.att.a2_proj.weight.data = z[att + "a2"]
        block.att.register_buffer("v0", z[att + "v0"])
        block.att.v1_proj.weight.data = z[att + "v1"]
        block.att.v2_proj.weight.data = z[att + "v2"]
        block.att.g1_proj.weight.data = z[att + "g1"]
        block.att.g2_proj.weight.data = z[att + "g2"]
        block.att.register_buffer("k_k", z[att + "k_k"].squeeze())
        block.att.register_buffer("k_a", z[att + "k_a"].squeeze())
        block.att.register_buffer("r_k", z[att + "r_k"].flatten())
        block.att.receptance_proj.weight.data = z[att + "receptance.weight"]
        block.att.key_proj.weight.data = z[att + "key.weight"]
        block.att.value_proj.weight.data = z[att + "value.weight"]
        block.att.output_proj.weight.data = z[att + "output.weight"]
        block.att.register_buffer("ln_x_weight", z[att + "ln_x.weight"])
        block.att.register_buffer("ln_x_bias", z[att + "ln_x.bias"])

    model.ln_out.gamma.data = z["ln_out.weight"]
    model.ln_out.beta.data = z["ln_out.bias"]


def apply_rwkv7_post_load_quantization(causal_lm):
    if getattr(causal_lm.config, "rwkv_quant_int8", False):
        get_marlin_impl_or_raise()
        att_ffn_linear_cls = MarlinInt8Linear
        for idx, block in enumerate(causal_lm.model.blocks):
            device = block.att.x_r.device
            block.att.receptance_proj = att_ffn_linear_cls.from_float(block.att.receptance_proj).to(device=device)
            block.att.key_proj = att_ffn_linear_cls.from_float(block.att.key_proj).to(device=device)
            block.att.value_proj = att_ffn_linear_cls.from_float(block.att.value_proj).to(device=device)
            block.att.output_proj = att_ffn_linear_cls.from_float(block.att.output_proj).to(device=device)
            for suffix in (
                "att.receptance.weight",
                "att.key.weight",
                "att.value.weight",
                "att.output.weight",
            ):
                key = f"blocks.{idx}.{suffix}"
                if key in causal_lm.model.z:
                    del causal_lm.model.z[key]

        for idx, block in enumerate(causal_lm.model.blocks):
            device = block.ffn.x_k.device
            block.ffn.key_proj = att_ffn_linear_cls.from_float(block.ffn.key_proj).to(device=device)
            block.ffn.value_proj = att_ffn_linear_cls.from_float(block.ffn.value_proj).to(device=device)
            for suffix in ("ffn.key.weight", "ffn.value.weight"):
                key = f"blocks.{idx}.{suffix}"
                if key in causal_lm.model.z:
                    del causal_lm.model.z[key]

    if causal_lm.lm_head.tp_size == 1:
        causal_lm.lm_head.weight.data = causal_lm.model.z["head.weight"].t()
    else:
        shard_size = causal_lm.lm_head.num_embeddings_per_partition
        start_idx = causal_lm.lm_head.tp_rank * shard_size
        causal_lm.lm_head.weight.data.copy_(
            causal_lm.model.z["head.weight"].narrow(0, start_idx, shard_size).t().contiguous()
        )
    if getattr(causal_lm.config, "rwkv_quant_int8", False) and getattr(
        causal_lm.config, "rwkv_quant_int8_lm_head", False
    ):
        causal_lm.lm_head.quantize_weight_marlin_int8()
    if "head.weight" in causal_lm.model.z:
        del causal_lm.model.z["head.weight"]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
