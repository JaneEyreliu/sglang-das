"""Torch-free HYV4 MTP checkpoint names and completeness checks."""


def normalize_hyv4_mtp_weight_name(name):
    """Return a name relative to the single MTP layer, or None for trunk weights."""
    for prefix in ("model.mtp_layers.0.", "model.mtp.layers.0."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    else:
        return None
    name = name.replace("self_attn.g_proj.", "self_attn.linear_gate.")
    for source, target in (
        (".weight.weight", ".weight"),
        (".bias.weight", ".bias"),
        (".weight.packed", ".weight"),
        (".weight.scale", ".weight_scale"),
    ):
        if name.endswith(source):
            name = name.removesuffix(source) + target
            break
    for suffix in ("learnable_sink_param", "e_score_correction_bias"):
        if name.endswith(suffix + ".weight"):
            name = name.removesuffix(".weight")
    if name == "final_layernorm.weight":
        name = "shared_head.norm.weight"
    return name


def validate_hyv4_mtp_metadata(config, weights, *, packed_experts):
    """Check unsharded checkpoint metadata before returning a loaded draft.

    weights maps normalized relative names to {shape, dtype}. Both the offline
    header checker and the streaming runtime loader use this contract. No tensor
    data is retained. Packed HY4 v1 experts have two signed INT4 values per byte.
    """
    if config.get("num_nextn_predict_layers", 1) != 1:
        raise ValueError("HYV4 MTP requires exactly one checkpoint MTP layer")
    d = config["hidden_size"]
    h, v = config["num_attention_heads"], config["v_head_dim"]
    q, kv = config["q_lora_rank"], config["kv_lora_rank"]
    nope, rope = config["qk_nope_head_dim"], config["qk_rope_head_dim"]
    ih, idim = config["index_n_heads"], config["index_head_dim"]
    experts = config["n_routed_experts"]
    dense = {
        "enorm.weight": [d],
        "hnorm.weight": [d],
        "eh_proj.weight": [d, 2 * d],
        "shared_head.norm.weight": [d],
        "input_layernorm.weight": [d],
        "post_attention_layernorm.weight": [d],
        "self_attn.linear_gate.weight": [h * v, d],
        "self_attn.learnable_sink_param": [h],
        "self_attn.q_a_proj.weight": [q, d],
        "self_attn.q_a_layernorm.weight": [q],
        "self_attn.q_b_proj.weight": [h * (nope + rope), q],
        "self_attn.kv_a_proj_with_mqa.weight": [kv + rope, d],
        "self_attn.kv_a_layernorm.weight": [kv],
        "self_attn.kv_b_proj.weight": [h * (nope + v), kv],
        "self_attn.o_proj.weight": [d, h * v],
        "self_attn.indexer.wq_b.weight": [ih * idim, q],
        "self_attn.indexer.wk.weight": [idim, d],
        "self_attn.indexer.k_norm.weight": [idim],
        "self_attn.indexer.k_norm.bias": [idim],
        "self_attn.indexer.weights_proj.weight": [ih, d],
        "mlp.gate.weight": [experts, d],
        "mlp.gate.e_score_correction_bias": [experts],
    }
    floating = {
        "BF16",
        "F16",
        "F32",
        "torch.bfloat16",
        "torch.float16",
        "torch.float32",
    }
    packed = {"I8", "U8", "torch.int8", "torch.uint8"}

    def check(name, shape, dtypes):
        meta = weights.get(name)
        if meta is None:
            raise ValueError(f"Missing HYV4 MTP weight: {name}")
        if list(meta["shape"]) != shape or meta["dtype"] not in dtypes:
            raise ValueError(
                f"Invalid HYV4 MTP weight {name}: {meta}; expected shape={shape}, "
                f"dtype in {sorted(dtypes)}"
            )

    for name, shape in dense.items():
        check(name, shape, floating)
    intermediate = config["moe_intermediate_size"]
    modules = [(f"mlp.experts.{i}", intermediate) for i in range(experts)]
    shared = config.get("n_shared_experts", 0) or 0
    if shared:
        modules.append(("mlp.shared_experts", intermediate * shared))
    for prefix, width in modules:
        for proj, out_size, in_size in (
            ("gate_proj", width, d),
            ("up_proj", width, d),
            ("down_proj", d, width),
        ):
            name = f"{prefix}.{proj}"
            check(
                name + ".weight",
                [out_size, in_size // 2 if packed_experts else in_size],
                packed if packed_experts else floating,
            )
            if packed_experts:
                scale = weights.get(name + ".weight_scale")
                if (
                    scale is None
                    or list(scale["shape"]) not in ([out_size], [out_size, 1])
                    or scale["dtype"] not in floating
                ):
                    raise ValueError(
                        f"Invalid or missing HYV4 MTP channel scale: {name}"
                    )
    return {"mtp_layers": 1, "mtp_hidden_size": d, "mtp_packed_experts": packed_experts}
