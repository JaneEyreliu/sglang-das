"""HY4 checkpoint layout, format selection and dense/shared expert semantics."""

import json
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8LinearMethod
from sglang.srt.layers.quantization.slimquant_w4a8_marlin import (
    HYV4SharedExpertLinearMethod,
    SlimQuantW4A8Int8MarlinConfig,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.hunyuan_v4 import (
    HYV4ForCausalLM,
    hyv4_shared_experts_fusion_disable_reason,
    normalize_hyv4_weight_name,
)
from sglang.srt.models.hunyuan_v4_nextn import HYV4ForCausalLMNextN, _mtp_quant_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _layer():
    layer = torch.nn.Module()
    # Every signed INT4 value, including both extrema, in the low-even layout.
    values = torch.arange(-8, 8, dtype=torch.int8).view(2, 8)
    packed = (
        (values[:, 0::2].to(torch.uint8) & 15)
        | ((values[:, 1::2].to(torch.uint8) & 15) << 4)
    ).view(torch.int8)
    for name in ("w13_weight", "w2_weight"):
        layer.register_parameter(
            name, torch.nn.Parameter(packed.clone(), requires_grad=False)
        )
        layer.register_parameter(
            name + "_scale",
            torch.nn.Parameter(torch.tensor([[0.25], [2.0]]), requires_grad=False),
        )
    return layer, values


def test_routed_weight_normalization_preserves_dequantized_values():
    layer, values = _layer()
    expected = values.float() * layer.w13_weight_scale
    SlimQuantW4A8Int8MarlinConfig(
        checkpoint_format="hy4_w4a8_v1"
    ).normalize_checkpoint_weights(layer)
    for name in ("w13_weight", "w2_weight"):
        packed = getattr(layer, name)
        even = packed >> 4
        odd = (packed << 4) >> 4
        unpacked = torch.stack((even, odd), dim=-1).flatten(-2)
        actual = unpacked.float() * getattr(layer, name + "_scale") * 16
        torch.testing.assert_close(actual, expected)


def test_legacy_format_is_not_normalized():
    layer, _ = _layer()
    before = {k: v.clone() for k, v in layer.state_dict().items()}
    SlimQuantW4A8Int8MarlinConfig().normalize_checkpoint_weights(layer)
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_shared_weight_unpack_preserves_true_scale(monkeypatch):
    source, values = _layer()
    layer = torch.nn.Module()
    layer.weight = source.w13_weight
    layer.weight_scale = source.w13_weight_scale
    monkeypatch.setattr(
        SlimQuantW4A8Int8LinearMethod,
        "process_weights_after_loading",
        lambda self, layer: None,
    )
    method = HYV4SharedExpertLinearMethod(
        SlimQuantW4A8Int8MarlinConfig(checkpoint_format="hy4_w4a8_v1")
    )
    method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight, values)
    torch.testing.assert_close(layer.weight_scale, torch.tensor([[0.25], [2.0]]))


@pytest.mark.parametrize(
    "name,expected",
    [
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight.packed",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
        ),
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight.scale",
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale",
        ),
        (
            "model.layers.0.self_attn.learnable_sink_param.weight",
            "model.layers.0.self_attn.learnable_sink_param",
        ),
        (
            "model.layers.0.hc_attn_layer.hc_pre.hc_base.weight",
            "model.layers.0.hc_attn_layer.hc_pre.hc_base",
        ),
        ("model.mtp_layers.0.enorm.weight.weight", "model.mtp_layers.0.enorm.weight"),
        (
            "model.layers.0.self_attn.linear_gate.weight",
            "model.layers.0.self_attn.linear_gate.weight",
        ),
    ],
)
def test_component_weight_names(name, expected):
    assert normalize_hyv4_weight_name(name) == expected


def test_format_dispatch_preserves_legacy_dense_layers():
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    new = SlimQuantW4A8Int8MarlinConfig(checkpoint_format="hy4_w4a8_v1")
    assert isinstance(
        new.get_quant_method(layer, "model.layers.0.self_attn.linear_gate"),
        UnquantizedLinearMethod,
    )
    assert isinstance(
        new.get_quant_method(layer, "model.layers.0.mlp.shared_experts.down_proj"),
        HYV4SharedExpertLinearMethod,
    )
    legacy = SlimQuantW4A8Int8MarlinConfig()
    assert isinstance(
        legacy.get_quant_method(layer, "model.layers.0.self_attn.q_proj"),
        SlimQuantW4A8Int8LinearMethod,
    )
    qwen = SlimQuantW4A8Int8MarlinConfig.from_config(
        {"hf_config": {"architectures": ["Qwen4ExpForConditionalGeneration"]}}
    )
    assert isinstance(
        qwen.get_quant_method(layer, "model.layers.0.self_attn.q_proj"),
        UnquantizedLinearMethod,
    )
    assert _mtp_quant_config(new).checkpoint_format == "hy4_w4a8_v1"


@pytest.mark.parametrize(
    "limit,shared,disabled", [(7.0, 1, True), (None, 1, False), (7.0, 0, False)]
)
def test_shared_fusion_respects_clipping(limit, shared, disabled):
    config = SimpleNamespace(swiglu_limit=limit, n_shared_experts=shared)
    for cls in (HYV4ForCausalLM, HYV4ForCausalLMNextN):
        assert bool(cls.shared_experts_fusion_disable_reason(config, None)) == disabled
    assert bool(hyv4_shared_experts_fusion_disable_reason(config, None)) == disabled


@pytest.mark.parametrize("fmt", ["hy4_w4a8_v1", "unknown"])
def test_manifest_selects_format_without_hf_quantization_config(tmp_path, fmt):
    from sglang.srt.model_loader.weight_utils import get_quant_config

    (tmp_path / "hy4-assets.json").write_text(json.dumps({"format": fmt}))
    model = SimpleNamespace(
        quantization="slimquant_w4a8_marlin",
        hf_config=SimpleNamespace(model_type="hy_v4"),
        model_path=str(tmp_path),
    )
    if fmt == "unknown":
        with pytest.raises(ValueError, match="Unsupported HY4 checkpoint format"):
            get_quant_config(model, SimpleNamespace(), {})
    else:
        assert get_quant_config(model, SimpleNamespace(), {}).checkpoint_format == fmt


@pytest.mark.parametrize("tp_size", [1, 2, 8])
@pytest.mark.parametrize("projection", ["gate_up", "down"])
def test_shared_weights_load_real_tp_shards(monkeypatch, tp_size, projection):
    from sglang.srt.layers import linear as linear_module

    monkeypatch.setattr(linear_module, "_disable_hip_linear_quant", False)
    # Strategy 3 retains the row-major INT8 layout without kernel autotuning.
    monkeypatch.setenv("W8A8_SUPPORT_METHODS", "3")
    config = SlimQuantW4A8Int8MarlinConfig(checkpoint_format="hy4_w4a8_v1")
    input_size, output_size = (16, 32) if projection == "gate_up" else (32, 16)
    parts = 2 if projection == "gate_up" else 1
    weights = [
        (
            (
                torch.arange(output_size * input_size).view(output_size, input_size)
                + 3 * shard
            )
            % 16
            - 8
        ).to(torch.int8)
        for shard in range(parts)
    ]
    scales = [
        torch.arange(1, output_size + 1, dtype=torch.float32).view(-1, 1) / 8 + shard
        for shard in range(parts)
    ]
    for rank in range(tp_size):
        common = dict(bias=False, quant_config=config, tp_rank=rank, tp_size=tp_size)
        prefix = "model.layers.0.mlp.shared_experts."
        if projection == "gate_up":
            layer = linear_module.MergedColumnParallelLinear(
                input_size,
                [output_size, output_size],
                prefix=prefix + "gate_up_proj",
                **common,
            )
        else:
            layer = linear_module.RowParallelLinear(
                input_size,
                output_size,
                prefix=prefix + "down_proj",
                **common,
            )
        assert isinstance(layer.quant_method, HYV4SharedExpertLinearMethod)
        for shard, (values, scale) in enumerate(zip(weights, scales)):
            packed = (values[:, ::2].to(torch.uint8) & 15) | (
                (values[:, 1::2].to(torch.uint8) & 15) << 4
            )
            args = (shard,) if projection == "gate_up" else ()
            layer.weight.weight_loader(layer.weight, packed.view(torch.int8), *args)
            layer.weight_scale.weight_loader(layer.weight_scale, scale, *args)
        layer.quant_method.process_weights_after_loading(layer)
        if projection == "gate_up":
            span = slice(
                rank * output_size // tp_size, (rank + 1) * output_size // tp_size
            )
            expected_weight = torch.cat([w[span] for w in weights])
            expected_scale = torch.cat([s[span] for s in scales])
        else:
            span = slice(
                rank * input_size // tp_size, (rank + 1) * input_size // tp_size
            )
            expected_weight = weights[0][:, span]
            expected_scale = scales[0]
        torch.testing.assert_close(layer.weight, expected_weight)
        torch.testing.assert_close(layer.weight_scale, expected_scale)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
