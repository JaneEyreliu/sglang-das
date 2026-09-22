"""Exercise load-time scales for both HIPC weight-loading implementations."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.quantization import slimquant_w4a8_marlin as quant
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "convention,multiplier", [(None, 16), ("legacy", 16), ("kernel_x16", 1)]
)
@pytest.mark.parametrize(
    "method_name",
    ["SlimQuantW4A8Int8MarlinMoEMethod", "SlimQuantW4A8Int8AiterMoEMethod"],
)
def test_hipc_loader_scales(monkeypatch, convention, multiplier, method_name):
    import deepgemm

    if convention is None:
        monkeypatch.delenv("SGLANG_W4A8_HIPC_SCALE_CONVENTION", raising=False)
    else:
        monkeypatch.setenv("SGLANG_W4A8_HIPC_SCALE_CONVENTION", convention)
    monkeypatch.setattr(quant, "_use_lightop_w4a8_marlin_moe", False)
    monkeypatch.setattr(
        quant, "get_moe_a2a_backend", lambda: SimpleNamespace(is_megamoe=lambda: False)
    )
    monkeypatch.setattr(deepgemm, "pack_w4a8_moe_hipc_weight", lambda w: w.clone())
    layer = torch.nn.Module()
    for name in ("w13_weight", "w2_weight"):
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.ones(2, 4, 4, dtype=torch.int8), requires_grad=False
            ),
        )
        layer.register_parameter(
            name + "_scale",
            torch.nn.Parameter(torch.full((2, 4, 1), 0.125), requires_grad=False),
        )
    # Call the actual loaders, bypassing constructors that resolve GPU kernels.
    method = SimpleNamespace(
        use_deepep=True,
        quant_config=SimpleNamespace(normalize_checkpoint_weights=lambda _: None),
    )
    getattr(quant, method_name).process_weights_after_loading(method, layer)
    for name in ("w13_weight_scale", "w2_weight_scale"):
        torch.testing.assert_close(
            getattr(layer, name), torch.full((2, 4, 1), 0.125 * multiplier)
        )
    kernel_multiplier = 16 if convention == "kernel_x16" else 1
    torch.testing.assert_close(
        layer.w13_weight_scale * kernel_multiplier, torch.full((2, 4, 1), 2.0)
    )


def test_reject_unknown_scale_convention(monkeypatch):
    monkeypatch.setenv("SGLANG_W4A8_HIPC_SCALE_CONVENTION", "guess")
    with pytest.raises(ValueError, match="legacy or kernel_x16"):
        quant._hipc_weight_scale_multiplier()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
