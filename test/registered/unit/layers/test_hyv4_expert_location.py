"""HYV4 expert layout and HIP shared-slot remapping regressions."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def test_hyv4_expert_location_config(monkeypatch):
    from sglang.srt import model_loader
    from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
    from sglang.srt.models.hunyuan_v4 import HYV4ForCausalLM

    monkeypatch.setattr(
        model_loader,
        "get_model_architecture",
        lambda _: (HYV4ForCausalLM, "HYV4ForCausalLM"),
    )
    config = SimpleNamespace(
        hf_config=SimpleNamespace(num_hidden_layers=48, n_routed_experts=256)
    )
    actual = ModelConfigForExpertLocation.from_model_config(config)
    assert actual == ModelConfigForExpertLocation(
        num_layers=48, num_logical_experts=256, num_groups=None
    )


@pytest.mark.parametrize("use_aiter", [False, True])
@pytest.mark.parametrize("shared", [1, 2])
@pytest.mark.parametrize("rows", [0, 3])
def test_hip_eplb_maps_only_routed_columns_before_shared_slots(
    monkeypatch, shared, rows, use_aiter
):
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe import topk as topk_module

    monkeypatch.setattr(topk_module, "_is_cuda", False)
    monkeypatch.setattr(topk_module, "_is_hip", True)
    monkeypatch.setattr(topk_module, "_is_hcu", True)
    monkeypatch.setattr(topk_module, "_use_aiter", use_aiter)
    monkeypatch.setattr(topk_module, "_eplb_remap_enabled", lambda: True)
    monkeypatch.setattr(topk_module, "has_per_rank_fused_shared_slots", lambda n: n > 0)
    monkeypatch.setattr(
        topk_module,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(is_deepep=lambda: True),
    )
    monkeypatch.setattr(
        topk_module,
        "get_moe_runner_backend",
        lambda: SimpleNamespace(is_deep_gemm=lambda: True),
    )
    monkeypatch.setattr(
        topk_module,
        "get_parallel",
        lambda: SimpleNamespace(moe_ep_size=2, moe_ep_rank=1),
    )
    config = SimpleNamespace(
        num_fused_shared_experts=shared,
        fused_shared_experts_scaling_factor=None,
        routed_scaling_factor=2.0,
        allow_routed_experts_capture=False,
    )
    mapping = torch.tensor([7, 3, 6, 0], dtype=torch.int64)
    info = ExpertLocationDispatchInfo("static", mapping, None, None, 8)
    routed = torch.tensor([[0, 1], [2, 3], [1, 2]], dtype=torch.int64)[:rows]
    # Shared placeholders deliberately exceed the routed-only map's size.
    ids = (
        routed.clone()
        if use_aiter
        else torch.cat([routed, torch.full((rows, shared), 4)], dim=1)
    )
    if use_aiter:
        from sglang.kernels.ops.moe import fused_moe_triton_kernels

        def append_shared(ids, weights, count, scale, shared_base, local_routed, **kw):
            # Aiter has NOT appended shared placeholders at the remap stage.
            # Every input column here must already contain physical routed IDs.
            torch.testing.assert_close(ids, mapping[routed])
            physical = ids + (ids // local_routed) * count
            shared_ids = (shared_base + torch.arange(count)).expand(rows, count)
            return (
                torch.cat([physical, shared_ids], dim=1),
                torch.cat([weights, torch.full((rows, count), scale)], dim=1),
            )

        monkeypatch.setattr(
            fused_moe_triton_kernels,
            "fused_append_remap_shared_experts_deepep",
            append_shared,
        )
    weights = torch.ones(ids.shape)
    actual, actual_weights, recorded = topk_module._post_process_topk_ids(
        ids,
        weights,
        config,
        torch.zeros((rows, 4)),
        1,
        expert_location_dispatch_info=info,
    )
    physical = mapping[routed]
    expected_routed = physical + (physical // 4) * shared
    expected_shared = (8 + shared + torch.arange(shared)).expand(rows, shared)
    torch.testing.assert_close(
        actual, torch.cat([expected_routed, expected_shared], dim=1)
    )
    torch.testing.assert_close(recorded, physical)
    torch.testing.assert_close(actual_weights[:, :2], torch.ones((rows, 2)))
    torch.testing.assert_close(
        actual_weights[:, 2:], torch.full((rows, shared), 1.0 if use_aiter else 0.5)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
