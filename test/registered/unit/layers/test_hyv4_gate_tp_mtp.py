"""HYV4 gate TP8: embedded MTP admission, local draft and verify pre-planning.

These tests do not load the checkpoint or run attention kernels. Distributed
projection/capture coverage lives in test_hyv4_gate_tp_cuda_graph.py.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from sglang.srt.layers.hyv4_gate_tp import GateTPGraphPlan
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_utils import eagle_prepare_for_verify
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _args(**overrides):
    values = dict(
        hyv4_linear_gate_tp_size=8,
        enable_dp_attention=True,
        dp_size=32,
        tp_size=32,
        pp_size=1,
        attn_cp_size=1,
        nnodes=4,
        disaggregation_mode="decode",
        enable_torch_compile=False,
        disable_cuda_graph_padding=False,
        enable_two_batch_overlap=False,
        enable_pdmux=False,
        ep_join_mode=None,
        enable_lora=False,
        model_path="/models/hy4",
        speculative_draft_model_path="/models/hy4",
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        speculative_num_steps=2,
        speculative_num_draft_tokens=3,
        speculative_adaptive=False,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend="full"),
            prefill=SimpleNamespace(backend="disabled"),
        ),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("algorithm", [None, "NEXTN", "EAGLE"])
@pytest.mark.parametrize("backend", ["full", "disabled"])
def test_admit_plain_decode_and_fixed_mtp(monkeypatch, algorithm, backend):
    monkeypatch.setenv("SGLANG_RAGGED_VERIFY_MODE", "static")
    args = _args(speculative_algorithm=algorithm)
    args.cuda_graph_config.decode.backend = backend
    ServerArgs._check_hyv4_gate_tp(args)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"enable_two_batch_overlap": True}, "two-batch"),
        ({"speculative_algorithm": "EAGLE3"}, "NEXTN/EAGLE"),
        ({"speculative_algorithm": "DFLASH"}, "NEXTN/EAGLE"),
        ({"speculative_draft_model_path": "/models/other"}, "embedded draft"),
        ({"speculative_eagle_topk": 2}, "fixed topk"),
        ({"speculative_num_steps": 0}, "fixed topk"),
        ({"speculative_num_draft_tokens": 4}, "fixed topk"),
        ({"speculative_adaptive": True}, "fixed topk"),
    ],
)
def test_reject_unintegrated_spec_paths(monkeypatch, overrides, match):
    monkeypatch.setenv("SGLANG_RAGGED_VERIFY_MODE", "static")
    with pytest.raises(ValueError, match=match):
        ServerArgs._check_hyv4_gate_tp(_args(**overrides))


@pytest.mark.parametrize("mode", ["compact", "cap-accept"])
def test_reject_ragged_mtp_without_affecting_plain_decode(monkeypatch, mode):
    monkeypatch.setenv("SGLANG_RAGGED_VERIFY_MODE", mode)
    with pytest.raises(ValueError, match="RAGGED_VERIFY_MODE=static"):
        ServerArgs._check_hyv4_gate_tp(_args())
    ServerArgs._check_hyv4_gate_tp(_args(speculative_algorithm=None))


def test_draft_gate_is_local_without_changing_target_or_shared_args(monkeypatch):
    from sglang.srt.models import hunyuan_v4 as hy

    args = _args()
    monkeypatch.setattr(hy, "get_global_server_args", lambda: args)
    monkeypatch.setattr(
        hy, "get_parallel", lambda: SimpleNamespace(attn_tp_size=1, attn_tp_rank=0)
    )
    monkeypatch.setattr(hy, "is_dp_attention_enabled", lambda: True)
    monkeypatch.setattr(hy, "is_dsa_enable_prefill_cp", lambda: False)
    group = SimpleNamespace(world_size=8, rank_in_group=3)
    group_lookup = Mock(return_value=group)
    monkeypatch.setattr(hy, "get_hyv4_gate_tp_group", group_lookup)

    def attention_init(self, **kwargs):
        nn.Module.__init__(self)
        self.num_local_heads = kwargs["num_heads"]

    class Linear(nn.Module):
        def __init__(self, input_size, output_size, *, tp_size, tp_rank, **kwargs):
            super().__init__()
            self.tp_size, self.tp_rank = tp_size, tp_rank
            self.output_size = self.output_size_per_partition = output_size
            width = (
                input_size // tp_size if kwargs.get("input_is_parallel") else input_size
            )
            self.weight = nn.Parameter(torch.randn(output_size, width))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight), None

    monkeypatch.setattr(hy.DeepseekV2AttentionMLA, "__init__", attention_init)
    monkeypatch.setattr(hy, "RowParallelLinear", Linear)
    monkeypatch.setattr(hy, "ColumnParallelLinear", Linear)
    config = SimpleNamespace(
        rope_parameters={"rope_theta": 10000, "rope_type": "default"},
        hidden_size=64,
        num_attention_heads=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        max_position_embeddings=128,
    )
    target = hy.HYV4Attention(config, 0)
    draft = hy.HYV4Attention(config, 0, is_nextn=True)
    target_after_draft = hy.HYV4Attention(config, 1)
    assert args.hyv4_linear_gate_tp_size == 8
    assert target.gate_tp_size == target_after_draft.gate_tp_size == 8
    assert target.linear_gate.weight.shape == (16, 8)
    assert draft.gate_tp_size == draft.linear_gate.tp_size == 1
    assert draft.linear_gate.weight.shape == (16, 64)
    assert group_lookup.call_count == 2  # The draft must not access the gate group.
    x = torch.randn(6, 64)  # Two requests, three draft-extend tokens each.
    torch.testing.assert_close(
        draft.prepare_attention_output_gate(x),
        torch.nn.functional.linear(x, draft.linear_gate.weight),
    )
    assert draft.prepare_attention_output_gate(x[:0]).shape == (0, 16)


def _verify_batch():
    batch = object.__new__(ForwardBatch)
    batch.forward_mode = ForwardMode.IDLE
    batch.batch_size = 0
    batch.input_ids = torch.empty(0, dtype=torch.long)
    batch.positions = torch.empty(0, dtype=torch.long)
    batch.seq_lens_cpu = torch.empty(0, dtype=torch.long)
    batch.spec_info = EagleVerifyInput.create_idle_input(1, 2, 3, "cpu")
    return batch


def _graph_runner():
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.hyv4_gate_tp = True
    runner.require_mlp_tp_gather = False
    runner.dsa_dual_graph = True
    runner.dsa_index_topk = 8
    runner.captured_req_width = 3
    runner.capture_bs = [1, 2, 4]
    runner.model_runner = SimpleNamespace(tp_group=object())
    runner.can_run_graph = lambda fb: runner._decode_graph_batch_size(fb) <= 4
    return runner


def test_verify_coordinates_before_loading_and_reuses_prepared_geometry(monkeypatch):
    fb = _verify_batch()
    graph = _graph_runner()
    loaded = []

    def load_batch(batch):
        assert batch.hyv4_gate_tp_graph_plan is not None
        graph.bs = graph._pad_to_bucket(
            graph._decode_graph_batch_size(batch), graph.capture_bs
        )
        loaded.append(
            (
                graph.bs,
                graph.bs * graph.captured_req_width,
                graph._resolve_dsa_variant(batch),
            )
        )

    graph.load_batch = load_batch
    target = SimpleNamespace(
        model_runner=SimpleNamespace(
            decode_cuda_graph_runner=graph,
            spec_algorithm=SimpleNamespace(is_standalone=lambda: False),
        )
    )
    monkeypatch.setattr(ForwardBatch, "init_new", lambda *a, **kw: fb)
    plan = GateTPGraphPlan(3, True, "sparse")
    with patch.object(GateTPGraphPlan, "synchronize", return_value=plan) as sync:
        result, can_run = eagle_prepare_for_verify(
            fb.spec_info, None, SimpleNamespace(forward_mode=ForwardMode.IDLE), target
        )
        assert result is fb and can_run
        assert loaded == [(4, 12, "sparse")]
        assert not fb.needs_forward_metadata_init()
        assert sync.call_count == 1

        # Execute the actual ModelRunner dispatch. A second negotiation here
        # could select a different geometry than the pre-filled graph buffers.
        runner = object.__new__(ModelRunner)
        runner.device = "cuda"
        runner.is_draft_worker = False
        runner.server_args = _args()
        runner.decode_cuda_graph_runner = graph
        graph.execute = Mock(return_value="verified")
        with patch(
            "sglang.srt.model_executor.model_runner.has_forward_context",
            return_value=True,
        ):
            output = runner._forward_raw(fb, None)
        assert output.logits_output == "verified" and output.can_run_graph
        assert sync.call_count == 1
        assert graph.bs == 4


def test_verify_peer_fallback_does_not_preload_or_mark_metadata(monkeypatch):
    fb = _verify_batch()
    graph = _graph_runner()
    graph.load_batch = Mock(side_effect=AssertionError("preloaded eager fallback"))
    target = SimpleNamespace(
        model_runner=SimpleNamespace(
            decode_cuda_graph_runner=graph,
            spec_algorithm=SimpleNamespace(is_standalone=lambda: False),
        )
    )
    monkeypatch.setattr(ForwardBatch, "init_new", lambda *a, **kw: fb)
    with patch.object(
        GateTPGraphPlan, "synchronize", return_value=GateTPGraphPlan(3, False, "sparse")
    ):
        result, can_run = eagle_prepare_for_verify(
            fb.spec_info, None, SimpleNamespace(forward_mode=ForwardMode.IDLE), target
        )
    assert result is fb and not can_run
    assert fb.needs_forward_metadata_init()
    graph.load_batch.assert_not_called()


def test_fresh_verify_does_not_inherit_previous_round_plan():
    graph = _graph_runner()
    fb = _verify_batch()
    fb.hyv4_gate_tp_graph_plan = GateTPGraphPlan(1, True, "dense")
    with patch.object(
        GateTPGraphPlan, "synchronize", return_value=GateTPGraphPlan(3, True, "sparse")
    ) as sync:
        assert graph.prepare_hyv4_gate_graph(fb, reuse_prepared=True)
    sync.assert_called_once()
    assert fb.hyv4_gate_tp_graph_plan.batch_size == 3


def test_other_models_keep_existing_verify_preparation(monkeypatch):
    fb = _verify_batch()
    graph = SimpleNamespace(can_run_graph=Mock(return_value=True), load_batch=Mock())
    target = SimpleNamespace(
        model_runner=SimpleNamespace(
            decode_cuda_graph_runner=graph,
            spec_algorithm=SimpleNamespace(is_standalone=lambda: False),
        )
    )
    monkeypatch.setattr(ForwardBatch, "init_new", lambda *a, **kw: fb)
    with patch.object(
        GateTPGraphPlan,
        "synchronize",
        side_effect=AssertionError("gate collective on another model"),
    ):
        result, can_run = eagle_prepare_for_verify(
            fb.spec_info, None, SimpleNamespace(forward_mode=ForwardMode.IDLE), target
        )
    assert result is fb and can_run
    graph.can_run_graph.assert_called_once_with(fb)
    graph.load_batch.assert_called_once_with(fb)
    assert not fb.needs_forward_metadata_init()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
