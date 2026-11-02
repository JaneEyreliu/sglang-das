"""Unit tests for the HCU HIPC W4A8 SBO completion-signal contract."""

import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers import hyv4_gate_a2a_overlap
from sglang.srt.layers.moe import hcu_w4a8_sbo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHcuW4A8Sbo(CustomTestCase):
    def setUp(self):
        super().setUp()
        env_patch = patch.dict(os.environ, {"SGLANG_HCU_SBO_GRAPH_FORK_BARRIER": "0"})
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_overlap_records_event_and_metadata(self):
        out, signal = object(), object()
        calls = []
        gemm = Mock(return_value=(64, 24))
        module = SimpleNamespace(m_grouped_w4a8_gemm_nt_masked_hipc=gemm)
        args = SimpleNamespace(
            signal=signal,
            start_event=SimpleNamespace(record=lambda: calls.append("event")),
        )
        metadata = {}
        with patch.object(hcu_w4a8_sbo, "import_module", return_value=module):
            result = hcu_w4a8_sbo.run_hcu_w4a8_hipc(
                (), (), out, object(), 12, overlap_args=args, meta_overlap_args=metadata
            )
        self.assertIs(result, out)
        self.assertEqual(calls, ["event"])
        self.assertEqual(metadata, {"block_m": 64, "threshold": 24})
        self.assertTrue(gemm.call_args.kwargs["enable_overlap"])
        self.assertIs(gemm.call_args.kwargs["signal"], signal)

    def test_invalid_metadata_fails_closed(self):
        module = SimpleNamespace(
            m_grouped_w4a8_gemm_nt_masked_hipc=Mock(return_value=(64, 0))
        )
        args = SimpleNamespace(
            signal=object(), start_event=SimpleNamespace(record=lambda: None)
        )
        with patch.object(hcu_w4a8_sbo, "import_module", return_value=module):
            with self.assertRaisesRegex(
                ValueError, "Invalid W4A8 HIPC completion metadata"
            ):
                hcu_w4a8_sbo.run_hcu_w4a8_hipc(
                    (),
                    (),
                    object(),
                    object(),
                    12,
                    overlap_args=args,
                    meta_overlap_args={},
                )

    def test_overlap_requires_shared_metadata(self):
        module = SimpleNamespace(
            m_grouped_w4a8_gemm_nt_masked_hipc=Mock(return_value=(64, 24))
        )
        with patch.object(hcu_w4a8_sbo, "import_module", return_value=module):
            with self.assertRaisesRegex(ValueError, "shared DeepEP metadata"):
                hcu_w4a8_sbo.run_hcu_w4a8_hipc(
                    (), (), object(), object(), 12, overlap_args=Mock()
                )

    def test_graph_fork_precedes_start_event_and_gemm(self):
        for switch, accumulate in ((None, None), (None, "1"), ("1", "1")):
            with self.subTest(switch=switch, accumulate=accumulate):
                self._check_graph_fork_order(switch, accumulate)

    def _check_graph_fork_order(self, switch, accumulate):
        calls = []
        stream, out, signal = object(), object(), object()
        api = Mock()
        api.add_barrier.side_effect = lambda s: calls.append(("barrier", s))
        gemm = Mock(side_effect=lambda *a, **kw: (calls.append("gemm") or (64, 24)))
        args = SimpleNamespace(
            signal=signal,
            start_event=SimpleNamespace(record=lambda: calls.append("event")),
        )
        metadata = {}
        with (
            patch.dict(
                os.environ,
                {},
            ),
            patch.object(torch.version, "hip", "test-hip"),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(hyv4_gate_a2a_overlap, "_get_hip_graph_api", return_value=api),
            patch.object(
                hcu_w4a8_sbo,
                "import_module",
                return_value=SimpleNamespace(m_grouped_w4a8_gemm_nt_masked_hipc=gemm),
            ),
        ):
            for name, value in (
                ("SGLANG_HCU_SBO_GRAPH_FORK_BARRIER", switch),
                ("HIP_GRAPH_ACCUMULATE_DISPATCH", accumulate),
            ):
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            result = hcu_w4a8_sbo.run_hcu_w4a8_hipc(
                (), (), out, object(), 12, overlap_args=args, meta_overlap_args=metadata
            )
        self.assertEqual(calls, [("barrier", stream), "event", "gemm"])
        self.assertIs(result, out)
        self.assertIs(gemm.call_args.kwargs["signal"], signal)
        self.assertEqual(metadata, {"block_m": 64, "threshold": 24})

    def test_graph_fork_does_not_touch_other_execution_modes(self):
        cases = (
            # switch, HIP runtime, accumulate dispatch, capture, SBO
            ("0", "test-hip", "1", True, True),
            ("1", None, "1", True, True),
            ("1", "test-hip", "0", True, True),
            ("1", "test-hip", "1", False, True),
            ("1", "test-hip", "1", True, False),
            (None, None, "1", True, True),
            (None, "test-hip", "0", True, True),
            (None, "test-hip", "1", False, True),
            (None, "test-hip", "1", True, False),
        )
        for switch, hip, accumulate, capture, sbo in cases:
            with (
                self.subTest(case=(switch, hip, accumulate, capture, sbo)),
                patch.dict(
                    os.environ,
                    {
                        "HIP_GRAPH_ACCUMULATE_DISPATCH": accumulate,
                    },
                ),
                patch.object(torch.version, "hip", hip),
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=capture
                ) as is_capture,
                patch.object(torch.cuda, "current_stream") as current_stream,
                patch.object(hyv4_gate_a2a_overlap, "_get_hip_graph_api") as get_api,
                patch.object(
                    hcu_w4a8_sbo,
                    "import_module",
                    return_value=SimpleNamespace(
                        m_grouped_w4a8_gemm_nt_masked_hipc=Mock(return_value=(64, 24))
                    ),
                ),
            ):
                if switch is None:
                    os.environ.pop("SGLANG_HCU_SBO_GRAPH_FORK_BARRIER", None)
                else:
                    os.environ["SGLANG_HCU_SBO_GRAPH_FORK_BARRIER"] = switch
                event = Mock()
                args = (
                    SimpleNamespace(signal=object(), start_event=event) if sbo else None
                )
                hcu_w4a8_sbo.run_hcu_w4a8_hipc(
                    (),
                    (),
                    object(),
                    object(),
                    12,
                    overlap_args=args,
                    meta_overlap_args={},
                )
                get_api.assert_not_called()
                current_stream.assert_not_called()
                if not (
                    switch != "0" and hip is not None and accumulate != "0" and sbo
                ):
                    is_capture.assert_not_called()
                if sbo:
                    event.record.assert_called_once_with()
                else:
                    event.record.assert_not_called()

    def test_graph_fork_failure_does_not_launch_gemm(self):
        api = Mock()
        api.add_barrier.side_effect = RuntimeError("HIP fork failed")
        gemm, event = Mock(), Mock()
        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_HCU_SBO_GRAPH_FORK_BARRIER": "1",
                    "HIP_GRAPH_ACCUMULATE_DISPATCH": "1",
                },
            ),
            patch.object(torch.version, "hip", "test-hip"),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            patch.object(torch.cuda, "current_stream", return_value=object()),
            patch.object(hyv4_gate_a2a_overlap, "_get_hip_graph_api", return_value=api),
            patch.object(
                hcu_w4a8_sbo,
                "import_module",
                return_value=SimpleNamespace(m_grouped_w4a8_gemm_nt_masked_hipc=gemm),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "HIP fork failed"):
                hcu_w4a8_sbo.run_hcu_w4a8_hipc(
                    (),
                    (),
                    object(),
                    object(),
                    12,
                    overlap_args=SimpleNamespace(signal=object(), start_event=event),
                    meta_overlap_args={},
                )
        event.record.assert_not_called()
        gemm.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
