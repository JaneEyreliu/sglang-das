"""Unit tests for the HCU HIPC W4A8 SBO completion-signal contract."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.layers.moe import hcu_w4a8_sbo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHcuW4A8Sbo(CustomTestCase):
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
        args = SimpleNamespace(signal=object(), start_event=SimpleNamespace(record=lambda: None))
        with patch.object(hcu_w4a8_sbo, "import_module", return_value=module):
            with self.assertRaisesRegex(ValueError, "Invalid W4A8 HIPC completion metadata"):
                hcu_w4a8_sbo.run_hcu_w4a8_hipc(
                    (), (), object(), object(), 12, overlap_args=args, meta_overlap_args={}
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


if __name__ == "__main__":
    import unittest

    unittest.main()
