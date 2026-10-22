"""Check pair32 capability selection and the GEMM/DeepEP signal contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe import hcu_fp8_sbo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHcuFp8Sbo(CustomTestCase):
    def test_pair32_packing_is_specific_to_gfx938(self):
        pack = Mock()
        with (
            patch.object(torch.cuda, "get_device_properties") as properties,
            patch.object(hcu_fp8_sbo, "import_module", return_value=SimpleNamespace(
                pack_fp8_weight_pair32=pack
            )) as imports,
        ):
            self.assertIsNone(hcu_fp8_sbo.get_hcu_fp8_pair32_packer(torch.device("cpu")))
            properties.return_value.gcnArchName = "gfx936"
            self.assertIsNone(hcu_fp8_sbo.get_hcu_fp8_pair32_packer(torch.device("cuda:0")))
            imports.assert_not_called()
            properties.return_value.gcnArchName = "gfx938:sramecc+"
            self.assertIs(hcu_fp8_sbo.get_hcu_fp8_pair32_packer(torch.device("cuda:0")), pack)

    def test_completion_metadata_comes_from_selected_kernel(self):
        out, signal, counts = object(), object(), object()
        for block_m in (64, 128):
            with self.subTest(block_m=block_m):
                calls = []
                gemm = Mock(side_effect=lambda *a, **kw: (calls.append("gemm") or (block_m, 24)))
                args = SimpleNamespace(
                    signal=signal,
                    start_event=SimpleNamespace(record=lambda: calls.append("event")),
                )
                meta = {}
                with patch.object(hcu_fp8_sbo, "import_module", return_value=SimpleNamespace(
                    m_grouped_fp8_gemm_nt_masked_pair32=gemm
                )):
                    result = hcu_fp8_sbo.run_hcu_fp8_pair32(
                        (), (), out, counts, 7, overlap_args=args, meta_overlap_args=meta
                    )
                self.assertIs(result, out)
                self.assertEqual(calls, ["event", "gemm"])
                self.assertEqual(meta, {"block_m": block_m, "threshold": 24})
                self.assertIs(gemm.call_args.kwargs["signal"], signal)
                self.assertTrue(gemm.call_args.kwargs["enable_overlap"])

    def test_no_overlap_passes_valid_tensor_placeholder(self):
        counts, out = object(), object()
        gemm = Mock(return_value=(64, 3))
        with patch.object(hcu_fp8_sbo, "import_module", return_value=SimpleNamespace(
            m_grouped_fp8_gemm_nt_masked_pair32=gemm
        )):
            hcu_fp8_sbo.run_hcu_fp8_pair32((), (), out, counts, 7)
            with self.assertRaisesRegex(ValueError, "shared DeepEP metadata"):
                hcu_fp8_sbo.run_hcu_fp8_pair32((), (), out, counts, 7, overlap_args=Mock())
        self.assertEqual(gemm.call_count, 1)
        self.assertIs(gemm.call_args.kwargs["signal"], counts)
        self.assertFalse(gemm.call_args.kwargs["enable_overlap"])


if __name__ == "__main__":
    unittest.main()
