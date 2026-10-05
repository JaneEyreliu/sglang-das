import math
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.attention.dsa_backend as dsa_backend
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DSAFlashMLAMetadata,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSAFlashMLAKVDCP(unittest.TestCase):
    def test_uses_hcu_flashmla_package(self):
        hcu_flashmla = ModuleType("flash_mla")

        with (
            patch.object(dsa_backend, "_is_hcu", True),
            patch.dict(sys.modules, {"flash_mla": hcu_flashmla}),
        ):
            self.assertIs(dsa_backend._get_flashmla_module(), hcu_flashmla)

    @patch("sglang.srt.layers.attention.dsa_backend.fixup_zero_kv_rows")
    @patch("sglang.srt.layers.attention.dsa_backend.get_flashmla_op")
    def test_returns_base2_lse_and_zero_row_metadata(self, mock_get_op, mock_fixup):
        mock_flashmla = mock_get_op.return_value
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.real_page_size = 64
        backend.kv_cache_dim = 4
        backend.dsa_kv_cache_store_fp8 = True
        backend.dsa_index_topk = 4
        backend.flashmla_kv_num_q_heads = 4
        backend.get_device_int32_arange = lambda length: torch.arange(
            length, dtype=torch.int32
        )

        batch_size = 2
        actual_heads = 2
        value_dim = 3
        q = torch.zeros(batch_size, actual_heads, 4, dtype=torch.bfloat16)
        kv = torch.zeros(2 * 64 * 4, dtype=torch.bfloat16)
        page_table = torch.tensor([[0, 1, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32)
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor([2, 0], dtype=torch.int32),
            flashmla_metadata=DSAFlashMLAMetadata(
                flashmla_metadata=torch.empty(0, dtype=torch.int32),
                num_splits=torch.empty(0, dtype=torch.int32),
            ),
        )
        layer = SimpleNamespace(tp_q_head_num=actual_heads, head_dim=4)

        mock_flashmla.return_value = (
            torch.ones(batch_size, 1, 4, value_dim, dtype=torch.bfloat16),
            torch.log(
                torch.tensor(
                    [
                        [[2.0], [8.0], [4.0], [16.0]],
                        [[4.0], [16.0], [2.0], [8.0]],
                    ]
                )
            ),
        )

        out, lse = backend._forward_flashmla_kv(
            q_all=q,
            kv_cache=kv,
            v_head_dim=value_dim,
            sm_scale=1.0,
            layer=layer,
            metadata=metadata,
            page_table_1=page_table,
            return_lse=True,
        )

        self.assertEqual(out.shape, (batch_size, actual_heads, value_dim))
        self.assertTrue(out.is_contiguous())
        torch.testing.assert_close(
            lse,
            torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertAlmostEqual(lse[0, 0].item(), math.log2(2.0))

        mock_fixup.assert_called_once()
        fixup_args = mock_fixup.call_args.args
        torch.testing.assert_close(
            fixup_args[2], torch.tensor([2, 0], dtype=torch.int32)
        )
        torch.testing.assert_close(
            fixup_args[3], torch.tensor([0, 1, 2], dtype=torch.int32)
        )


if __name__ == "__main__":
    unittest.main()
