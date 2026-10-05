import unittest

import torch

from sglang.srt.mem_cache.memory_pool import (
    _remap_hcu_dcp_graph_kv_write_locs,
    compute_hcu_dcp_graph_kv_padding_capacity,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHCUDCPCudaGraphKVWrite(unittest.TestCase):
    def test_non_owner_rows_are_redirected_to_distinct_padding_slots(self):
        loc = torch.arange(8, dtype=torch.int64)

        mapped = _remap_hcu_dcp_graph_kv_write_locs(
            loc,
            dcp_size=8,
            dcp_rank=3,
            pool_size=100,
            padding_capacity=192,
        )

        self.assertEqual(mapped[3].item(), 0)
        self.assertEqual(mapped[:3].tolist(), [100, 101, 102])
        self.assertEqual(mapped[4:].tolist(), [104, 105, 106, 107])
        self.assertEqual(mapped.unique().numel(), loc.numel())
        self.assertLess(mapped.max().item(), 100 + 192)

    def test_capture_rows_fit_configured_reserved_padding(self):
        for rows in (64, 65, 160, 192):
            with self.subTest(rows=rows):
                mapped = _remap_hcu_dcp_graph_kv_write_locs(
                    torch.arange(rows),
                    dcp_size=8,
                    dcp_rank=0,
                    pool_size=100,
                    padding_capacity=192,
                )
                self.assertLess(mapped.max().item(), 100 + 192)

    def test_capture_rows_must_fit_reserved_padding_capacity(self):
        with self.assertRaisesRegex(AssertionError, "not to exceed padding capacity"):
            _remap_hcu_dcp_graph_kv_write_locs(
                torch.arange(193),
                dcp_size=8,
                dcp_rank=0,
                pool_size=100,
                padding_capacity=192,
            )

    def test_padding_capacity_is_aligned_to_base_page(self):
        common = {
            "pool_page_size": 64,
            "page_size": 64,
            "dcp_enabled": True,
            "is_hcu_platform": True,
            "speculative_algorithm": "EAGLE",
            "decode_cuda_graph_backend": "full",
            "speculative_num_draft_tokens": 5,
        }
        expected = {8: 64, 13: 128, 32: 192, 64: 320}
        for max_bs, capacity in expected.items():
            with self.subTest(max_bs=max_bs):
                self.assertEqual(
                    compute_hcu_dcp_graph_kv_padding_capacity(
                        **common, decode_cuda_graph_max_bs=max_bs
                    ),
                    capacity,
                )

    def test_padding_capacity_tracks_configured_draft_tokens(self):
        self.assertEqual(
            compute_hcu_dcp_graph_kv_padding_capacity(
                pool_page_size=64,
                page_size=64,
                dcp_enabled=True,
                is_hcu_platform=True,
                speculative_algorithm="EAGLE",
                decode_cuda_graph_backend="full",
                decode_cuda_graph_max_bs=32,
                speculative_num_draft_tokens=9,
            ),
            320,
        )

    def test_replicated_draft_keeps_its_wider_padding_page(self):
        self.assertEqual(
            compute_hcu_dcp_graph_kv_padding_capacity(
                pool_page_size=512,
                page_size=64,
                dcp_enabled=True,
                is_hcu_platform=True,
                speculative_algorithm="EAGLE",
                decode_cuda_graph_backend="full",
                decode_cuda_graph_max_bs=32,
                speculative_num_draft_tokens=5,
            ),
            512,
        )


if __name__ == "__main__":
    unittest.main()
