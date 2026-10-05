import unittest

import torch

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode_ref,
    transform_index_page_table_prefill_ref,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _expected_owner_mapping(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    selected = torch.gather(page_table, 1, topk_indices.clamp(min=0))
    owned = (topk_indices >= 0) & (selected.remainder(dcp_size) == dcp_rank)
    return torch.where(owned, selected // dcp_size, -1).to(torch.int32)


class TestDSATransformIndexDCP(unittest.TestCase):
    def test_decode_filters_owner_and_maps_global_slots(self):
        page_table = torch.tensor(
            [
                [11, 4, 19, 8, 0, 15, 2, 21, 6],
                [32, 25, 18, 11, 4, 29, 22, 15, 8],
            ],
            dtype=torch.int32,
        )
        topk_indices = torch.tensor(
            [[0, 1, 2, 3, 4, 5, -1], [8, 7, 6, 5, 4, -1, -1]],
            dtype=torch.int64,
        )

        for dcp_size in (2, 4, 8):
            for dcp_rank in range(dcp_size):
                actual = transform_index_page_table_decode_ref(
                    page_table,
                    topk_indices,
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                )
                expected = _expected_owner_mapping(
                    page_table, topk_indices, dcp_size, dcp_rank
                )
                torch.testing.assert_close(actual, expected)

    def test_prefill_preserves_empty_local_kv_rows(self):
        page_table = torch.tensor([[0, 8, 16, 24], [1, 9, 17, 25]], dtype=torch.int32)
        topk_indices = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int64)

        actual = transform_index_page_table_prefill_ref(
            page_table,
            topk_indices,
            extend_lens_cpu=[1, 1],
            dcp_size=8,
            dcp_rank=7,
        )

        torch.testing.assert_close(actual, torch.full_like(actual, -1))


if __name__ == "__main__":
    unittest.main()
