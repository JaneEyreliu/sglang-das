import unittest

import torch

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode_fast,
    transform_index_page_table_decode_ref,
    transform_index_page_table_prefill_fast,
    transform_index_page_table_prefill_ref,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestDSATransformIndexDCPCUDA(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator(device="cuda").manual_seed(20260826)
        self.page_table = torch.stack(
            [
                torch.randperm(4096, generator=generator, device="cuda"),
                torch.randperm(4096, generator=generator, device="cuda") + 4096,
            ]
        ).to(torch.int32)
        self.decode_topk = torch.randint(
            0, 4096, (2, 2048), generator=generator, device="cuda"
        ).to(torch.int32)
        self.decode_topk[:, -17:] = -1

    def test_decode_matches_reference_for_dcp_2_4_8(self):
        for dcp_size in (2, 4, 8):
            for dcp_rank in range(dcp_size):
                actual = transform_index_page_table_decode_fast(
                    self.page_table,
                    self.decode_topk,
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                )
                expected = transform_index_page_table_decode_ref(
                    self.page_table,
                    self.decode_topk,
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                )
                torch.testing.assert_close(actual, expected)

    def test_prefill_matches_reference_for_dcp_2_4_8(self):
        topk = torch.cat((self.decode_topk, self.decode_topk[:1]), dim=0)
        for dcp_size in (2, 4, 8):
            for dcp_rank in range(dcp_size):
                actual = transform_index_page_table_prefill_fast(
                    self.page_table,
                    topk,
                    extend_lens_cpu=[2, 1],
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                )
                expected = transform_index_page_table_prefill_ref(
                    self.page_table,
                    topk,
                    extend_lens_cpu=[2, 1],
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                )
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
