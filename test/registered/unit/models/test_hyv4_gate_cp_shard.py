"""CPU-safe contracts for HYV4 linear-gate CP sharding."""

import unittest

import torch

from sglang.srt.layers.hyv4_gate_cp_shard import create_cp_sharded_gate_weight
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHYV4GateCPShard(CustomTestCase):
    def test_output_rows_are_sharded_contiguously(self):
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        self.assertTrue(
            torch.equal(
                create_cp_sharded_gate_weight(weight, cp_size=3, cp_rank=1),
                weight[2:4],
            )
        )
        self.assertTrue(
            create_cp_sharded_gate_weight(weight, 3, 1).is_contiguous()
        )

    def test_shape_and_rank_contracts(self):
        weight = torch.zeros(8, 4)
        with self.assertRaisesRegex(ValueError, "2-D"):
            create_cp_sharded_gate_weight(weight[None], 2, 0)
        with self.assertRaisesRegex(ValueError, "rank/size"):
            create_cp_sharded_gate_weight(weight, 2, 2)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            create_cp_sharded_gate_weight(weight[:7], 3, 0)
        with self.assertRaisesRegex(ValueError, "output size"):
            create_cp_sharded_gate_weight(weight, 2, 0, expected_output_size=16)
        with self.assertRaisesRegex(ValueError, "input size"):
            create_cp_sharded_gate_weight(weight, 2, 0, expected_input_size=8)

    def test_all_cp_shards_reconstruct_full_weight(self):
        weight = torch.randn(12, 5)
        shards = [create_cp_sharded_gate_weight(weight, 4, rank) for rank in range(4)]
        self.assertTrue(torch.equal(torch.cat(shards, dim=0), weight))


if __name__ == "__main__":
    unittest.main()
