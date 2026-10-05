"""LightOp selection must preserve logical IDs for DCP and MTP sharing."""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode_fast,
)
from sglang.srt.layers.attention.dsa import dsa_topk_backend as backend
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.utils import is_hcu
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=30, suite="nightly-hcu-core-functional", nightly=True)


@unittest.skipUnless(is_hcu(), "requires a wave64 HCU GPU and LightOp")
class TestDSALightOpRawTopK(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "SGLANG_DSA_HCU_LIGHTOP_TOPK": "1",
                "SGLANG_DSA_FUSE_TOPK": "false",
                "SGL_USE_LIGHTOP_TOPK_BACKAND": "0",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def inputs(self, n=8192):
        # Noncontiguous rows and nonzero starts exercise the indexer's views.
        scores = torch.empty((6, n + 32), device="cuda")[:, : n + 8]
        values = torch.randperm(n, device="cuda").float() - n // 2
        starts = torch.tensor([0, 1, 3, 7, 0, 2], dtype=torch.int32, device="cuda")
        lengths = torch.tensor(
            [0, 1, 2047, 2048, 2049, n], dtype=torch.int32, device="cuda"
        )
        scores.fill_(float("inf"))  # Outside each valid window must be ignored.
        for row, start in enumerate(starts.tolist()):
            scores[row, start : start + n] = values
        return scores, lengths, starts

    def select(self, scores, lengths, starts=None):
        # Exercise the production unfused dispatch, with no page table at all.
        return DSATopKBackend.SGL_KERNEL.topk_transform(
            scores,
            lengths,
            2048,
            TopkTransformMethod.PAGED,
            None,
            row_starts=starts,
        )

    def reference(self, scores, lengths, starts=None):
        result = torch.full((scores.shape[0], 2048), -1, dtype=torch.int32)
        for row, n in enumerate(lengths.tolist()):
            start = 0 if starts is None else int(starts[row])
            # Adaptive TopK breaks ties by smaller logical ID; order is unspecified.
            ids = torch.argsort(
                scores[row, start : start + n].cpu(), descending=True, stable=True
            )[:2048].sort().values
            result[row, : ids.numel()] = ids.int()
        return result

    def assert_same_indices(self, actual, expected):
        # Keep multiplicities and padding in the comparison; sorting here does
        # not impose an ordering requirement on the production output.
        self.assertTrue(
            torch.equal(
                actual.cpu().sort(dim=-1).values,
                expected.cpu().sort(dim=-1).values,
            )
        )

    def test_windows_padding_stride_and_input_unchanged(self):
        for n in (8192, 131072):
            with self.subTest(n=n):
                scores, lengths, starts = self.inputs(n)
                original = scores.clone()
                actual = self.select(scores, lengths, starts)
                self.assert_same_indices(
                    actual, self.reference(scores, lengths, starts)
                )
                self.assertTrue(torch.equal(scores, original))
                self.assertEqual(actual.dtype, torch.int32)

    def test_matches_original_kernel_with_separated_cutoff(self):
        from sgl_kernel import fast_topk_v2

        # Compare the original implementation on an unambiguous cutoff. The
        # broad-range FP32 and boundary-tie cases use the exact reference above.
        scores = torch.zeros((3, 131072), device="cuda")
        selected = torch.randperm(131072, device="cuda")[:2048]
        scores[:, selected] = 1 + torch.arange(2048, device="cuda") / 2048
        lengths = torch.full((3,), 131072, dtype=torch.int32, device="cuda")
        original = fast_topk_v2(scores, lengths, 2048)
        actual = self.select(scores, lengths)
        self.assert_same_indices(actual, original)
        self.assert_same_indices(actual, selected.int()[None].expand(3, -1))

    def test_ties_and_masked_scores(self):
        scores = torch.zeros((3, 8192), device="cuda")
        scores[1, :3072] = float("-inf")
        scores[2] = -torch.arange(8192, device="cuda").float()
        lengths = torch.full((3,), 8192, dtype=torch.int32, device="cuda")
        self.assert_same_indices(
            self.select(scores, lengths), self.reference(scores, lengths)
        )

    def test_dcp_mapping_keeps_shared_logical_ids(self):
        scores, lengths, starts = self.inputs()
        raw = self.select(scores, lengths, starts)
        raw_before = raw.clone()
        self.assert_same_indices(raw, self.reference(scores, lengths, starts))
        expected_raw = raw_before
        pages = torch.randperm(8192, device="cuda").int()[None].expand(6, -1)
        global_slots = pages.gather(1, expected_raw.clamp_min(0).long())
        for size in (2, 4, 8):
            for rank in range(size):
                mapped = transform_index_page_table_decode_fast(
                    pages, raw, dcp_size=size, dcp_rank=rank
                )
                expected = torch.where(
                    (expected_raw >= 0) & (global_slots % size == rank),
                    global_slots // size,
                    -1,
                )
                self.assertTrue(torch.equal(mapped, expected))
        self.assertTrue(torch.equal(raw, raw_before))

    def test_graph_replay_uses_current_lengths_and_starts(self):
        scores, lengths, starts = self.inputs()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.select(scores, lengths, starts)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.select(scores, lengths, starts)
        for n in (0, 1, 2048, 8192):
            lengths.fill_(n)
            starts.zero_()
            scores.copy_(torch.arange(scores.shape[1], device="cuda").float())
            graph.replay()
            torch.cuda.synchronize()
            self.assert_same_indices(actual, self.reference(scores, lengths, starts))

    def test_opt_out_and_unsupported_layout_use_original(self):
        import sgl_kernel

        scores, lengths, starts = self.inputs()
        with patch.object(sgl_kernel, "fast_topk_v2", return_value="original") as old:
            with patch.dict(os.environ, {"SGLANG_DSA_HCU_LIGHTOP_TOPK": "0"}):
                self.assertEqual(self.select(scores, lengths, starts), "original")
            self.assertEqual(self.select(scores, lengths.long(), starts), "original")
            self.assertEqual(old.call_count, 2)

    def test_kernel_errors_are_not_silently_fallback(self):
        from lightop import op

        scores, lengths, starts = self.inputs()
        with patch.object(
            op,
            "fast_topk_transform_ragged_interface",
            side_effect=RuntimeError("probe"),
        ):
            with self.assertRaisesRegex(RuntimeError, "probe"):
                self.select(scores, lengths, starts)

    def test_profile_confirms_adaptive_kernel_without_memset(self):
        scores, lengths, starts = self.inputs()
        self.select(scores, lengths, starts)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            self.select(scores, lengths, starts)
            torch.cuda.synchronize()
        names = [event.name for event in prof.events()]
        self.assertTrue(
            any("adaptive_topk_kernel<2048" in name for name in names), names
        )
        self.assertFalse(any("memset" in name.lower() for name in names), names)


if __name__ == "__main__":
    unittest.main()
