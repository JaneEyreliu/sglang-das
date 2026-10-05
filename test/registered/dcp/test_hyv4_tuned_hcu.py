"""Numerical and graph regressions for the optional HY4 gfx936 kernels."""

import math
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.hyv4_tuned_gate import prepare_model_gates, try_tuned_gate
from sglang.srt.layers.hyv4_tuned_hadamard import hadamard128
from sglang.srt.layers.hyv4_tuned_mtp import prepare_mtp_input, try_tuned_mtp_input
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=60, suite="nightly-hcu-core-functional", nightly=True)


def _is_gfx936():
    return (
        torch.cuda.is_available()
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").startswith(
            "gfx936"
        )
    )


@unittest.skipUnless(_is_gfx936(), "HY4 tuning targets gfx936")
class TestHYV4TunedKernels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261005)

    def assert_graph_matches_eager(self, fn):
        expected = fn().clone()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_gate_preserves_parameter_and_matches_linear_and_graph(self):
        model = nn.Module()
        model.self_attn = nn.Module()
        layer = nn.Linear(6144, 8192, bias=False, device="cuda", dtype=torch.bfloat16)
        model.self_attn.linear_gate = layer
        model.requires_grad_(False)
        weight = layer.weight
        self.assertEqual(prepare_model_gates(model, "triton"), 1)
        self.assertIs(layer.weight, weight)
        self.assertEqual(weight.stride(), (1, 8192))
        for rows in (1, 4, 8, 3):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
                reference = F.linear(x.float(), weight.float()).to(x.dtype)
                actual = try_tuned_gate(layer, x)
                torch.testing.assert_close(actual, reference, rtol=1e-2, atol=5e-3)
                self.assert_graph_matches_eager(lambda: try_tuned_gate(layer, x))

    @torch.inference_mode()
    def test_mtp_projection_matches_linear_and_graph(self):
        model = nn.Module()
        model.eh_proj = nn.Linear(
            12288, 6144, bias=False, device="cuda", dtype=torch.bfloat16
        )
        model.requires_grad_(False)
        layer = model.eh_proj
        weight = layer.weight
        self.assertEqual(prepare_mtp_input(model, enabled=True), 1)
        self.assertIs(layer.weight, weight)
        for rows in (1, 3):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 12288, device="cuda", dtype=torch.bfloat16)
                reference = F.linear(x.float(), weight.float()).to(x.dtype)
                actual = try_tuned_mtp_input(layer, x)
                torch.testing.assert_close(actual, reference, rtol=1e-2, atol=5e-3)
                self.assert_graph_matches_eager(lambda: try_tuned_mtp_input(layer, x))

    @torch.inference_mode()
    def test_hadamard_matches_dense_reference_and_graph(self):
        matrix = torch.ones(1, 1, device="cuda", dtype=torch.bfloat16)
        for _ in range(7):
            matrix = torch.cat(
                (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
            )
        for rows in (0, 1, 5, 128):
            for scale in (1.0, 1 / math.sqrt(128)):
                with self.subTest(rows=rows, scale=scale):
                    x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)[:, ::2]
                    reference = (x.float() @ matrix.float()).to(x.dtype) * scale
                    actual = hadamard128(x, scale=scale)
                    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=5e-3)
                    self.assert_graph_matches_eager(lambda: hadamard128(x, scale=scale))


if __name__ == "__main__":
    unittest.main()
