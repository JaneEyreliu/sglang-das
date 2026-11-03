# SPDX-License-Identifier: Apache-2.0
"""HCU FP8 V expansion: head layout, scaling, output slices and graph replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.test_utils import CustomTestCase

register_hcu_ci(est_time=60, suite="stage-b-test-1-hcu-small")


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "requires HCU")
class TestHipblasltFp8Bmm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx938"):
            raise unittest.SkipTest("requires native FP8 HCU gfx938")
        from sglang.kernels.ops.gemm import hipblaslt_fp8_bmm, load_hipblaslt_fp8_bmm

        load_hipblaslt_fp8_bmm()
        cls.bmm = staticmethod(hipblaslt_fp8_bmm)

    @staticmethod
    def quantize(x, dtype=torch.float8_e4m3fn):
        scale = (
            x.float().abs().amax().clamp_min(1e-12).reshape(1) / torch.finfo(dtype).max
        )
        return (
            (x.float() / scale)
            .clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max)
            .to(dtype),
            scale,
        )

    def test_layout_accuracy_and_output_slice(self):
        torch.manual_seed(19)
        for tokens, heads, value in (
            (1, 64, 256),
            (3, 64, 256),
            (5, 16, 128),
            (129, 2, 256),
        ):
            with self.subTest(tokens=tokens, heads=heads, value=value):
                x = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(heads, value, 512, device="cuda", dtype=torch.bfloat16)
                q, xs = self.quantize(x)
                qw, ws = self.quantize(w)
                out = torch.full(
                    (tokens + 2, heads, value), 123, device="cuda", dtype=x.dtype
                )
                self.bmm(q, qw, xs, ws, out[:tokens])
                reference = torch.einsum(
                    "thk,hvk->thv", q.float() * xs, qw.float() * ws
                )
                torch.testing.assert_close(
                    out[:tokens], reference.to(x.dtype), atol=0.125, rtol=0.02
                )
                bf16_reference = torch.einsum("thk,hvk->thv", x.float(), w.float())
                relative_rmse = (
                    out[:tokens].float() - bf16_reference
                ).square().mean().sqrt() / bf16_reference.square().mean().sqrt()
                self.assertLess(relative_rmse.item(), 0.06)
                self.assertTrue(torch.all(out[tokens:] == 123).item())

    def test_graph_replay_tracks_current_scales_and_inputs(self):
        q, xs = self.quantize(
            torch.randn(5, 16, 512, device="cuda", dtype=torch.bfloat16)
        )
        qw, ws = self.quantize(
            torch.randn(16, 256, 512, device="cuda", dtype=torch.bfloat16)
        )
        out = torch.empty(5, 16, 256, device="cuda", dtype=torch.bfloat16)
        self.bmm(q, qw, xs, ws, out)
        # Cache hit must refresh pointers, including scales with new allocations.
        xs, ws = xs.clone() * 0.5, ws.clone() * 1.5
        self.bmm(q, qw, xs, ws, out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.bmm(q, qw, xs, ws, out)
        for multiplier in (0.75, 1.25):
            xs.mul_(multiplier)
            q.copy_(torch.randn_like(q.float()).to(q.dtype))
            graph.replay()
            expected = torch.einsum("thk,hvk->thv", q.float() * xs, qw.float() * ws)
            torch.testing.assert_close(
                out, expected.to(out.dtype), atol=0.125, rtol=0.02
            )

    def test_validation_and_empty_batch(self):
        q = torch.empty(0, 2, 512, device="cuda", dtype=torch.float8_e4m3fn)
        w = torch.empty(2, 128, 512, device="cuda", dtype=q.dtype)
        s = torch.ones(1, device="cuda", dtype=torch.float32)
        out = torch.empty(0, 2, 128, device="cuda", dtype=torch.bfloat16)
        self.bmm(q, w, s, s, out)
        with self.assertRaisesRegex(RuntimeError, "scalar FP32"):
            self.bmm(q, w, s.expand(2), s, out)
        with self.assertRaisesRegex(RuntimeError, "BF16 output"):
            self.bmm(q, w, s, s, out.float())

    def test_sglang_projection_quantization_and_graph(self):
        from sglang.srt.environ import envs
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
            prepare_hy4_fp8_v_up_proj,
            rocm_absorb_v_bmm,
        )

        # A disabled switch must not compile or require any weight attributes.
        with envs.SGLANG_HY4_FP8_V_UP_PROJ.override(False), patch(
            "sglang.kernels.ops.gemm.load_hipblaslt_fp8_bmm"
        ) as loader:
            disabled = SimpleNamespace()
            prepare_hy4_fp8_v_up_proj(disabled)
            self.assertFalse(disabled.use_hipblaslt_fp8_v_up_proj)
            loader.assert_not_called()

        weight = torch.randn(16, 512, 256, device="cuda", dtype=torch.bfloat16)
        attn = SimpleNamespace(
            w_vc=weight,
            w_scale=0.75,
            use_deep_gemm_bmm=False,
            num_local_heads=16,
            v_head_dim=256,
        )
        with envs.SGLANG_HY4_FP8_V_UP_PROJ.override(True):
            prepare_hy4_fp8_v_up_proj(attn)
        self.assertIs(attn.w_vc, weight)
        self.assertEqual(attn.w_vc_fp8.shape, (16, 256, 512))
        x = torch.randn(5, 16, 512, device="cuda", dtype=torch.bfloat16)
        # Include a noncontiguous head layout produced by attention reduction.
        for input_x in (x, x.transpose(0, 1).contiguous().transpose(0, 1)):
            result = rocm_absorb_v_bmm(attn, input_x)
            reference = torch.einsum(
                "thk,hkv->thv", input_x.float(), (weight * 0.75).float()
            ).flatten(1, 2)
            relative_rmse = (
                result.float() - reference
            ).square().mean().sqrt() / reference.square().mean().sqrt()
            self.assertLess(relative_rmse.item(), 0.06)
        # This graph includes dynamic input quantization, not just the GEMM.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = rocm_absorb_v_bmm(attn, x)
        for magnitude in (0.25, 2.0):
            x.copy_(torch.randn_like(x) * magnitude)
            graph.replay()
            eager = rocm_absorb_v_bmm(attn, x)
            torch.testing.assert_close(result, eager, atol=0, rtol=0)
        zeros = rocm_absorb_v_bmm(attn, torch.zeros_like(x))
        self.assertTrue(torch.all(zeros == 0).item())
        self.assertEqual(rocm_absorb_v_bmm(attn, x[:0]).shape, (0, 4096))


if __name__ == "__main__":
    unittest.main()
