import ast
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeHCUKVCacheOps:
    def __init__(self):
        self.calls = []

    def fused_quantize_and_store_mla_kv_cache(self, *args):
        self.calls.append(args)


class TestHCUMLAKVStoreDCP(unittest.TestCase):
    @staticmethod
    def _load_write_method(dcp_rank: int, dcp_enabled: bool):
        source_path = (
            Path(__file__).parents[3] / "python/sglang/srt/mem_cache/memory_pool.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        pool_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MLATokenToKVPool"
        )
        method = next(
            node
            for node in pool_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_write_mla_kv_buffer"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                method,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        namespace = {
            "torch": torch,
            "_is_hip": False,
            "_is_hcu": True,
            "fp8_dtype": torch.float8_e4m3fn,
            "get_parallel": lambda: SimpleNamespace(
                dcp_enabled=dcp_enabled,
                attn_dcp_size=8,
                attn_dcp_rank=dcp_rank,
            ),
            "quantize_k_cache_separate": None,
            "set_mla_kv_buffer_triton": None,
            "set_mla_kv_buffer_triton_fp8_quant": None,
        }
        exec(compile(module, source_path, "exec"), namespace)
        return namespace["_write_mla_kv_buffer"]

    def _run_store(self, dcp_rank: int, dcp_enabled: bool = True):
        ops = _FakeHCUKVCacheOps()
        lightop = types.ModuleType("lightop")
        lightop.kvcache = ops
        pool = SimpleNamespace(
            dsa_kv_cache_store_fp8=True,
            use_dsa=True,
            dtype=torch.float8_e4m3fn,
        )

        loc = torch.tensor([17, 18, 23, 24, 31, 32], dtype=torch.int64)
        cache_k_nope = torch.arange(6 * 4, dtype=torch.bfloat16).view(6, 4)
        cache_k_rope = torch.arange(6 * 2, dtype=torch.bfloat16).view(6, 2)
        dst_buffer = torch.empty(8, 1, 6, dtype=torch.uint8)

        write_mla_kv_buffer = self._load_write_method(dcp_rank, dcp_enabled)
        with patch.dict(sys.modules, {"lightop": lightop}):
            write_mla_kv_buffer(
                pool,
                dst_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
            )

        return SimpleNamespace(
            ops=ops,
            loc=loc,
            cache_k_nope=cache_k_nope,
            cache_k_rope=cache_k_rope,
        )

    def test_filters_owner_and_maps_virtual_loc_to_local_slot(self):
        result = self._run_store(dcp_rank=7)

        self.assertEqual(len(result.ops.calls), 1)
        cache_k_nope, cache_k_rope, _, loc, *_ = result.ops.calls[0]
        torch.testing.assert_close(loc, torch.tensor([2, 3], dtype=torch.int64))
        torch.testing.assert_close(cache_k_nope, result.cache_k_nope[[2, 4]])
        torch.testing.assert_close(cache_k_rope, result.cache_k_rope[[2, 4]])

    def test_skips_lightop_when_rank_owns_no_rows(self):
        result = self._run_store(dcp_rank=3)

        self.assertEqual(result.ops.calls, [])

    def test_dcp_disabled_preserves_original_lightop_inputs(self):
        result = self._run_store(dcp_rank=7, dcp_enabled=False)

        self.assertEqual(len(result.ops.calls), 1)
        cache_k_nope, cache_k_rope, _, loc, *_ = result.ops.calls[0]
        self.assertIs(loc, result.loc)
        self.assertIs(cache_k_nope, result.cache_k_nope)
        self.assertIs(cache_k_rope, result.cache_k_rope)


if __name__ == "__main__":
    unittest.main()
