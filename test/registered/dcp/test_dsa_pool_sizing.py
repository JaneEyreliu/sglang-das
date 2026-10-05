import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.pool_configurator import (
    DefaultPoolConfigurator,
    _get_dsa_indexer_cache_token_multiplier,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _kvc():
    return SimpleNamespace(server_args=SimpleNamespace())


class TestDSAPoolSizing(unittest.TestCase):
    def test_indexer_cache_is_replicated_across_dcp_ranks(self):
        with (
            get_parallel().override(dcp_enabled=True, attn_dcp_size=8),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
        ):
            self.assertEqual(_get_dsa_indexer_cache_token_multiplier(_kvc()), 8)

    def test_indexer_cache_has_no_multiplier_without_dcp(self):
        with (
            get_parallel().override(dcp_enabled=False, attn_dcp_size=1),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
        ):
            self.assertEqual(_get_dsa_indexer_cache_token_multiplier(_kvc()), 1)

    def test_cell_size_accounts_for_dcp_indexer_replication(self):
        kvc = SimpleNamespace(
            use_mla_backend=True,
            kv_cache_dtype=torch.float8_e4m3fn,
            kv_cache_dtype_str="fp8_e4m3",
            model_config=SimpleNamespace(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                hf_config=SimpleNamespace(),
                hf_text_config=SimpleNamespace(),
            ),
            server_args=SimpleNamespace(enable_dsa_cache_layer_split=False),
            is_draft_worker=False,
            page_size=64,
            layer_info=SimpleNamespace(start_layer=0, end_layer=2),
        )
        configurator = object.__new__(DefaultPoolConfigurator)

        with (
            get_parallel().override(attn_tp_size=8, dcp_enabled=True, attn_dcp_size=8),
            patch(
                "sglang.srt.model_executor.pool_configurator._is_hcu",
                False,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator."
                "is_hcu_native_fp8_supported",
                return_value=True,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
            patch(
                "sglang.srt.layers.cp.utils."
                "get_glm_dsa_layer_split_effective_num_layers",
                return_value=2,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.is_deepseek_dsa",
                return_value=True,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_dsa_index_head_dim",
                return_value=128,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator."
                "_should_elide_dsa_index_k",
                return_value=False,
            ),
        ):
            cell_size = configurator._compute_cell_size(kvc, num_layers=2)

        # Per layer: 576 bytes MLA KV + 132 bytes replicated across eight ranks.
        self.assertEqual(cell_size, (576 + 132 * 8) * 2)

    def test_hcu_bf16_indexer_fallback_keeps_dcp_replication(self):
        kvc = SimpleNamespace(
            use_mla_backend=True,
            kv_cache_dtype=torch.bfloat16,
            kv_cache_dtype_str="bfloat16",
            model_config=SimpleNamespace(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                hf_config=SimpleNamespace(),
                hf_text_config=SimpleNamespace(),
            ),
            server_args=SimpleNamespace(enable_dsa_cache_layer_split=False),
            is_draft_worker=False,
            page_size=64,
            layer_info=SimpleNamespace(start_layer=0, end_layer=2),
        )
        configurator = object.__new__(DefaultPoolConfigurator)

        with (
            get_parallel().override(attn_tp_size=8, dcp_enabled=True, attn_dcp_size=8),
            patch(
                "sglang.srt.model_executor.pool_configurator._is_hcu",
                True,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator."
                "is_hcu_native_fp8_supported",
                return_value=False,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
            patch(
                "sglang.srt.layers.cp.utils."
                "get_glm_dsa_layer_split_effective_num_layers",
                return_value=2,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.is_deepseek_dsa",
                return_value=True,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_dsa_index_head_dim",
                return_value=128,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator."
                "_should_elide_dsa_index_k",
                return_value=False,
            ),
        ):
            cell_size = configurator._compute_cell_size(kvc, num_layers=2)

        # Per layer: 576 bf16 MLA values + 128 bf16 index values on 8 ranks.
        self.assertEqual(cell_size, (576 * 2 + 128 * 2 * 8) * 2)


if __name__ == "__main__":
    unittest.main()
