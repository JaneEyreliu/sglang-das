import unittest

from sglang.srt.layers.attention.dsa.hcu_int8_index_k_cache import IndexKCacheMode
from sglang.srt.layers.attention.dsa_backend import _validate_dsa_dcp_launch
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _valid_config(**overrides):
    config = {
        "dcp_enabled": True,
        "dcp_size": 8,
        "is_hcu_platform": False,
        "device_capability": (9, 0),
        "dsa_prefill_impl": "flashmla_kv",
        "dsa_decode_impl": "flashmla_kv",
        "dsa_kv_cache_store_fp8": True,
        "page_size": 64,
        "enable_prefill_cp": False,
        "enable_hisparse": False,
        "enable_hierarchical_cache": False,
        "enable_symm_mem": False,
        "speculative_algorithm": None,
        "fused_topk_enabled": False,
        "dcp_comm_backend": "ag_rs",
        "speculative_num_steps": None,
        "speculative_eagle_topk": None,
        "speculative_num_draft_tokens": None,
        "pp_size": 1,
        "attn_cp_size": 1,
        "enable_dp_attention": True,
        "attn_tp_size": 8,
        "dcp_group_ranks": tuple(range(8)),
        "attn_tp_group_ranks": tuple(range(8)),
        "index_share_for_mtp_iteration": False,
        "decode_cuda_graph_backend": "disabled",
        "decode_cuda_graph_max_bs": 8,
    }
    config.update(overrides)
    return config


class TestDSADCPLaunchContract(unittest.TestCase):
    def test_transfer_abi_uses_physical_index_page_size(self):
        pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
        pool.index_k_cache_mode = IndexKCacheMode.BF16
        pool.index_page_size = 64
        pool.page_size = 512
        pool.index_head_dim = 128

        self.assertEqual(
            pool.get_index_k_cache_transfer_abi(),
            "dsa-index-k-page-v1:bf16:page_size=64:head_dim=128",
        )

    def test_supported_hopper_contract(self):
        _validate_dsa_dcp_launch(**_valid_config())

    def test_supported_bw1000_contract(self):
        _validate_dsa_dcp_launch(
            **_valid_config(is_hcu_platform=True, device_capability=(9, 3))
        )

    def test_dcp_disabled_is_noop(self):
        _validate_dsa_dcp_launch(
            **_valid_config(
                dcp_enabled=False,
                dcp_size=1,
                device_capability=(8, 0),
                dsa_prefill_impl="trtllm",
                dsa_decode_impl="trtllm",
                dsa_kv_cache_store_fp8=False,
                page_size=1,
                enable_prefill_cp=True,
                enable_hisparse=True,
                enable_hierarchical_cache=True,
                enable_symm_mem=True,
                speculative_algorithm="EAGLE",
                fused_topk_enabled=True,
                dcp_comm_backend="a2a",
            )
        )

    def test_rejects_each_out_of_scope_combination(self):
        cases = {
            "dcp size": {"dcp_size": 3},
            "Hopper SM90": {"device_capability": (10, 0)},
            "flashmla_kv": {"dsa_prefill_impl": "trtllm"},
            "FP8 KV": {"dsa_kv_cache_store_fp8": False},
            "page size 64": {"page_size": 1},
            "prefill CP": {"enable_prefill_cp": True},
            "HiSparse": {"enable_hisparse": True},
            "HiCache": {"enable_hierarchical_cache": True},
            "symmetric memory": {"enable_symm_mem": True},
            "fused DSA top-k": {"fused_topk_enabled": True},
            "ag_rs": {"dcp_comm_backend": "a2a"},
        }
        for expected, overrides in cases.items():
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    _validate_dsa_dcp_launch(**_valid_config(**overrides))

    def test_rejects_capability_platform_mismatch(self):
        cases = (
            _valid_config(is_hcu_platform=True, device_capability=(9, 0)),
            _valid_config(is_hcu_platform=False, device_capability=(9, 3)),
        )
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "Hopper SM90 or HCU BW1000"):
                    _validate_dsa_dcp_launch(**config)

    def test_supports_configurable_hcu_eagle_contract(self):
        cases = (
            {},
            {
                "speculative_num_steps": 2,
                "speculative_eagle_topk": 2,
                "speculative_num_draft_tokens": 7,
                "decode_cuda_graph_max_bs": 64,
            },
            {
                "dcp_size": 4,
                "attn_tp_size": 4,
                "dcp_group_ranks": tuple(range(4)),
                "attn_tp_group_ranks": tuple(range(4)),
            },
        )
        base = {
            "is_hcu_platform": True,
            "device_capability": (9, 3),
            "speculative_algorithm": "EAGLE",
            "speculative_num_steps": 4,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 5,
            "decode_cuda_graph_backend": "full",
            "decode_cuda_graph_max_bs": 32,
        }
        for overrides in cases:
            with self.subTest(overrides=overrides):
                _validate_dsa_dcp_launch(
                    **_valid_config(**{**base, **overrides})
                )

    def test_rejects_out_of_scope_eagle_combinations(self):
        cases = {
            "only supported on HCU": {
                "is_hcu_platform": False,
                "device_capability": (9, 0),
            },
            "only supports EAGLE": {"speculative_algorithm": "EAGLE3"},
            "positive speculative parameters": {"speculative_num_steps": 0},
            "attn_tp_size == dcp_size": {"attn_tp_size": 4},
            "identical DCP and attention TP group ranks": {
                "attn_tp_group_ranks": tuple(range(8, 16))
            },
            "PP1, attnCP1, and DP attention": {"pp_size": 2},
            "index_share_for_mtp_iteration": {
                "index_share_for_mtp_iteration": True
            },
            "disabled or full decode CUDA Graph": {
                "decode_cuda_graph_backend": "breakable"
            },
            "positive max BS": {"decode_cuda_graph_max_bs": 0},
        }
        base = {
            "is_hcu_platform": True,
            "device_capability": (9, 3),
            "speculative_algorithm": "EAGLE",
            "speculative_num_steps": 4,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 5,
            "decode_cuda_graph_backend": "full",
        }
        for expected, overrides in cases.items():
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    _validate_dsa_dcp_launch(
                        **_valid_config(**{**base, **overrides})
                    )


if __name__ == "__main__":
    unittest.main()
