"""CPU regressions for metadata returned by the deployed HY4 prefill peer."""

import dataclasses
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from test_hyv4_dcp_cpu import definitions

ns = definitions(
    "disaggregation/common/conn.py",
    ["PrefillServerInfo", "validate_pd_dcp_prefill_topology"],
    {"dataclasses": dataclasses, "__name__": __name__},
)
PrefillServerInfo = ns["PrefillServerInfo"]

# Captured from the live prefill bootstrap /route endpoint on 2026-09-23.
PREFILL_METADATA = dict(
    attn_tp_size=1,
    attn_cp_size=8,
    dp_size=1,
    pp_size=2,
    page_size=64,
    kv_cache_dtype="fp8_e4m3",
    follow_bootstrap_room=True,
    enable_dsa_cache_layer_split=True,
    kv_cache_layout=None,
    prefill_http_port=30000,
    target_tp_rank=None,
    target_tp_ranks=None,
    target_cp_ranks=None,
    target_pp_ranks=None,
    required_dst_info_num=None,
    required_prefill_response_num=None,
)


class TestHYV4PDBootstrap(unittest.TestCase):
    def test_live_metadata_and_legacy_missing_field_both_parse(self):
        current = PrefillServerInfo(**PREFILL_METADATA)
        old = {k: v for k, v in PREFILL_METADATA.items() if k != "kv_cache_layout"}
        self.assertEqual(current, PrefillServerInfo(**old))
        self.assertEqual(current.pp_size, 2)
        self.assertEqual(current.attn_cp_size, 8)
        self.assertTrue(current.enable_dsa_cache_layer_split)

    def test_target_supported_layouts_remain_accepted(self):
        for layout in ("hnd", "nhd", "vectorized_5d", "page_major_layer_major"):
            with self.subTest(layout=layout):
                info = PrefillServerInfo(**{**PREFILL_METADATA, "kv_cache_layout": layout})
                self.assertEqual(info.kv_cache_layout, layout)

    def test_named_layout_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "Unsupported prefill KV cache layout"):
            PrefillServerInfo(**{**PREFILL_METADATA, "kv_cache_layout": "other-layout"})

    def test_bootstrap_fetch_caches_metadata_and_keeps_compatibility_checks(self):
        get = Mock()
        logger = NS(error=Mock(), debug=Mock())
        fetch = definitions(
            "disaggregation/common/conn.py",
            ["try_ensure_parallel_info"],
            {**ns, "requests": NS(get=get), "logger": logger},
            cls="CommonKVManager",
        )["try_ensure_parallel_info"]
        for changes, error in (
            ({}, None),
            ({"page_size": 128}, "Page size mismatch"),
            ({"kv_cache_dtype": "bfloat16"}, "KV cache dtype mismatch"),
            ({"enable_dsa_cache_layer_split": False}, "ALL_CP_RANKS_TRANSFER"),
        ):
            with self.subTest(changes=changes):
                payload = {**PREFILL_METADATA, **changes}
                get.return_value = NS(status_code=200, json=lambda: payload)
                manager = NS(
                    prefill_info_table={},
                    kv_args=NS(page_size=64),
                    kv_cache_dtype_str="fp8_e4m3",
                    dcp_size=2,
                    is_mla_backend=True,
                    is_hybrid_mla_backend=False,
                    _resolve_rank_mapping=Mock(),
                )
                if error:
                    with self.assertRaisesRegex(RuntimeError, error):
                        fetch(manager, "prefill:8998")
                    self.assertFalse(manager.prefill_info_table)
                else:
                    self.assertTrue(fetch(manager, "prefill:8998"))
                    info = manager.prefill_info_table["prefill:8998"]
                    manager._resolve_rank_mapping.assert_called_once_with(info)
                    self.assertEqual(info.attn_cp_size, 8)
                    get.reset_mock()
                    self.assertTrue(fetch(manager, "prefill:8998"))
                    get.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
