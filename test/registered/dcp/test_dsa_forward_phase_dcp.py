import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    is_dcp_mla_decode_phase,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSAForwardPhaseDCP(unittest.TestCase):
    def test_dsa_extend_uses_q_gather_and_lse_merge_contract(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        with get_parallel().override(dcp_enabled=True):
            self.assertTrue(is_dcp_mla_decode_phase(forward_batch, use_dsa=True))

    def test_dense_mla_extend_keeps_existing_kv_gather_contract(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        with get_parallel().override(dcp_enabled=True):
            self.assertFalse(is_dcp_mla_decode_phase(forward_batch, use_dsa=False))

    def test_dsa_draft_extend_v2_uses_q_gather_and_lse_merge_contract(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)
        with get_parallel().override(dcp_enabled=True):
            self.assertTrue(is_dcp_mla_decode_phase(forward_batch, use_dsa=True))

    def test_dense_draft_extend_v2_keeps_existing_contract(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)
        with get_parallel().override(dcp_enabled=True):
            self.assertFalse(is_dcp_mla_decode_phase(forward_batch, use_dsa=False))

    def test_target_verify_contract_is_unchanged(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        with get_parallel().override(dcp_enabled=True):
            self.assertTrue(is_dcp_mla_decode_phase(forward_batch, use_dsa=True))

    def test_dcp_disabled_rejects_all_phases(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)
        with get_parallel().override(dcp_enabled=False):
            self.assertFalse(is_dcp_mla_decode_phase(forward_batch, use_dsa=True))

    def test_decode_contract_is_unchanged(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        with get_parallel().override(dcp_enabled=True):
            self.assertTrue(is_dcp_mla_decode_phase(forward_batch, use_dsa=False))


if __name__ == "__main__":
    unittest.main()
