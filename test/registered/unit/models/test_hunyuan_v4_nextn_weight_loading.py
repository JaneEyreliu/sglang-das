"""Regression tests for Hunyuan-V4 NEXTN weight loading configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from sglang.srt.models.hunyuan_v4_nextn import (
    _gather_dsa_topk_indices_for_cp,
    _mtp_quant_config,
    _restore_hyv4_nextn_cp_output,
    _split_hyv4_nextn_cp_inputs,
)


class TestHunyuanV4NextNQuantConfig(CustomTestCase):
    def test_prefill_cp_splits_embedding_and_target_hidden_states_together(self):
        forward_batch = object()
        embedding = torch.arange(12).view(6, 2)
        target = embedding + 100
        positions = torch.arange(6)
        calls = []

        def split_data(batch, value):
            calls.append(value)
            return value[::2]

        with patch(
            "sglang.srt.models.hunyuan_v4_nextn.cp_split_and_rebuild_data",
            side_effect=split_data,
        ) as split, patch(
            "sglang.srt.models.hunyuan_v4_nextn.cp_split_and_rebuild_position",
            side_effect=lambda batch, value: value[::2],
        ):
            got_embedding, got_target, got_positions = _split_hyv4_nextn_cp_inputs(
                forward_batch, embedding, target, positions
            )

        self.assertEqual(split.call_count, 2)
        self.assertIs(calls[0], embedding)
        self.assertIs(calls[1], target)
        self.assertTrue(torch.equal(got_embedding, embedding[::2]))
        self.assertTrue(torch.equal(got_target, target[::2]))
        self.assertTrue(torch.equal(got_positions, positions[::2]))

    def test_prefill_cp_output_and_dsa_topk_are_gathered(self):
        local_hidden = torch.tensor([[2.0], [4.0]])
        local_topk = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
        gathered_hidden = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        gathered_topk = torch.tensor(
            [[10, 11], [30, 31], [20, 21], [40, 41]], dtype=torch.int32
        )
        forward_batch = object()

        with patch(
            "sglang.srt.models.hunyuan_v4_nextn.cp_all_gather_rerange_output",
            side_effect=[gathered_hidden, gathered_topk],
        ) as gather, patch(
            "sglang.srt.models.hunyuan_v4_nextn.is_dsa_prefill_cp_round_robin_split",
            return_value=False,
        ), patch(
            "torch.cuda.current_stream", return_value=object()
        ):
            got_hidden = _restore_hyv4_nextn_cp_output(
                local_hidden, 2, forward_batch
            )
            got_topk = _gather_dsa_topk_indices_for_cp(
                local_topk, 2, 2, forward_batch, object()
            )

        self.assertIs(got_hidden, gathered_hidden)
        self.assertIs(got_topk, gathered_topk)
        self.assertEqual(gather.call_count, 2)

    def test_round_robin_topk_is_padded_before_gather(self):
        local_topk = torch.tensor([[10, 11]], dtype=torch.int32)
        forward_batch = object()
        with patch(
            "sglang.srt.models.hunyuan_v4_nextn.is_dsa_prefill_cp_round_robin_split",
            return_value=True,
        ), patch(
            "sglang.srt.models.hunyuan_v4_nextn.cp_all_gather_rerange_output",
            side_effect=lambda value, cp_size, batch, stream: value,
        ) as gather:
            got = _gather_dsa_topk_indices_for_cp(
                local_topk, 2, 2, forward_batch, object()
            )

        self.assertEqual(got.shape, (2, 2))
        self.assertTrue(torch.equal(got[1], torch.tensor([-1, -1])))
        self.assertEqual(gather.call_args.args[1], 2)

    def test_mtp_ignore_paths_follow_draft_decoder_without_broad_exclusion(self):
        fp8_config = object()
        quant_config = SimpleNamespace(
            ignored_layers=[
                "model.mtp.layers.0.self_attn.q_proj",
                "model.mtp_layers.0.self_attn.q_proj",
            ],
            ignore=["model.mtp.layers.0.self_attn.linear_gate"],
            linear_fp8_config=fp8_config,
        )

        mtp_config = _mtp_quant_config(quant_config)

        self.assertIsNot(mtp_config, quant_config)
        self.assertEqual(
            quant_config.ignored_layers,
            [
                "model.mtp.layers.0.self_attn.q_proj",
                "model.mtp_layers.0.self_attn.q_proj",
            ],
        )
        self.assertEqual(
            mtp_config.ignored_layers,
            ["model.decoder.self_attn.q_proj"],
        )
        self.assertEqual(
            mtp_config.ignore,
            ["model.decoder.self_attn.linear_gate"],
        )
        self.assertIs(quant_config.linear_fp8_config, fp8_config)
        self.assertIsNone(mtp_config.linear_fp8_config)
        self.assertNotIn("model.decoder", mtp_config.ignored_layers)
        self.assertNotIn("model.decoder", mtp_config.ignore)
        self.assertFalse(
            any(target.startswith("re:") for target in mtp_config.ignored_layers)
        )

        self.assertTrue(
            should_ignore_layer(
                "model.decoder.self_attn.q_proj", mtp_config.ignored_layers
            )
        )
        self.assertTrue(
            should_ignore_layer(
                "model.decoder.self_attn.linear_gate", mtp_config.ignore
            )
        )
        self.assertFalse(
            should_ignore_layer(
                "model.decoder.self_attn.q_a_proj", mtp_config.ignored_layers
            )
        )

    def test_mtp_fusion_and_norm_paths_match_root_level_modules(self):
        quant_config = SimpleNamespace(
            ignored_layers=[
                "model.mtp.layers.0.eh_proj",
                "model.mtp_layers.0.enorm",
                "mtp.layers.0.hnorm",
                "mtp_layers.0.shared_head.norm",
            ],
            ignore=[],
            linear_fp8_config=None,
        )

        mtp_config = _mtp_quant_config(quant_config)

        self.assertEqual(
            mtp_config.ignored_layers,
            [
                "model.eh_proj",
                "model.enorm",
                "model.hnorm",
                "model.shared_head.norm",
            ],
        )

    def test_none_quant_config_stays_none(self):
        self.assertIsNone(_mtp_quant_config(None))


if __name__ == "__main__":
    unittest.main()
