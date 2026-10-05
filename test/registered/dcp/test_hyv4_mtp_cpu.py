"""HY4 MTP CPU contracts; GPU kernels, collectives and P/D need deployment tests."""

import copy
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from test_hyv4_dcp_cpu import MODEL, ROOT, definitions, hy4
from torch import nn

NEXTN = "models/hunyuan_v4_nextn.py"
spec = importlib.util.spec_from_file_location("hy4_mtp", ROOT / "configs/hy_v4_mtp.py")
mtp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mtp)


def checkpoint_fixture():
    # Tiny but complete exported MTP, independent of production validation code.
    config = dict(
        hidden_size=8,
        num_attention_heads=2,
        v_head_dim=4,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        index_n_heads=2,
        index_head_dim=4,
        n_routed_experts=2,
        n_shared_experts=1,
        moe_intermediate_size=4,
        num_nextn_predict_layers=1,
        num_hidden_layers=3,
    )
    shapes = {
        "enorm.weight": [8],
        "hnorm.weight": [8],
        "eh_proj.weight": [8, 16],
        "shared_head.norm.weight": [8],
        "input_layernorm.weight": [8],
        "post_attention_layernorm.weight": [8],
        "self_attn.linear_gate.weight": [8, 8],
        "self_attn.learnable_sink_param": [2],
        "self_attn.q_a_proj.weight": [4, 8],
        "self_attn.q_a_layernorm.weight": [4],
        "self_attn.q_b_proj.weight": [8, 4],
        "self_attn.kv_a_proj_with_mqa.weight": [6, 8],
        "self_attn.kv_a_layernorm.weight": [4],
        "self_attn.kv_b_proj.weight": [12, 4],
        "self_attn.o_proj.weight": [8, 8],
        "self_attn.indexer.wq_b.weight": [8, 4],
        "self_attn.indexer.wk.weight": [4, 8],
        "self_attn.indexer.k_norm.weight": [4],
        "self_attn.indexer.k_norm.bias": [4],
        "self_attn.indexer.weights_proj.weight": [2, 8],
        "mlp.gate.weight": [2, 8],
        "mlp.gate.e_score_correction_bias": [2],
    }
    metadata = {name: dict(shape=shape, dtype="BF16") for name, shape in shapes.items()}
    for module in ("mlp.experts.0", "mlp.experts.1", "mlp.shared_experts"):
        for proj, shape in (
            ("gate_proj", [4, 4]),
            ("up_proj", [4, 4]),
            ("down_proj", [8, 2]),
        ):
            metadata[f"{module}.{proj}.weight"] = dict(shape=shape, dtype="I8")
            metadata[f"{module}.{proj}.weight_scale"] = dict(
                shape=[shape[0], 1], dtype="F32"
            )
    return config, metadata


class TestHYV4MTP(unittest.TestCase):
    def test_target_loader_leaves_mtp_to_draft(self):
        normalize = definitions(MODEL, ["normalize_hyv4_weight_name"])[
            "normalize_hyv4_weight_name"
        ]
        load = definitions(
            MODEL,
            ["load_weights"],
            {
                "normalize_hyv4_weight_name": normalize,
                "hyv4_linear_scale_suffix": lambda _: "",
                "permute_hyv4_indexer_weight": lambda name, weight, config: weight,
            },
            cls="HYV4ForCausalLM",
        )["load_weights"]
        loaded = []
        model = NS(
            config=None,
            quant_config=None,
            do_load_weights=lambda weights: loaded.extend(weights),
        )
        tensor = torch.ones(2, 2)
        load(
            model,
            [
                (prefix + "self_attn.g_proj.weight.weight", tensor)
                for prefix in (
                    "model.layers.0.",
                    "model.mtp_layers.0.",
                    "model.mtp.layers.0.",
                )
            ],
        )
        self.assertEqual(
            [name for name, _ in loaded],
            ["model.layers.0.self_attn.linear_gate.weight"],
        )

    def test_hidden_contract_preserves_other_models(self):
        ns = definitions(
            "configs/model_config.py",
            ["_hf_arch", "is_hy_v4", "resolve_spec_hidden_size"],
        )
        resolve = ns["resolve_spec_hidden_size"]
        for arch in ("HYV4ForCausalLM", "HYV4ForCausalLMNextN"):
            self.assertEqual(resolve(NS(architectures=[arch]), 2816, 4), (2816, None))
        for arch in (
            "GlmMoeDsaForCausalLM",
            "GlmMoeDsaForCausalLMNextN",
            "DeepseekV4ForCausalLM",
            "Qwen4ExpForCausalLM",
        ):
            for hc in (1, 4):
                self.assertEqual(
                    resolve(NS(architectures=[arch]), 8, hc),
                    (8, None) if hc == 1 else (32, 32),
                )

    def test_draft_registration_and_dsa_preserve_glm(self):
        ns = definitions(
            "configs/model_config.py",
            ["_hf_arch", "_hf_attr", "is_hy_v4", "is_deepseek_dsa"],
        )
        configure = definitions(
            "configs/model_config.py",
            ["_config_draft_model"],
            {"MIMO_V2_MODEL_ARCHS": ()},
            cls="ModelConfig",
        )["_config_draft_model"]
        with patch.dict(
            sys.modules,
            {"sglang.srt.configs.dots3": NS(Dots3Config=type("Dots3", (), {}))},
        ):
            for target, draft in (
                ("HYV4ForCausalLM", "HYV4ForCausalLMNextN"),
                ("GlmMoeDsaForCausalLM", "GlmMoeDsaForCausalLMNextN"),
                ("DeepseekV3ForCausalLM", "DeepseekV3ForCausalLMNextN"),
            ):
                for is_draft in (True, False):
                    config = NS(
                        architectures=[target],
                        num_nextn_predict_layers=1,
                        index_topk=2048,
                    )
                    configure(
                        NS(
                            is_draft_model=is_draft,
                            hf_config=config,
                            hf_text_config=config,
                        )
                    )
                    self.assertEqual(
                        config.architectures, [draft if is_draft else target]
                    )
                    self.assertTrue(ns["is_deepseek_dsa"](config))
            config = NS(architectures=["HYV4ForCausalLM"], num_nextn_predict_layers=0)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                configure(
                    NS(is_draft_model=True, hf_config=config, hf_text_config=config)
                )

    def test_launch_eagle_chain(self):
        args = NS(
            pp_size=1,
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            enable_two_batch_overlap=False,
            enable_single_batch_overlap=False,
            cuda_graph_config=NS(
                prefill=NS(backend="disabled"), decode=NS(backend="full")
            ),
            attention_backend="dsa",
            dsa_prefill_backend="flashmla_kv",
            dsa_decode_backend="flashmla_kv",
            enable_dp_attention=True,
            moe_dense_tp_size=1,
        )
        parallel = NS(
            attn_cp_size=1,
            dcp_enabled=True,
            attn_tp_size=2,
            attn_dcp_size=2,
            dcp_group=NS(ranks=(0, 1)),
            attn_tp_group=NS(ranks=(0, 1)),
            dcp_comm_backend="ag_rs",
        )
        hy4.validate_hyv4_launch(args, parallel)
        for key, value in (
            ("speculative_algorithm", "EAGLE3"),
            ("speculative_eagle_topk", 2),
            ("speculative_num_steps", 0),
            ("speculative_num_draft_tokens", None),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                hy4.validate_hyv4_launch(NS(**{**vars(args), key: value}), parallel)

    def test_quant_config_isolated_and_names_match_runtime(self):
        remap = definitions(NEXTN, ["_mtp_quant_config"], {"copy": copy})[
            "_mtp_quant_config"
        ]
        config = NS(
            ignore=["model.mtp_layers.0.self_attn.g_proj", "model.mtp.layers.0.enorm"],
            ignored_layers=["model.mtp_layers.0.mlp.experts"],
            linear_fp8_config="original",
            checkpoint_format="hy4_w4a8_v1",
        )
        original = copy.deepcopy(config)
        draft = remap(config)
        self.assertEqual(vars(config), vars(original))
        self.assertEqual(
            draft.ignore, ["model.decoder.self_attn.linear_gate", "model.enorm"]
        )
        self.assertEqual(draft.ignored_layers, ["model.decoder.mlp.experts"])
        self.assertIsNone(draft.linear_fp8_config)
        self.assertEqual(draft.checkpoint_format, "hy4_w4a8_v1")
        self.assertIsNone(remap(None))

    def test_checkpoint_names(self):
        for suffix, expected in (
            ("self_attn.g_proj.weight.weight", "self_attn.linear_gate.weight"),
            ("mlp.experts.0.gate_proj.weight.packed", "mlp.experts.0.gate_proj.weight"),
            (
                "mlp.experts.0.gate_proj.weight.scale",
                "mlp.experts.0.gate_proj.weight_scale",
            ),
            ("self_attn.learnable_sink_param.weight", "self_attn.learnable_sink_param"),
            ("final_layernorm.weight.weight", "shared_head.norm.weight"),
        ):
            for prefix in ("model.mtp_layers.0.", "model.mtp.layers.0."):
                self.assertEqual(
                    mtp.normalize_hyv4_mtp_weight_name(prefix + suffix), expected
                )
        self.assertIsNone(
            mtp.normalize_hyv4_mtp_weight_name("model.layers.0.self_attn.g_proj.weight")
        )

    def test_checkpoint_completeness_and_shapes(self):
        config, metadata = checkpoint_fixture()
        self.assertEqual(
            mtp.validate_hyv4_mtp_metadata(config, metadata, packed_experts=True)[
                "mtp_hidden_size"
            ],
            8,
        )
        for missing in metadata:
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                mtp.validate_hyv4_mtp_metadata(
                    config,
                    {k: v for k, v in metadata.items() if k != missing},
                    packed_experts=True,
                )
        for name, bad in (
            ("eh_proj.weight", dict(shape=[8, 40], dtype="BF16")),
            ("self_attn.linear_gate.weight", dict(shape=[8, 8], dtype="I8")),
            ("mlp.experts.0.gate_proj.weight", dict(shape=[4, 8], dtype="BF16")),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                mtp.validate_hyv4_mtp_metadata(
                    config, {**metadata, name: bad}, packed_experts=True
                )

    def test_streaming_loader_names_permutation_and_missing_weights(self):
        config, metadata = checkpoint_fixture()
        permute = definitions(MODEL, ["permute_hyv4_indexer_weight"])[
            "permute_hyv4_indexer_weight"
        ]
        load = definitions(
            NEXTN,
            ["load_weights"],
            {
                "normalize_hyv4_mtp_weight_name": mtp.normalize_hyv4_mtp_weight_name,
                "validate_hyv4_mtp_metadata": mtp.validate_hyv4_mtp_metadata,
                "hyv4_linear_scale_suffix": lambda _: "",
                "permute_hyv4_indexer_weight": permute,
            },
            cls="HYV4ForCausalLMNextN",
        )["load_weights"]
        dtypes = {"BF16": torch.bfloat16, "I8": torch.int8, "F32": torch.float32}
        weights = [
            (
                "model.mtp_layers.0." + name,
                torch.zeros(meta["shape"], dtype=dtypes[meta["dtype"]]),
            )
            for name, meta in metadata.items()
        ]
        weights.append(("model.layers.0.self_attn.g_proj.weight", torch.zeros(8, 8)))
        loaded = []

        def do_load(iterator, is_nextn):
            self.assertTrue(is_nextn)
            loaded.extend(iterator)

        model = NS(
            config=NS(**config),
            quant_config=NS(checkpoint_format="hy4_w4a8_v1"),
            _initialize_nextn_conf=lambda _: NS(nextn_layer_prefix="model.layers.3"),
            do_load_weights=do_load,
        )
        load(model, weights)
        self.assertEqual(len(loaded), len(metadata))
        self.assertTrue(all(name.startswith("model.layers.3.") for name, _ in loaded))
        with self.assertRaisesRegex(ValueError, "Missing HYV4 MTP weight"):
            load(model, weights[1:])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load(model, weights + weights[:1])

    def test_mtp_decoder_tp2_residual_and_context_cleanup(self):
        for rank in (0, 1):
            context = NS(
                set_attn_inputs=lambda _: None,
                clear_attn_inputs=lambda: events.append("clear"),
            )
            events = []
            forward = definitions(
                NEXTN,
                ["forward"],
                {
                    "get_attn_tp_context": lambda: context,
                    "AttentionInputs": lambda *a: a,
                    "hyv4_attn_tp_reduce_scatter": lambda x: (x * 2).chunk(2)[rank],
                    "hyv4_attn_tp_split": lambda x: x.chunk(2)[rank],
                },
                cls="HYV4MTPDecoderLayer",
            )["forward"]
            x = torch.arange(24, dtype=torch.float32).view(4, 6)

            class Attention:
                prepare_qkv_latent = None

                def __call__(self, pos, hidden, *args, **kwargs):
                    return hidden * 3, None

            model = NS(
                input_layernorm=lambda x: x + 1,
                self_attn=Attention(),
                dp_attn_scattered=True,
                post_attention_layernorm=lambda h, r: (h + r, h + r),
                mlp=lambda h, batch: h.square(),
            )
            output, residual, _ = forward(model, None, x, None, None)
            expected = ((x + 1) * 6 + x).chunk(2)[rank]
            torch.testing.assert_close(output, expected.square())
            torch.testing.assert_close(residual, expected)
            self.assertEqual(events, ["clear"])
            model.self_attn = NS(prepare_qkv_latent=None)
            with self.assertRaises(TypeError):
                forward(model, None, x, None, None)
            self.assertEqual(events, ["clear", "clear"])

    def test_mtp_fuses_d_hidden_and_returns_dp_full_tokens(self):
        forward = definitions(
            NEXTN,
            ["forward"],
            {
                "BumpAllocator": lambda **kw: None,
                "IndexTopKShareState": NS(
                    from_mtp_carry=lambda _: NS(
                        topk_indices=None, update=lambda _: None, publish=lambda: None
                    )
                ),
                "hyv4_attn_tp_gather": lambda x: torch.cat((x, x)),
            },
            cls="HYV4ModelNextN",
        )["forward"]
        projection = nn.Linear(8, 4, bias=False)
        captured = []

        def decoder(pos, h, batch, alloc, **kw):
            captured.append(h)
            return h[:2], h[:2] * 2, None

        model = NS(
            embed_tokens=lambda ids: torch.ones(len(ids), 4),
            enorm=lambda x: x * 2,
            hnorm=lambda x: x * 3,
            eh_proj=projection,
            decoder=decoder,
            alt_stream=None,
            shared_head=NS(norm=lambda h, r: (h + r, None)),
            dp_attn_scattered=True,
        )
        hidden = torch.arange(16, dtype=torch.float32).view(4, 4)
        batch = NS(
            spec_info=NS(hidden_states=hidden), forward_mode=NS(is_idle=lambda: False)
        )
        output = forward(model, torch.arange(4), None, batch)
        expected = projection(torch.cat((torch.full((4, 4), 2.0), hidden * 3), -1))
        torch.testing.assert_close(captured[0], expected)
        torch.testing.assert_close(
            output, torch.cat((expected[:2] * 3, expected[:2] * 3))
        )
        batch.spec_info.hidden_states = torch.zeros(4, 16)
        with self.assertRaisesRegex(ValueError, "D-dimensional"):
            forward(model, torch.arange(4), None, batch)

    def test_offline_header_checker_mtp_and_non_mtp(self):
        path = ROOT.parents[2] / "test/manual/hyv4_dcp/check_checkpoint.py"
        spec = importlib.util.spec_from_file_location("hy4_checkpoint_check", path)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        config, metadata = checkpoint_fixture()
        config.update(
            architectures=["HYV4ForCausalLM"],
            model_type="hy_v4",
            quantization_config={"checkpoint_format": "hy4_w4a8_v1"},
            n_routed_experts=32,
        )
        metadata["mlp.gate.weight"]["shape"] = [32, 8]
        metadata["mlp.gate.e_score_correction_bias"]["shape"] = [32]
        for expert in range(2, 32):
            metadata.update(
                {
                    name.replace("experts.0.", f"experts.{expert}."): meta
                    for name, meta in list(metadata.items())
                    if "experts.0." in name
                }
            )
        headers = {
            "model.mtp_layers.0." + name: meta for name, meta in metadata.items()
        }
        for layer in range(3):
            headers[f"model.layers.{layer}.self_attn.g_proj.weight.weight"] = dict(
                shape=[8, 8], dtype="BF16"
            )
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "config.json").write_text(json.dumps(config))

            def save():
                data = json.dumps(headers).encode()
                (folder / "model.safetensors").write_bytes(
                    struct.pack("<Q", len(data)) + data
                )

            save()
            self.assertEqual(checker.check(folder, mtp=True)["mtp_layers"], 1)
            del headers["model.mtp_layers.0.eh_proj.weight"]
            save()
            self.assertEqual(checker.check(folder)["attention_tp"], 2)
            with self.assertRaisesRegex(ValueError, "eh_proj"):
                checker.check(folder, mtp=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
