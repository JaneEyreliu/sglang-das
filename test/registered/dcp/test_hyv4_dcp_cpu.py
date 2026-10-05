"""CPU numerical/contract tests, runnable without the GPU serving dependencies.

Load pure helpers directly and compile selected production definitions with
CPU stand-ins for infrastructure. Kernel/RCCL/PD end-to-end tests still require
an HCU deployment; these tests do not claim to execute those dependencies.

Run: python test/registered/dcp/test_hyv4_dcp_cpu.py
"""

import ast
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def definitions(path, names, namespace=None, cls=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if cls:
        nodes = next(n for n in nodes if getattr(n, "name", None) == cls).body
    selected = [n for n in nodes if getattr(n, "name", None) in names]
    assert len(selected) == len(names), (path, names)
    for n in selected:
        if cls:
            n.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + selected,
        type_ignores=[],
    )
    ns = {"torch": torch, "nn": nn, **(namespace or {})}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), ns)
    return ns


spec = importlib.util.spec_from_file_location("hy4_dcp_cpu", ROOT / "layers/hy4_dcp.py")
hy4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hy4)
MODEL = "models/hunyuan_v4.py"
QUANT = "layers/quantization/slimquant_w4a8_marlin.py"


class TestHYV4DCP(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_topk_v2_plan_requires_fused_topk(self):
        ns = definitions(
            "layers/attention/dsa_backend.py",
            ["_build_topk_v2_plan", "_refresh_topk_v2_plan"],
            cls="DeepseekSparseAttnBackend",
        )
        seq_lens = torch.tensor([1, 2], dtype=torch.int32)
        module_name = "sglang.kernels.ops.attention.dsv4.topk"
        for fused, v2 in ((False, True), (False, False), (True, False), (True, True)):
            with self.subTest(fused=fused, v2=v2):
                expected_plan = torch.zeros(3, 2, dtype=torch.int32)
                planner = Mock(return_value=expected_plan)
                backend = NS(
                    use_fused_topk=fused,
                    dsa_topk_backend=NS(should_use_topk_v2=lambda: v2),
                )
                with (
                    patch.dict(sys.modules, {module_name: NS(plan_topk_v2=planner)}),
                    patch("builtins.__import__", wraps=__import__) as imports,
                ):
                    plan = ns["_build_topk_v2_plan"](backend, seq_lens)
                    if fused and v2:
                        self.assertIs(plan, expected_plan)
                        planner.assert_called_once_with(seq_lens)
                        pointer = plan.data_ptr()
                        planner.return_value = torch.ones_like(expected_plan)
                        metadata = NS(topk_v2_plan=plan, dsa_seqlens_expanded=seq_lens)
                        ns["_refresh_topk_v2_plan"](backend, metadata)
                        self.assertEqual(plan.data_ptr(), pointer)
                        torch.testing.assert_close(plan, torch.ones_like(plan))
                    else:
                        self.assertIsNone(plan)
                        ns["_refresh_topk_v2_plan"](backend, NS(topk_v2_plan=plan))
                        planner.assert_not_called()
                        self.assertFalse(
                            any(
                                call.args[0] == module_name
                                for call in imports.call_args_list
                            )
                        )

    def test_sink_matches_full_attention(self):
        # Independent dense reference: concatenate a virtual key whose V is 0.
        for dcp in (1, 2, 4, 8):
            for base2 in (False, True):
                with self.subTest(dcp=dcp, base2=base2):
                    logits = torch.randn(3, 6, 9) * 4
                    values = torch.randn(3, 6, 9, 5)
                    sink = torch.linspace(-12, 12, 6)
                    ref_logits = torch.cat(
                        (logits, sink[None, :, None].expand(3, -1, 1)), -1
                    )
                    ref_values = torch.cat((values, torch.zeros(3, 6, 1, 5)), -2)
                    expected = (ref_logits.softmax(-1)[..., None] * ref_values).sum(-2)
                    parts, lses = [], []
                    for rank in range(dcp):
                        local = logits[..., rank::dcp]
                        local_v = values[..., rank::dcp, :]
                        o = (local.softmax(-1)[..., None] * local_v).sum(-2)
                        lse = local.logsumexp(-1)
                        if base2:
                            lse *= math.log2(math.e)
                        o, lse = hy4.apply_hyv4_sink(
                            o,
                            lse,
                            sink,
                            torch.full((3,), local.shape[-1]),
                            dcp_size=dcp,
                            lse_base2=base2,
                        )
                        parts.append(o)
                        lses.append(lse * (math.log(2) if base2 else 1))
                    weights = torch.stack(lses).softmax(0)
                    actual = (torch.stack(parts) * weights[..., None]).sum(0)
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_sink_empty_shard_is_finite_and_counted_once(self):
        sink = torch.tensor([-5.0, 5.0])
        o0, l0 = hy4.apply_hyv4_sink(
            torch.full((1, 2, 3), float("nan")),
            torch.full((1, 2), float("inf")),
            sink,
            torch.tensor([0]),
            dcp_size=2,
        )
        torch.testing.assert_close(o0, torch.zeros_like(o0))
        torch.testing.assert_close(l0, sink[None] - math.log(2))
        o1, l1 = hy4.apply_hyv4_sink(
            torch.ones(1, 2, 3), torch.zeros(1, 2), sink, torch.tensor([1]), dcp_size=2
        )
        weights = torch.stack((l0, l1)).softmax(0)
        actual = weights[0, ..., None] * o0 + weights[1, ..., None] * o1
        expected = (1 / (1 + sink.exp()))[None, :, None].expand_as(actual)
        torch.testing.assert_close(actual, expected)

    def test_sink_all_empty_and_zero_batch(self):
        for tokens in (0, 3):
            o, lse = hy4.apply_hyv4_sink(
                torch.full((tokens, 4, 2), float("nan")),
                torch.full((tokens, 4), -torch.inf),
                torch.zeros(4),
                torch.zeros(tokens),
                dcp_size=2,
                lse_base2=True,
            )
            self.assertTrue(torch.isfinite(o).all())
            self.assertTrue(torch.isfinite(lse).all())
            self.assertEqual(o.shape, (tokens, 4, 2))

    def test_sink_rejects_mismatched_heads(self):
        with self.assertRaises(ValueError):
            hy4.apply_hyv4_sink(
                torch.zeros(1, 2, 3), torch.zeros(1, 2), torch.zeros(4), torch.ones(1)
            )

    def test_gate_tp2_uses_real_column_weight_loader(self):
        loader = definitions(
            "layers/linear.py",
            ["weight_loader"],
            {"_is_cpu": False},
            cls="ColumnParallelLinear",
        )["weight_loader"]
        gate = definitions(MODEL, ["apply_attention_output_gate"], cls="HYV4Attention")[
            "apply_attention_output_gate"
        ]
        hidden, heads, value_dim = 12, 8, 3
        x, w = torch.randn(5, hidden), torch.randn(heads * value_dim, hidden)
        attention, oproj = torch.randn(5, heads * value_dim), torch.randn(
            7, heads * value_dim
        )
        expected = (attention * torch.sigmoid(x @ w.T)) @ oproj.T
        partials = []
        for rank in (0, 1):
            parameter = nn.Parameter(
                torch.empty(heads * value_dim // 2, hidden), requires_grad=False
            )
            parameter.output_dim = 0
            loader(
                NS(tp_rank=rank, tp_size=2, use_presharded_weights=False), parameter, w
            )
            local = attention.chunk(2, -1)[rank]
            gated = gate(None, local, x @ parameter.T)
            partials.append(gated @ oproj.chunk(2, -1)[rank].T)
        torch.testing.assert_close(sum(partials), expected, atol=1e-5, rtol=1e-5)
        with self.assertRaises(ValueError):
            gate(None, torch.empty(5, 12), torch.empty(5, 24))

    def test_sink_head_slices_follow_attention_tp(self):
        for rank in (0, 1):
            ns = definitions(
                MODEL,
                ["get_local_attention_sink", "get_dcp_attention_sink"],
                {"get_parallel": lambda: NS(attn_tp_rank=rank, attn_dcp_size=2)},
                cls="HYV4Attention",
            )
            attn = NS(num_local_heads=4, learnable_sink_param=torch.arange(8))
            torch.testing.assert_close(
                ns["get_local_attention_sink"](attn), torch.arange(8).chunk(2)[rank]
            )
            torch.testing.assert_close(
                ns["get_dcp_attention_sink"](attn), torch.arange(8)
            )

    def test_reduce_scatter_has_separate_output_storage(self):
        def collective(out, inp):
            self.assertNotEqual(
                out.untyped_storage().data_ptr(), inp.untyped_storage().data_ptr()
            )
            out.copy_(inp[: out.shape[0]] * 2)

        ns = definitions(
            MODEL,
            ["hyv4_attn_tp_split", "hyv4_attn_tp_reduce_scatter"],
            {
                "get_parallel": lambda: NS(attn_tp_size=2, attn_tp_rank=0),
                "attn_tp_reduce_scatter_tensor": collective,
            },
        )
        x = torch.randn(6, 8)
        before = x.clone()
        out = ns["hyv4_attn_tp_reduce_scatter"](x)
        torch.testing.assert_close(x, before)
        torch.testing.assert_close(out, before[:3] * 2)

    def test_int4_conversion_preserves_signed_values_and_true_scales(self):
        normalize = definitions(
            QUANT, ["normalize_checkpoint_weights"], cls="SlimQuantW4A8Int8MarlinConfig"
        )["normalize_checkpoint_weights"]
        # All 256 byte patterns cover both signed nibbles, including -8 and -1.
        packed = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
        signed = lambda x: (x.to(torch.int16) + 8) % 16 - 8
        original = torch.stack((signed(packed & 15), signed(packed >> 4)), -1).flatten(
            -2
        )
        scale = torch.rand(16, 1)
        layer = NS(
            w13_weight=nn.Parameter(packed.clone(), requires_grad=False),
            w2_weight=nn.Parameter(packed.clone(), requires_grad=False),
            w13_weight_scale=nn.Parameter(scale.clone()),
            w2_weight_scale=nn.Parameter(scale.clone()),
        )
        normalize(NS(checkpoint_format="hy4_w4a8_v1"), layer)
        for name in ("w13_weight", "w2_weight"):
            converted = getattr(layer, name)
            decoded = torch.stack(
                (signed(converted >> 4), signed(converted & 15)), -1
            ).flatten(-2)
            actual = decoded * getattr(layer, name + "_scale") * 16
            torch.testing.assert_close(actual, original * scale)

    def test_deepep_hipc_weight_loading_preserves_scale_contract(self):
        normalize = definitions(
            QUANT, ["normalize_checkpoint_weights"], cls="SlimQuantW4A8Int8MarlinConfig"
        )["normalize_checkpoint_weights"]
        process = definitions(
            QUANT,
            ["process_weights_after_loading"],
            {"Parameter": nn.Parameter, "_use_lightop_w4a8_marlin_moe": False,
             "get_moe_a2a_backend": lambda: NS(is_megamoe=lambda: False)},
            cls="SlimQuantW4A8Int8MarlinMoEMethod",
        )["process_weights_after_loading"]
        # Cover every signed nibble value. The packer only reorders weights;
        # leave that GPU operation out of this CPU scale-contract test.
        packed = torch.arange(256, dtype=torch.uint8).reshape(1, 16, 16)
        signed = lambda x: (x.to(torch.int16) + 8) % 16 - 8
        original = torch.stack((signed(packed & 15), signed(packed >> 4)), -1)
        scales = (torch.rand(1, 16, 1), torch.rand(1, 16, 1))
        for fmt in ("hy4_w4a8_v1", None, "legacy_other_format"):
            with self.subTest(checkpoint_format=fmt):
                config = NS(checkpoint_format=fmt)
                config.normalize_checkpoint_weights = lambda layer: normalize(config, layer)
                layer = NS()
                for name, scale in zip(("w13_weight", "w2_weight"), scales):
                    setattr(
                        layer, name, nn.Parameter(packed.clone(), requires_grad=False)
                    )
                    setattr(
                        layer, name + "_scale",
                        nn.Parameter(scale.clone(), requires_grad=False),
                    )
                pack = Mock(side_effect=lambda weight: weight.clone())
                with patch.dict(
                    sys.modules, {"deepgemm": NS(pack_w4a8_moe_hipc_weight=pack)}
                ):
                    process(NS(quant_config=config, use_deepep=True), layer)
                self.assertEqual(pack.call_count, 2)
                for name, scale in zip(("w13_weight", "w2_weight"), scales):
                    actual_scale = getattr(layer, name + "_scale")
                    if fmt == "hy4_w4a8_v1":
                        torch.testing.assert_close(actual_scale, scale / 16)
                        converted = getattr(layer, name)
                        decoded = torch.stack(
                            (signed(converted >> 4), signed(converted & 15)), -1
                        ).flatten(-2)
                        # The installed HIPC kernel contributes the x16 factor.
                        torch.testing.assert_close(
                            decoded * actual_scale * 16, original.flatten(-2) * scale
                        )
                    else:
                        # Preserve the existing conversion for other formats.
                        torch.testing.assert_close(actual_scale, scale * 16)
                        torch.testing.assert_close(getattr(layer, name), packed)

    def test_legacy_quantization_is_noop(self):
        normalize = definitions(
            QUANT, ["normalize_checkpoint_weights"], cls="SlimQuantW4A8Int8MarlinConfig"
        )["normalize_checkpoint_weights"]
        for format in (None, "legacy_other_format"):
            # No access to weights at all on old formats.
            normalize(NS(checkpoint_format=format), object())

    def test_shared_int4_unpack_keeps_scales(self):
        class LinearBase:
            def process_weights_after_loading(self, layer):
                pass

        cls = definitions(
            QUANT,
            ["HYV4SharedExpertLinearMethod"],
            {"SlimQuantW4A8Int8LinearMethod": LinearBase},
        )["HYV4SharedExpertLinearMethod"]
        packed = torch.tensor([[0x78, 0xF1, 0x80]], dtype=torch.uint8)
        scale = torch.tensor([[0.3]])
        layer = NS(weight=nn.Parameter(packed, requires_grad=False), weight_scale=scale)
        cls().process_weights_after_loading(layer)
        torch.testing.assert_close(
            layer.weight, torch.tensor([[-8, 7, 1, -1, 0, -8]], dtype=torch.int8)
        )
        torch.testing.assert_close(layer.weight_scale, scale)

    def test_weight_name_mapping(self):
        normalize = definitions(MODEL, ["normalize_hyv4_weight_name"])[
            "normalize_hyv4_weight_name"
        ]
        pairs = {
            "self_attn.g_proj.weight.weight": "self_attn.linear_gate.weight",
            "mlp.experts.3.up_proj.weight.packed": "mlp.experts.3.up_proj.weight",
            "mlp.experts.3.up_proj.weight.scale": "mlp.experts.3.up_proj.weight_scale",
            "self_attn.learnable_sink_param.weight": "self_attn.learnable_sink_param",
            "hc_scale.weight": "hc_scale",
            "self_attn.q_a_proj.weight": "self_attn.q_a_proj.weight",
        }
        for source, target in pairs.items():
            self.assertEqual(
                normalize("model.layers.1." + source), "model.layers.1." + target
            )

    def test_indexer_rope_permutation(self):
        permute = definitions(MODEL, ["permute_hyv4_indexer_weight"])[
            "permute_hyv4_indexer_weight"
        ]
        config = NS(index_n_heads=2, index_head_dim=4, qk_rope_head_dim=2)
        x = torch.arange(24).reshape(8, 3)
        expected = x[torch.tensor([2, 3, 0, 1, 6, 7, 4, 5])]
        torch.testing.assert_close(
            permute("model.layers.0.self_attn.indexer.wq_b.weight", x, config), expected
        )
        self.assertIs(permute("model.layers.0.mlp.gate.weight", x, config), x)

    def test_hyv4_only_indexer_sharing(self):
        ns = definitions(
            "configs/model_config.py",
            [
                "_hf_arch",
                "_hf_attr",
                "is_hy_v4",
                "is_deepseek_dsa",
                "dsa_layer_skips_topk",
            ],
        )
        hy = NS(
            architectures=["HYV4ForCausalLM"],
            index_topk=2048,
            indexer_types=["full", "full", "shared", "shared", "shared", "full"],
        )
        glm = NS(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            indexer_types=["shared"] * 6,
        )
        self.assertEqual(
            [ns["dsa_layer_skips_topk"](hy, i) for i in range(7)],
            [False, False, True, True, True, False, False],
        )
        self.assertFalse(any(ns["dsa_layer_skips_topk"](glm, i) for i in range(6)))

    def test_hy4_backend_defaults_resolve_omitted_flags_without_cli_mutation(self):
        ns = definitions(
            "arg_groups/overrides.py",
            [
                "register_model_override",
                "_register_for",
                "_invoke_provider",
                "collect_model_override_declarations",
                "materialize_declarations",
                "resolution_result",
                "_hyv4_dsa_backend_overrides",
            ],
            {
                "_MODEL_OVERRIDE_FNS": {},
                "MODEL_OVERRIDES": {},
                "_PREDICATE_OVERRIDE_FNS": [],
                "logger": NS(info=lambda *a: None, warning=lambda *a: None),
            },
        )
        collect = ns["collect_model_override_declarations"]
        for arch in ("HYV4ForCausalLM", "HYV4ForCausalLMNextN"):
            for prefill, decode in (
                (None, "flashmla_kv"),
                (None, None),
                ("flashmla_kv", None),
                ("flashmla_sparse", "flashmla_kv"),
            ):
                with self.subTest(arch=arch, prefill=prefill, decode=decode):
                    args = NS(dsa_prefill_backend=prefill, dsa_decode_backend=decode,
                              dcp_size=2, is_attention_backend_not_set=lambda: True)
                    config = NS(architectures=[arch])
                    declared = collect(arch, args, config)
                    self.assertEqual(args.dsa_prefill_backend, prefill)
                    self.assertEqual(args.dsa_decode_backend, decode)
                    args._resolved_overrides = declared
                    ns["materialize_declarations"](args)
                    self.assertEqual(args.dsa_prefill_backend, prefill or "flashmla_kv")
                    self.assertEqual(args.dsa_decode_backend, decode or "flashmla_kv")
                    for name in ("dsa_prefill_backend", "dsa_decode_backend"):
                        self.assertEqual(
                            ns["resolution_result"](args, name), getattr(args, name)
                        )
        # Exact architecture registration leaves all other models' defaults alone.
        for arch in ("GlmMoeDsaForCausalLM", "DeepseekV3ForCausalLM"):
            args = NS(dsa_prefill_backend=None, dsa_decode_backend=None)
            self.assertEqual(collect(arch, args, NS(architectures=[arch])), [])
            self.assertIsNone(args.dsa_prefill_backend)

    def test_non_dcp_backend_defaults_and_prefill_cp(self):
        ns = definitions(
            "arg_groups/overrides.py",
            ["_hyv4_dsa_backend_overrides"],
            {
                "_register_for": lambda *arches: lambda fn: fn,
                "is_hcu": lambda: True,
                "logger": NS(info=lambda *a: None, warning=lambda *a: None),
            },
        )
        for dtype, decode in (("auto", "flashmla_sparse"), ("fp8_e4m3", "flashmla_kv")):
            for prefill_cp in (False, True):
                with self.subTest(dtype=dtype, prefill_cp=prefill_cp):
                    args = NS(
                        dcp_size=1, is_attention_backend_not_set=lambda: True,
                        kv_cache_dtype=dtype, enable_prefill_cp=prefill_cp,
                        moe_a2a_backend="none", tp_size=8, dp_size=1,
                        dsa_prefill_backend=None, dsa_decode_backend=None,
                    )
                    result = ns["_hyv4_dsa_backend_overrides"](args, NS())
                    self.assertEqual(result["dsa_prefill_backend"], "flashmla_sparse")
                    self.assertEqual(result["dsa_decode_backend"], decode)
                    self.assertEqual(result.get("kv_cache_dtype", dtype),
                                     "bfloat16" if dtype == "auto" else dtype)
                    if prefill_cp:
                        self.assertEqual(result["attn_cp_size"], 8)
                        self.assertEqual(result["ep_size"], 8)
                        self.assertEqual(result["moe_a2a_backend"], "deepep")
                        self.assertEqual(result["moe_dense_tp_size"], 1)
                        self.assertTrue(result["enable_dp_attention"])
                    else:
                        self.assertNotIn("attn_cp_size", result)

    def test_launch_contract(self):
        args = NS(
            cuda_graph_config=None,
            pp_size=1,
            speculative_algorithm=None,
            enable_two_batch_overlap=False,
            enable_single_batch_overlap=False,
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
            dcp_group=NS(ranks=(6, 7)),
            attn_tp_group=NS(ranks=(6, 7)),
            dcp_comm_backend="ag_rs",
        )
        for backend in ("ag_rs", "a2a"):
            with self.subTest(backend=backend):
                hy4.validate_hyv4_launch(
                    args, NS(**{**vars(parallel), "dcp_comm_backend": backend})
                )
        for key, value in (
            ("pp_size", 2),
            ("speculative_algorithm", "EAGLE"),
            ("enable_two_batch_overlap", True),
            ("enable_single_batch_overlap", True),
            ("dsa_prefill_backend", "flashmla_sparse"),
            ("moe_dense_tp_size", 2),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                hy4.validate_hyv4_launch(NS(**{**vars(args), key: value}), parallel)
        for key, value in (
            ("attn_tp_size", 4),
            ("attn_cp_size", 2),
            ("dcp_comm_backend", "fi_a2a"),
            ("dcp_group", NS(ranks=(7, 6))),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                hy4.validate_hyv4_launch(args, NS(**{**vars(parallel), key: value}))

    def test_non_dcp_prefill_cp_keeps_existing_launch_support(self):
        hy4.validate_hyv4_launch(
            NS(attention_backend="dsa", dsa_prefill_backend="flashmla_sparse"),
            NS(dcp_enabled=False, attn_cp_size=8),
        )

    def test_flashmla_backend_sink_plumbing_and_legacy_output(self):
        # Exercise production head padding, LSE layout conversion, empty-KV
        # handling, and the no-sink branch. FlashMLA itself is a CPU stand-in.
        def fake_flashmla(**kwargs):
            self.assertIsNone(kwargs.get("attn_sink"), "HY4 sink must be applied exactly once")
            q = kwargs["q"]
            return torch.ones(q.shape[0], 1, q.shape[2], 3), torch.zeros(
                q.shape[0], q.shape[2], 1
            )

        def fix_empty(o, lse, counts, *args, **kwargs):
            o[counts == 0] = 0
            lse[counts == 0] = -torch.inf

        ns = definitions(
            "layers/attention/dsa_backend.py",
            ["_forward_flashmla_kv"],
            {
                "get_flashmla_op": lambda *args, **kwargs: fake_flashmla,
                "_is_hcu": False,
                "_LOG2_E": math.log2(math.e),
                "fixup_zero_kv_rows": fix_empty,
            },
            cls="DeepseekSparseAttnBackend",
        )
        backend = NS(
            flashmla_kv_num_q_heads=64,
            real_page_size=64,
            kv_cache_dim=8,
            dsa_kv_cache_store_fp8=True,
            dsa_index_topk=4,
            dcp_size=2,
            get_device_int32_arange=torch.arange,
        )
        metadata = NS(
            dsa_cache_seqlens_int32=torch.tensor([1, 1]),
            flashmla_metadata=NS(flashmla_metadata=None, num_splits=None),
        )
        indices = torch.tensor([[0, -1, -1, -1], [-1, -1, -1, -1]])
        layer = NS(tp_q_head_num=4, head_dim=8)
        for dcp in (False, True):
            kwargs = dict(
                q_all=torch.ones(2, 4, 8),
                kv_cache=torch.zeros(1, 64, 1, 8),
                v_head_dim=3,
                sm_scale=1.0,
                layer=layer,
                metadata=metadata,
                page_table_1=indices,
                return_lse=dcp,
            )
            result = ns["_forward_flashmla_kv"](backend, **kwargs)
            if dcp:
                torch.testing.assert_close(result[0][0], torch.ones(4, 3))
                self.assertTrue(torch.isneginf(result[1][1]).all())
            else:
                torch.testing.assert_close(result, torch.ones(2, 1, 4, 3))
            layer.hyv4_sink_getter = lambda: torch.zeros(4)
            # The merged MLA forward may also pass the native sink argument.
            # The HY4 correction owns it, so the kernel must not consume it.
            kwargs["attn_sink"] = torch.zeros(4)
            with patch.dict(sys.modules, {"sglang.srt.layers.hy4_dcp": hy4}):
                result = ns["_forward_flashmla_kv"](backend, **kwargs)
            o = result[0] if dcp else result.squeeze(1)
            torch.testing.assert_close(o[0], torch.full((4, 3), 2 / 3 if dcp else 0.5))
            torch.testing.assert_close(o[1], torch.zeros(4, 3))
            if dcp:
                torch.testing.assert_close(result[1][1], torch.full((4,), -1.0))
            del layer.hyv4_sink_getter

    def test_mla_cores_apply_gate_after_dcp_before_output_projection(self):
        for rocm, backend in (
            (False, "ag_rs"), (True, "ag_rs"),
            (False, "a2a"), (True, "a2a"),
        ):
            path = (
                "models/deepseek_common/attention_forward_methods/forward_mla"
                + ("_rocm" if rocm else "")
                + ".py"
            )
            method = "forward_absorb_rocm_core" if rocm else "forward_absorb_core"
            events = []
            local = torch.randn(3, 2, 4, dtype=torch.bfloat16)
            wvc = torch.randn(2, 4, 3, dtype=torch.bfloat16)
            gate = torch.randn(3, 6, dtype=torch.bfloat16)

            def attention(*args, **kwargs):
                events.append("attention")
                return torch.zeros(3, 4, 4, dtype=torch.bfloat16), torch.zeros(3, 4)

            def combine(*args, **kwargs):
                events.append("combine")
                return local.transpose(0, 1)

            def a2a_combine(output, lse, group, **kwargs):
                self.assertEqual(backend, "a2a")
                self.assertEqual(kwargs["comm_backend"], "a2a")
                self.assertFalse(kwargs["is_lse_base_on_e"])
                self.assertEqual(tuple(output.shape), (3, 4, 4))
                events.append("combine")
                return local

            def apply_gate(x, g):
                events.append("gate")
                return x * torch.sigmoid(g)

            def project(x):
                events.append("o_proj")
                return x, None

            namespace = {
                "FORWARD_ABSORB_CORE_ATTENTION_BACKENDS": {"dsa"},
                "is_dcp_mla_decode_phase": lambda *args, **kwargs: True,
                "get_parallel": lambda: NS(
                    attn_dcp_size=2, dcp_comm_backend=backend, dcp_group=None
                ),
                "get_in_autotune_dummy_run": lambda: False,
                "is_mla_dcp_lse_base_on_e": lambda backend: False,
                "cp_lse_ag_out_rs_mla": combine,
                "dcp_a2a_lse_reduce": a2a_combine,
                "_SGLANG_EXPERIMENTAL_LORA_OPTI": False,
                "is_kv_b_lora_active": lambda attn: False,
                "is_in_tc_piecewise_cuda_graph": lambda: False,
                "_is_musa": False,
                "_is_hcu": False,
                "_scaled_bmm_weight": lambda weight, scale: weight.to(torch.bfloat16) * scale,
                "_use_aiter_gfx95": False,
            }
            if rocm:
                namespace.update(definitions(path, ["rocm_absorb_v_bmm"], namespace))
            ns = definitions(
                path,
                [method],
                namespace,
                cls=(
                    "DeepseekMLARocmForwardMixin" if rocm else "DeepseekMLAForwardMixin"
                ),
            )
            attn = NS(
                current_attention_backend="dsa",
                use_dsa=True,
                _fuse_rope_for_trtllm_mla=lambda batch: False,
                _skip_rope_for_dsa_tilelang_fused=lambda: False,
                attn_mqa_for_dcp_decode=attention,
                num_local_heads=2,
                kv_lora_rank=4,
                v_head_dim=3,
                use_deep_gemm_bmm=False,
                w_vc=wvc,
                w_kc=wvc,
                w_scale=1.0,
                o_proj=project,
                next_skip_topk=None,
                apply_attention_output_gate=apply_gate,
                prepare_attention_output_gate=lambda x: gate,
            )
            kwargs = dict(
                q_pe=None,
                k_pe=None,
                q_nope_out=None,
                k_nope=None,
                forward_batch=None,
                zero_allocator=None,
                positions=None,
                topk_indices=None,
                llama_4_scaling=None,
                attention_output_gate=gate,
            )
            output = ns[method](attn, **kwargs)
            expected = torch.bmm(local.transpose(0, 1), wvc).transpose(0, 1).flatten(
                1, 2
            ) * torch.sigmoid(gate)
            torch.testing.assert_close(output, expected)
            self.assertEqual(events, ["attention", "combine", "gate", "o_proj"])
            # Old models pass no gate and retain the original output.
            events.clear()
            kwargs["attention_output_gate"] = None
            output = ns[method](attn, **kwargs)
            torch.testing.assert_close(
                output,
                torch.bmm(local.transpose(0, 1), wvc).transpose(0, 1).flatten(1, 2),
            )
            self.assertEqual(events, ["attention", "combine", "o_proj"])

    def test_manifest_format_is_explicit_and_model_scoped(self):
        import json
        import os
        import tempfile

        resolve = definitions(
            "model_loader/weight_utils.py",
            ["_get_hyv4_checkpoint_format"],
            {"json": json, "os": os},
        )["_get_hyv4_checkpoint_format"]
        with tempfile.TemporaryDirectory() as folder:
            cfg = NS(model_type="hy_v4")
            quant = "slimquant_w4a8_marlin"
            self.assertIsNone(resolve(folder, cfg, quant))
            manifest = Path(folder) / "hy4-assets.json"
            manifest.write_text(json.dumps({"format": "hy4_w4a8_v1"}))
            self.assertEqual(resolve(folder, cfg, quant), "hy4_w4a8_v1")
            self.assertIsNone(resolve(folder, NS(model_type="glm_moe_dsa"), quant))
            self.assertIsNone(resolve(folder, cfg, "fp8"))
            manifest.write_text(json.dumps({"format": "unknown"}))
            with self.assertRaises(ValueError):
                resolve(folder, cfg, quant)

    def test_existing_dsa_launch_regression_suite(self):
        from enum import Enum

        enum = definitions(
            "layers/attention/dsa/hcu_int8_index_k_cache.py",
            ["IndexKCacheMode"],
            {"Enum": Enum},
        )["IndexKCacheMode"]
        abi = definitions(
            "mem_cache/memory_pool.py",
            ["get_index_k_cache_transfer_abi"],
            cls="DSATokenToKVPool",
        )["get_index_k_cache_transfer_abi"]
        pool = type("DSATokenToKVPool", (), {"get_index_k_cache_transfer_abi": abi})
        validator = definitions(
            "layers/attention/dsa_backend.py", ["_validate_dsa_dcp_launch"]
        )["_validate_dsa_dcp_launch"]
        tests = definitions(
            str(
                ROOT.parents[2] / "test/registered/dcp/test_dsa_dcp_launch_contract.py"
            ),
            ["_valid_config", "TestDSADCPLaunchContract"],
            {
                "unittest": unittest,
                "IndexKCacheMode": enum,
                "DSATokenToKVPool": pool,
                "_validate_dsa_dcp_launch": validator,
            },
        )
        result = unittest.TestResult()
        unittest.defaultTestLoader.loadTestsFromTestCase(
            tests["TestDSADCPLaunchContract"]
        ).run(result)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertGreaterEqual(result.testsRun, 6)

    def test_router_fp32_is_hyv4_only(self):
        # Execute the actual constructor and forward with CPU infrastructure.
        ns = definitions(
            "models/deepseek_v2.py",
            ["MoEGate"],
            {
                "_is_cpu": False,
                "_is_cpu_amx_available": False,
                "_use_aiter": False,
                "is_hy_v4": lambda cfg: cfg.architectures == ["HYV4ForCausalLM"],
                "is_deepseek_dsa": lambda cfg: True,
                "F": torch.nn.functional,
                "use_intel_amx_backend": lambda self: False,
                "get_exec": lambda: NS(
                    deterministic=NS(enable_deterministic_inference=True)
                ),
            },
        )
        cls = ns["MoEGate"]
        for arch in ("HYV4ForCausalLM", "GlmMoeDsaForCausalLM"):
            old_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                cfg = NS(
                    architectures=[arch],
                    n_routed_experts=4,
                    hidden_size=8,
                    router_fp32=True,
                    topk_method="noaux_tc",
                )
                router = cls(cfg, quant_config=None)
                self.assertEqual(
                    router.weight.dtype,
                    torch.float32 if arch.startswith("HYV4") else torch.bfloat16,
                )
                router.weight.data.copy_(torch.randn(4, 8))
                x = torch.randn(3, 8)
                logits = router(x)
                self.assertEqual(logits.dtype, router.weight.dtype)
                torch.testing.assert_close(
                    logits,
                    torch.nn.functional.linear(
                        x.to(router.weight.dtype), router.weight
                    ),
                )
            finally:
                torch.set_default_dtype(old_dtype)


if __name__ == "__main__":
    unittest.main(verbosity=2)
