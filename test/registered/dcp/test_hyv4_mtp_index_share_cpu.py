"""HY4 MTP index sharing contracts; CPU stand-ins do not execute HCU graphs."""

import ast
import sys
import importlib.util
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from torch import nn
from test_hyv4_dcp_cpu import ROOT, definitions

NEXTN = "models/hunyuan_v4_nextn.py"
spec = importlib.util.spec_from_file_location(
    "hy4_index_share", ROOT / "layers/attention/index_topk_share.py"
)
share = importlib.util.module_from_spec(spec)
spec.loader.exec_module(share)
Share = share.IndexTopKShareState


class TestHYV4MTPIndexShare(unittest.TestCase):
    def test_constructor_accepts_both_modes_and_preserves_config(self):
        # Execute the real constructor inside a class so super() is exercised.
        tree = ast.parse((ROOT / NEXTN).read_text())
        cls = next(
            n for n in tree.body if getattr(n, "name", "") == "HYV4ForCausalLMNextN"
        )
        cls.bases = [
            ast.Attribute(
                value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load()
            )
        ]
        cls.body = [n for n in cls.body if getattr(n, "name", "") == "__init__"]
        model = NS(decoder=NS(mlp=NS(num_fused_shared_experts=0)))
        validate = Mock()
        ns = dict(
            nn=nn,
            _mtp_quant_config=lambda q: q,
            get_pp_group=lambda: None,
            validate_hyv4_launch=validate,
            get_global_server_args=lambda: NS(enable_dp_lm_head=False),
            get_parallel=lambda: NS(dcp_enabled=True, attn_dcp_size=2),
            get_attn_tp_context=lambda: NS(init_context=lambda *a: None),
            HYV4ModelNextN=lambda *a, **kw: model,
            ParallelLMHead=lambda *a, **kw: nn.Identity(),
            LogitsProcessor=lambda config: NS(),
            add_prefix=lambda name, prefix: name,
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                str(ROOT / NEXTN),
                "exec",
            ),
            ns,
        )
        for enabled in (False, True):
            config = NS(
                index_share_for_mtp_iteration=enabled,
                q_lora_rank=4,
                vocab_size=8,
                hidden_size=4,
                enable_lm_head_fp32=True,
            )
            result = ns["HYV4ForCausalLMNextN"](config)
            self.assertIs(result.config, config)
            self.assertEqual(config.index_share_for_mtp_iteration, enabled)
        self.assertEqual(validate.call_count, 2)

    def backend_launch(self, arch, enabled, **overrides):
        # Run the actual backend constructor through its launch validator.
        # GPU allocations and infrastructure are CPU stand-ins; the model
        # predicate, constructor argument wiring and validator are production.
        path = "layers/attention/dsa_backend.py"
        validator = definitions(path, ["_validate_dsa_dcp_launch"])[
            "_validate_dsa_dcp_launch"
        ]
        model_defs = definitions("configs/model_config.py", ["_hf_arch", "is_hy_v4"])
        contract_path = (
            ROOT.parents[2] / "test/registered/dcp/test_dsa_dcp_launch_contract.py"
        )
        valid = definitions(str(contract_path), ["_valid_config"])["_valid_config"]
        args = valid(
            is_hcu_platform=True,
            device_capability=(9, 3),
            dcp_size=2,
            attn_tp_size=2,
            dcp_group_ranks=(0, 1),
            attn_tp_group_ranks=(0, 1),
            speculative_algorithm="EAGLE",
            speculative_num_steps=2,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=3,
            index_share_for_mtp_iteration=enabled,
            decode_cuda_graph_backend="full",
        )
        args.update(overrides)
        parallel = NS(
            dcp_enabled=args["dcp_enabled"],
            attn_dcp_size=args["dcp_size"],
            attn_dcp_rank=0,
            attn_tp_size=args["attn_tp_size"],
            attn_cp_size=args["attn_cp_size"],
            dcp_comm_backend=args["dcp_comm_backend"],
            dcp_group=NS(ranks=args["dcp_group_ranks"]),
            attn_tp_group=NS(ranks=args["attn_tp_group_ranks"]),
        )
        server_args = NS(**args)
        server_args.dsa_topk_backend = "sgl-kernel"
        server_args.cuda_graph_config = NS(
            decode=NS(
                backend=args["decode_cuda_graph_backend"],
                max_bs=args["decode_cuda_graph_max_bs"],
            )
        )
        hf_config = NS(architectures=[arch], index_share_for_mtp_iteration=enabled)
        pool = NS(dsa_kv_cache_store_fp8=True, kv_cache_dim=576)
        runner = NS(
            device="cpu",
            page_size=args["page_size"],
            token_to_kv_pool=pool,
            kv_cache_dtype=torch.float8_e4m3fn,
            hisparse_coordinator=None,
            req_to_token_pool=NS(req_to_token=None),
            server_args=server_args,
            model_config=NS(
                hf_config=hf_config,
                context_len=131072,
                num_attention_heads=64,
                qk_nope_head_dim=128,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            ),
        )
        tree = ast.parse((ROOT / path).read_text())
        cls = next(
            n
            for n in tree.body
            if getattr(n, "name", "") == "DeepseekSparseAttnBackend"
        )
        init = next(n for n in cls.body if getattr(n, "name", "") == "__init__")
        for i, node in enumerate(init.body):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_validate_dsa_dcp_launch"
            ):
                init.body = init.body[1 : i + 1]  # omit super(), stop after validation
                break
        else:
            self.fail("Backend no longer invokes the launch validator")
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        ns = dict(
            torch=torch,
            get_exec=lambda: NS(
                deterministic=NS(enable_deterministic_inference=False),
                kernel=NS(
                    dsa_prefill_backend=args["dsa_prefill_impl"],
                    dsa_decode_backend=args["dsa_decode_impl"],
                ),
            ),
            get_parallel=lambda: parallel,
            get_spec=lambda: server_args,
            is_deepseek_dsa=lambda cfg: True,
            get_dsa_index_topk=lambda cfg: 2048,
            DSATopKBackend=lambda name: name,
            _is_hip=False,
            _is_hcu=True,
            envs=NS(SGLANG_DSA_FUSE_TOPK=NS(get=lambda: args["fused_topk_enabled"])),
            should_use_dsa_fused_topk=lambda seed: args["fused_topk_enabled"],
            _validate_dsa_dcp_launch=validator,
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, init], type_ignores=[])
                ),
                str(ROOT / path),
                "exec",
            ),
            ns,
        )
        with patch.dict(
            sys.modules,
            {"sglang.srt.configs.model_config": NS(is_hy_v4=model_defs["is_hy_v4"])},
        ), patch.object(torch.cuda, "get_device_capability", return_value=(9, 3)):
            ns["__init__"](NS(), runner)

    def test_backend_constructor_accepts_hy4_target_and_draft_both_modes(self):
        for arch in ("HYV4ForCausalLM", "HYV4ForCausalLMNextN"):
            for enabled in (False, True):
                with self.subTest(arch=arch, enabled=enabled):
                    self.backend_launch(arch, enabled)

    def test_backend_constructor_preserves_other_models_index_share_guard(self):
        for arch in (
            "GlmMoeDsaForCausalLM",
            "GlmMoeDsaForCausalLMNextN",
            "DeepseekV3ForCausalLM",
        ):
            self.backend_launch(arch, False)
            with self.assertRaisesRegex(ValueError, "index_share_for_mtp_iteration"):
                self.backend_launch(arch, True)

    def test_hy4_index_share_keeps_other_backend_guards(self):
        for overrides, error in (
            ({"page_size": 128}, "page size 64"),
            ({"speculative_eagle_topk": 2}, "speculative_eagle_topk=1"),
            ({"decode_cuda_graph_backend": "breakable"}, "disabled or full"),
            ({"fused_topk_enabled": True}, "fused DSA top-k"),
            ({"enable_prefill_cp": True}, "prefill CP"),
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError, error
            ):
                self.backend_launch("HYV4ForCausalLM", True, **overrides)

    def test_worker_enables_seed_only_for_shared_single_chain(self):
        init = definitions(
            "speculative/eagle_worker_v2.py",
            ["_init_dsa_index_share_state"],
            cls="EagleDraftWorker",
        )["_init_dsa_index_share_state"]
        for enabled in (False, True):
            for topk in (1, 2):
                worker = NS(
                    topk=topk,
                    draft_runner=NS(
                        model_config=NS(
                            hf_config=NS(
                                index_share_for_mtp_iteration=enabled, index_topk=4
                            )
                        )
                    ),
                )
                init(worker)
                self.assertEqual(
                    worker.index_share_for_mtp_iteration, enabled and topk == 1
                )
                self.assertEqual(
                    worker.seed_dsa_topk_from_draft_extend, enabled and topk == 1
                )

    def run_chain(self, enabled, seed, rank):
        forward = definitions(
            NEXTN,
            ["forward"],
            {
                "BumpAllocator": lambda **kw: None,
                "IndexTopKShareState": Share,
                "hyv4_attn_tp_gather": lambda x: torch.cat((x, x)),
            },
            cls="HYV4ModelNextN",
        )["forward"]
        should_run = definitions(
            "models/deepseek_common/attention_forward_methods/forward_mla.py",
            ["should_run_indexer"],
            cls="DeepseekMLAForwardMixin",
        )["should_run_indexer"]
        batch = NS(
            reuse_dsa_topk_indices=False,
            forward_mode=NS(is_idle=lambda: False, is_extend=lambda **kw: False),
            spec_info=NS(
                hidden_states=torch.ones(4, 4),
                dsa_topk_indices=seed,
                dsa_seed_topk_capture=None,
                dsa_seed_topk_select=None,
            ),
        )
        computed, observed = [], []

        def decoder(pos, h, fb, alloc, prev_topk_indices):
            if should_run(NS(skip_topk=True, is_nextn=True), prev_topk_indices):
                indices = torch.full((4, 3), len(computed) + 10, dtype=torch.int32)
                computed.append(indices)
            else:
                indices = prev_topk_indices
            observed.append(indices.clone())
            return h.chunk(2)[rank], h.chunk(2)[rank], indices

        model = NS(
            embed_tokens=lambda ids: torch.ones(len(ids), 4),
            enorm=lambda x: x,
            hnorm=lambda x: x,
            eh_proj=lambda x: x[:, :4],
            decoder=decoder,
            alt_stream=None,
            dp_attn_scattered=True,
            shared_head=NS(norm=lambda h, r: (h + r, None)),
        )
        with Share.mtp_iteration(batch, enabled=enabled, keep_carry_seed=True):
            for _ in range(3):
                output = forward(model, torch.arange(4), None, batch)
                self.assertEqual(output.shape, (4, 4))
        self.assertFalse(batch.reuse_dsa_topk_indices)
        if enabled:
            self.assertIsNone(batch.spec_info.dsa_topk_indices)
        return computed, observed

    def test_true_reuses_full_token_seed_on_both_attention_tp_ranks(self):
        seed = torch.arange(12, dtype=torch.int32).reshape(4, 3)
        for rank in (0, 1):
            computed, observed = self.run_chain(True, seed, rank)
            self.assertEqual(len(computed), 0)
            for indices in observed:
                torch.testing.assert_close(indices, seed)

    def test_true_without_seed_computes_once_then_reuses(self):
        for rank in (0, 1):
            computed, observed = self.run_chain(True, None, rank)
            self.assertEqual(len(computed), 1)
            for indices in observed:
                torch.testing.assert_close(indices, computed[0])

    def test_false_recomputes_every_step_even_with_stale_seed(self):
        for rank in (0, 1):
            computed, observed = self.run_chain(
                False, torch.zeros(4, 3, dtype=torch.int32), rank
            )
            self.assertEqual(len(computed), 3)
            self.assertEqual([int(x[0, 0]) for x in observed], [10, 11, 12])

    def test_extend_seed_selects_last_accepted_rows_and_does_not_alias(self):
        indices = torch.arange(24, dtype=torch.int32).reshape(8, 3)
        capture = torch.full((2, 3), -1, dtype=torch.int32)
        batch = NS(
            reuse_dsa_topk_indices=False,
            forward_mode=NS(is_extend=lambda **kw: True),
            spec_info=NS(
                dsa_topk_indices=None,
                dsa_seed_topk_capture=capture,
                dsa_seed_topk_select=torch.tensor([2, 6]),
            ),
        )
        state = Share.from_mtp_carry(batch)
        state.update(indices)
        state.publish()
        expected = indices[[2, 6]].clone()
        indices.fill_(-1)
        torch.testing.assert_close(capture, expected)
        self.assertIsNone(batch.spec_info.dsa_topk_indices)

    def test_failed_iteration_does_not_leak_seed_into_next_batch(self):
        batch = NS(
            reuse_dsa_topk_indices=False,
            spec_info=NS(dsa_topk_indices=torch.ones(2, 3)),
        )
        with self.assertRaisesRegex(RuntimeError, "draft failed"):
            with Share.mtp_iteration(batch, keep_carry_seed=True):
                raise RuntimeError("draft failed")
        self.assertFalse(batch.reuse_dsa_topk_indices)
        self.assertIsNone(batch.spec_info.dsa_topk_indices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
