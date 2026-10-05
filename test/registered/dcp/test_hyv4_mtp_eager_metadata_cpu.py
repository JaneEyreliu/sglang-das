"""Exercise draft pre-planning and eager dispatch with CPU attention stand-ins.

The real worker, plan marker and eager runner methods are loaded without the
serving/GPU dependencies. Padding and attention kernels are CPU stand-ins.
"""

import ast
import contextlib
import copy
import importlib.util
import sys
import unittest
from enum import IntEnum, auto
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch
from test_hyv4_dcp_cpu import ROOT, definitions


def load_class(path, name, members, namespace=None):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if getattr(n, "name", None) == name)
    cls.bases = []
    cls.decorator_list = []
    cls.body = [
        n
        for n in cls.body
        if getattr(n, "name", None) in members
        or (isinstance(n, ast.AnnAssign) and n.target.id in members)
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ns = namespace or {}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), ns)
    return ns[name]


spec = importlib.util.spec_from_file_location(
    "eager_metadata_forward_context", ROOT / "model_executor/forward_context.py"
)
ctx = importlib.util.module_from_spec(spec)
# dataclasses resolves postponed annotations through sys.modules.
sys.modules[spec.name] = ctx
spec.loader.exec_module(ctx)

Batch = load_class(
    "model_executor/forward_batch_info.py",
    "ForwardBatch",
    {"mark_forward_metadata_ready", "needs_forward_metadata_init"},
)
Eager = load_class(
    "model_executor/runner/eager_runner.py",
    "EagerRunner",
    {"_resolve_decode_pdmux", "_execute_decode"},
    dict(
        contextlib=contextlib,
        get_attn_backend=ctx.get_attn_backend,
        forward_context=ctx.forward_context,
        ForwardContext=ctx.ForwardContext,
        device_timer_ctx=lambda *a: contextlib.nullcontext(),
    ),
)


class StepBackend:
    def __init__(self, *args, **kwargs):
        self.planned_rows = []

    def init_forward_metadata(self, batch):
        self.planned_rows.append(batch.batch_size)
        self.page_table = torch.zeros(batch.batch_size, 8, dtype=torch.int32)
        self.cache_seqlens = batch.seq_lens.clone()


MultiStep = load_class(
    "layers/attention/dsa_backend.py",
    "DeepseekSparseAttnMultiStepBackend",
    {"supports_eager_metadata_replan", "__init__", "init_forward_metadata"},
    dict(DeepseekSparseAttnBackend=StepBackend),
)


class TestHYV4MTPEagerMetadata(unittest.TestCase):
    def run_draft(self, bs, num_steps=2, missing_seed=True, graph_eligible=True):
        batch = Batch()
        batch.batch_size = bs
        batch.input_ids = torch.zeros(bs, dtype=torch.long)
        batch.seq_lens = torch.ones(bs, dtype=torch.int32)
        batch.forward_mode = NS(is_idle=lambda: False)
        batch.spec_info = NS(dsa_topk_indices=None if missing_seed else object())
        backend = MultiStep(None, topk=1, speculative_num_steps=num_steps)
        default_backend = NS(init_forward_metadata=Mock())
        seen = []

        def model_forward(input_ids, positions, fb):
            active = ctx.get_attn_backend()
            # The real failure was page-table rows != Q / top-k rows.
            self.assertEqual(active.page_table.shape[0], input_ids.shape[0])
            self.assertEqual(active.cache_seqlens.shape[0], fb.batch_size)
            seen.append(active)

        eager = Eager()
        eager.enable_pdmux = False
        eager.load_batch = lambda fb, _: copy.copy(fb)
        eager.model_runner = NS(
            attn_backend=default_backend,
            model=NS(forward=model_forward),
            _pp_kwargs=lambda _: {},
            device_timer=None,
        )

        def draft_forward(fb):
            for step in backend.attn_backends:
                # ModelRunner pads the live batch after the worker pre-plan.
                # Keep the raw plan record, as ForwardBatch copies do.
                padded = copy.copy(fb)
                padded.batch_size = ((bs + 1) // 2) * 2
                padded.input_ids = torch.zeros(padded.batch_size, dtype=torch.long)
                padded.seq_lens = torch.ones(padded.batch_size, dtype=torch.int32)
                padded.positions = torch.zeros_like(padded.input_ids)
                with ctx.forward_context(ctx.ForwardContext(attn_backend=step)):
                    eager._execute_decode(padded)
            return None, None, None, None

        draft = definitions(
            "speculative/eagle_worker_v2.py",
            ["draft"],
            cls="EagleDraftWorker",
            namespace=dict(
                contextlib=contextlib,
                prepare_for_draft=lambda *a: (batch, graph_eligible),
                build_eagle_verify_input=lambda *a, **kw: "verified",
            ),
        )["draft"]
        graph = NS(execute=Mock(return_value=(None, None, None, None)))
        worker = NS(
            req_to_token_pool=None,
            cuda_graph_runner=graph,
            draft_runner=NS(canary_manager=None),
            topk=1,
            speculative_num_steps=num_steps,
            speculative_num_draft_tokens=num_steps + 1,
            seed_dsa_topk_from_draft_extend=True,
            draft_attn_backend=backend,
            draft_forward=draft_forward,
            target_worker=None,
            tree_mask_mode=None,
            device="cpu",
        )
        self.assertEqual(draft(worker, batch), "verified")
        default_backend.init_forward_metadata.assert_not_called()
        return backend, graph, seen

    def test_missing_seed_warmup_replans_one_to_two_rows(self):
        backend, graph, seen = self.run_draft(1)
        graph.execute.assert_not_called()
        self.assertEqual(backend.attn_backends[0].planned_rows, [1, 2])
        self.assertEqual(seen, backend.attn_backends)

    def test_each_step_replans_odd_batch(self):
        backend, graph, seen = self.run_draft(3, num_steps=3)
        for step in backend.attn_backends:
            self.assertEqual(step.planned_rows, [3, 4])
        self.assertEqual(seen, backend.attn_backends)

    def test_aligned_batch_keeps_preplanned_metadata(self):
        backend, _, _ = self.run_draft(2)
        self.assertEqual(backend.attn_backends[0].planned_rows, [2])

    def test_graph_ineligible_with_seed_also_replans(self):
        backend, graph, _ = self.run_draft(
            5, missing_seed=False, graph_eligible=False
        )
        graph.execute.assert_not_called()
        self.assertEqual(backend.attn_backends[0].planned_rows, [5, 6])

    def test_graph_replay_with_seed_is_unchanged(self):
        backend, graph, seen = self.run_draft(1, missing_seed=False)
        graph.execute.assert_called_once()
        self.assertEqual(seen, [])
        self.assertEqual(backend.attn_backends[0].planned_rows, [])

    def test_non_opted_in_plan_does_not_rebuild(self):
        batch = Batch()
        batch.batch_size = 1
        batch.input_ids = torch.zeros(1)
        batch.mark_forward_metadata_ready()
        batch.batch_size = 2
        batch.input_ids = torch.zeros(2)
        self.assertFalse(batch.needs_forward_metadata_init())

    def test_pdmux_uses_its_decode_backend(self):
        eager = Eager()
        eager.enable_pdmux = True
        eager.model_runner = NS(decode_attn_backend=object())
        backend, scope = eager._resolve_decode_pdmux()
        self.assertIs(backend, eager.model_runner.decode_attn_backend)
        with scope:
            self.assertIs(ctx.get_attn_backend(), backend)


class TestEagerDCPVerify(unittest.TestCase):
    def run_extend(self, mode_name, dcp_size, metadata_ready=False):
        mode = definitions(
            "model_executor/forward_batch_info.py",
            ["ForwardMode"],
            {"IntEnum": IntEnum, "auto": auto},
        )["ForwardMode"][mode_name]
        verify = mode.is_target_verify()
        batch = Batch()
        batch.forward_mode = mode
        batch.batch_size = 1
        batch.forward_metadata_ready = False
        batch.input_ids = torch.tensor([1, 2])
        batch.positions = torch.tensor([4, 5])
        batch.seq_lens = torch.tensor([6], dtype=torch.int32)
        batch.seq_lens_sum = 6
        batch.req_pool_indices = torch.tensor([0])
        # Verify intentionally has no ordinary extend lengths (ForwardBatch.init_new).
        batch.extend_prefix_lens = None if verify else torch.tensor([4])
        batch.extend_prefix_lens_cpu = None if verify else [4]
        batch.extend_seq_lens = None if verify else torch.tensor([2])
        batch.attn_dcp_metadata = None
        if metadata_ready:
            batch.mark_forward_metadata_ready()
        req_to_token = torch.arange(8).reshape(1, 8)
        prepared_metadata = object()
        events = []

        def prepare_dcp(*args):
            # Same input contract as the real ordinary-extend DCP planner.
            torch.cumsum(args[1], dim=0)
            events.append("dcp")
            return prepared_metadata

        planner = Mock(side_effect=prepare_dcp)
        backend = NS(
            init_forward_metadata=Mock(
                side_effect=lambda fb: events.append("attention")
            )
        )
        model = NS(
            prepare_context_parallel_metadata_for_dcp=planner,
            prepare_forward_batch=Mock(
                side_effect=lambda fb: events.append("model_metadata")
            ),
            forward=Mock(
                side_effect=lambda *a, **kw: events.append("forward") or "output"
            ),
        )
        eager_cls = load_class(
            "model_executor/runner/eager_runner.py",
            "EagerRunner",
            {"execute", "_execute_extend"},
            dict(
                is_cp_v2_active=lambda fb: False,
                get_req_to_token_pool=lambda: NS(req_to_token=req_to_token),
                get_token_to_kv_pool=lambda: NS(get_kv_buffer_shape=lambda: [(8, 2)]),
                create_chunked_prefix_cache_kv_indices=object(),
                device_timer_ctx=lambda *a: contextlib.nullcontext(),
                _is_hip=False,
            ),
        )
        eager = eager_cls()
        eager.enable_pdmux = False
        eager.load_batch = lambda fb, _: fb
        eager.model_runner = NS(
            model=model,
            attn_backend=backend,
            ps=NS(attn_dcp_size=dcp_size),
            _extend_forward_kwargs=lambda *a: {},
            kv_cache_dtype=torch.float32,
            device="cpu",
            device_timer=None,
            prefill_cuda_graph_runner=None,
        )
        self.assertEqual(eager.execute(batch), "output")
        needs_init = not metadata_ready or verify
        needs_dcp = needs_init and dcp_size > 1 and not verify
        if needs_dcp:
            planner.assert_called_once()
            self.assertIs(planner.call_args.args[1], batch.extend_prefix_lens)
            self.assertIs(planner.call_args.args[5], req_to_token)
            self.assertIs(batch.attn_dcp_metadata, prepared_metadata)
        else:
            planner.assert_not_called()
            self.assertIsNone(batch.attn_dcp_metadata)
        expected = ["dcp"] if needs_dcp else []
        if needs_init:
            expected += ["model_metadata", "attention"]
            backend.init_forward_metadata.assert_called_once_with(batch)
        else:
            backend.init_forward_metadata.assert_not_called()
        self.assertEqual(events, expected + ["forward"])

    def test_verify_without_extend_lengths_still_initializes_attention(self):
        for dcp in (1, 2, 4):
            for ready in (False, True):
                with self.subTest(dcp=dcp, metadata_ready=ready):
                    self.run_extend("TARGET_VERIFY", dcp, ready)

    def test_other_extend_modes_preserve_dcp_planning(self):
        for mode in (
            "EXTEND", "MIXED", "DRAFT_EXTEND_V2", "SPLIT_PREFILL", "DLLM_EXTEND"
        ):
            for dcp in (1, 2, 4):
                for ready in (False, True):
                    with self.subTest(mode=mode, dcp=dcp, metadata_ready=ready):
                        self.run_extend(mode, dcp, ready)


if __name__ == "__main__":
    unittest.main()
