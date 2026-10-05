"""HY4 DCP page-tail / relocated retraction regressions, using production code."""

import math
import sys
import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from test_hyv4_dcp_cpu import definitions, hy4
from test_hyv4_mtp_cache_cpu import Mode, PackedPool, is_hy, pool_class, quant

mla = definitions(
    "mem_cache/memory_pool.py",
    ["get_cpu_copy", "load_cpu_copy"],
    {"current_platform": NS(synchronize=lambda: None)},
    cls="MLATokenToKVPool",
)
paged = definitions(
    "mem_cache/allocator/paged.py",
    ["alloc", "clear"],
    cls="PagedTokenToKVPoolAllocator",
)


class BaseAllocator:
    alloc = paged["alloc"]
    clear = paged["clear"]

    def __init__(self, size, page_size, kvcache, **kwargs):
        self.size, self.page_size, self._kvcache = size, page_size, kvcache
        self.num_pages, self.device = size // page_size, "cpu"
        self.need_sort = self.debug_mode = False
        self.clear()


Allocator = definitions(
    "mem_cache/allocator/hyv4.py",
    ["HYV4DCPAllocator"],
    {
        "PagedTokenToKVPoolAllocator": BaseAllocator,
        "MLATokenToKVPool": NS(**mla),
        "current_platform": NS(synchronize=lambda: None),
    },
)["HYV4DCPAllocator"]


def pool(draft=False, packed=True):
    p = PackedPool(128 if draft else 64)
    p.size = 512 if draft else 256
    p.layer_num = 2
    p.kv_buffer = [
        torch.zeros((p.size + p.page_size, 1, 8), dtype=torch.uint8) for _ in range(2)
    ]
    if packed:
        p.index_key_cache.buffer.append(torch.zeros_like(p.index_key_cache.buffer[0]))
    else:
        p.index_key_cache = None
        p.index_k_buffer = [
            torch.zeros((10, 64, 1, 128), dtype=torch.bfloat16) for _ in range(2)
        ]
    return p


class TestHYV4Retraction(unittest.TestCase):
    def test_allocator_last_page_fits_all_index_storages(self):
        for draft in (False, True):
            for mode in (Mode.INT8_SCALED, Mode.BF16):
                with self.subTest(draft=draft, mode=mode):
                    Pool = pool_class(lambda *args: mode)
                    p = Pool(
                        size=512 if draft else 256,
                        page_size=128 if draft else 64,
                        kv_lora_rank=512,
                        dtype=torch.float8_e4m3fn,
                        qk_rope_head_dim=64,
                        layer_num=1,
                        device="cpu",
                        index_head_dim=128,
                        enable_memory_saver=False,
                        kv_cache_dim=576,
                        index_page_size=64,
                        index_buf_size=512,
                        allow_int8_virtual_index_pages=True,
                        index_padding_capacity=128,
                    )
                    self.assertEqual(p.index_num_pages, 10)
                    alloc = Allocator(size=512, page_size=128, kvcache=p, dcp_rank=0)
                    loc = alloc.alloc(512)
                    self.assertEqual((loc.min().item(), loc.max().item()), (128, 639))
                    # Execute the real workspace / BF16 allocation functions.
                    ns = definitions(
                        "mem_cache/memory_pool.py",
                        [
                            "_create_index_k_buffer",
                            "_initialize_int8_index_k_workspace",
                        ],
                        {
                            "nullcontext": nullcontext,
                            "GPU_MEMORY_TYPE_KV_CACHE": "kv",
                            **quant,
                        },
                        cls="DSATokenToKVPool",
                    )
                    p.custom_mem_pool = None
                    p.device = "cpu"
                    p.memory_saver_adapter = NS(region=lambda *a: nullcontext())
                    if mode == Mode.INT8_SCALED:
                        packed_pool = PackedPool()
                        p.index_key_cache = packed_pool.index_key_cache
                        ns["_initialize_int8_index_k_workspace"](p)
                        self.assertEqual(p.index_k_dequant_workspace.shape[0], 10)
                        self.assertEqual(p.index_k_page_claims.numel(), 10)
                        p.index_k_dequant_workspace[loc // 64, loc % 64] = 1
                        p.index_k_page_claims[loc // 64] = 1
                        keys = torch.randn(len(loc), 128).bfloat16()
                        packed_pool.set_index_k_int8_buffer(0, loc, keys)
                    else:
                        ns["_create_index_k_buffer"](p)
                        p.index_k_buffer[0][loc // 64, loc % 64] = 1
                        self.assertEqual(
                            p.index_k_buffer[0][9].count_nonzero(), 64 * 128
                        )

    def test_backup_relocates_target_index_and_draft_without_neighbor_writes(self):
        for rank in (0, 1):
            for mtp in (False, True):
                for packed in (False, True):
                    for length in (0, 1, 63, 64, 65, 127, 128, 129, 255, 256):
                        with self.subTest(
                            rank=rank, mtp=mtp, packed=packed, length=length
                        ):
                            target, draft = pool(packed=packed), pool(True, packed)
                            alloc = Allocator(
                                size=512, page_size=128, kvcache=target, dcp_rank=rank
                            )
                            if mtp:
                                alloc.register_draft_pool(draft)
                            # Nonadjacent pages, including the final virtual page.
                            src = torch.cat(
                                (torch.arange(512, 640), torch.arange(256, 384))
                            )[:length]
                            dst = torch.cat(
                                (torch.arange(128, 256), torch.arange(384, 512))
                            )[:length]
                            pools = [target, draft] if mtp else [target]
                            originals = []
                            for p in pools:
                                buffers = p.kv_buffer + alloc._index_buffers(p)
                                for i, buf in enumerate(buffers):
                                    # Values vary across pages, layers, slots and channels.
                                    values = (
                                        torch.arange(buf.numel()).reshape(buf.shape)
                                        * 17
                                        + i * 13
                                    ) % 251
                                    if buf.ndim == 2:
                                        values = (
                                            values + torch.arange(10)[:, None] * 19
                                        ) % 251
                                    buf.copy_(values)
                                originals.append([b.clone() for b in buffers])
                            saved = alloc.get_cpu_copy(src)
                            # Reproduce original OOB if virtual slots address local KV.
                            if length:
                                with self.assertRaises(IndexError):
                                    mla["get_cpu_copy"](target, src)
                            for p in pools:
                                for buf in p.kv_buffer + alloc._index_buffers(p):
                                    buf.zero_()
                            alloc.load_cpu_copy(saved, dst)
                            for p, old in zip(pools, originals):
                                oldloc = (
                                    alloc._local_target_indices(src)
                                    if p is target
                                    else src
                                )
                                newloc = (
                                    alloc._local_target_indices(dst)
                                    if p is target
                                    else dst
                                )
                                for i, buf in enumerate(p.kv_buffer):
                                    expected = torch.zeros_like(buf)
                                    expected[newloc] = old[i][oldloc]
                                    torch.testing.assert_close(
                                        buf, expected, rtol=0, atol=0
                                    )
                                srcpages, dstpages = src[::64] // 64, dst[::64] // 64
                                for i, buf in enumerate(alloc._index_buffers(p)):
                                    expected = torch.zeros_like(buf)
                                    expected[dstpages] = old[p.layer_num + i][srcpages]
                                    torch.testing.assert_close(
                                        buf, expected, rtol=0, atol=0
                                    )
                            # Repeated retract / restore must be stable too.
                            again = alloc.get_cpu_copy(dst)
                            alloc.load_cpu_copy(again, src)
                            self.assertEqual(again["tokens"], length)

    def test_retraction_rejects_misaligned_layout_or_changed_pool(self):
        alloc = Allocator(size=512, page_size=128, kvcache=pool(), dcp_rank=0)
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            alloc.get_cpu_copy(torch.arange(129, 193))
        with self.assertRaisesRegex(ValueError, "virtual space"):
            alloc.register_draft_pool(pool())
        saved = alloc.get_cpu_copy(torch.arange(128, 193))
        with self.assertRaisesRegex(ValueError, "length"):
            alloc.load_cpu_copy(saved, torch.arange(256, 320))
        alloc.register_draft_pool(pool(True))
        with self.assertRaisesRegex(ValueError, "draft pool changed"):
            alloc.load_cpu_copy(saved, torch.arange(256, 321))

    def test_allocator_builder_selects_hy_only_and_registers_shared_draft(self):
        build = definitions(
            "mem_cache/kv_cache_configurator.py",
            ["_build_token_to_kv_pool_allocator"],
            {
                "get_disagg": lambda: NS(disaggregation_mode="decode"),
                "current_platform": NS(is_out_of_tree=lambda: False),
                "_is_npu": False,
                "get_memory": lambda: NS(enable_hisparse=False),
                "get_schedule": lambda: NS(page_size=64),
                "get_parallel": lambda: NS(
                    dcp_enabled=True, attn_dcp_size=2, attn_dcp_rank=1
                ),
                "PagedTokenToKVPoolAllocator": BaseAllocator,
                "is_hy_v4": is_hy,
            },
            cls="KVCacheConfigurator",
        )["_build_token_to_kv_pool_allocator"]
        modules = {
            "sglang.srt.mem_cache.allocator.hyv4": NS(HYV4DCPAllocator=Allocator)
        }
        with patch.dict(sys.modules, modules):
            for arch in ("HYV4ForCausalLM", "GlmMoeDsaForCausalLM"):
                cfg = NS(
                    model_config=NS(hf_config=NS(architectures=[arch])),
                    is_hybrid_swa=False,
                    is_draft_worker=False,
                    kv_cache_dtype=torch.uint8,
                    device="cpu",
                )
                kwargs = dict(
                    sizes=NS(max_total_num_tokens=256),
                    token_to_kv_pool=pool(),
                    is_dsv4_model=False,
                    req_to_token_pool=NS(),
                    token_to_kv_pool_allocator=None,
                )
                alloc = build(cfg, **kwargs)
                self.assertIs(
                    type(alloc), Allocator if arch.startswith("HY") else BaseAllocator
                )
                self.assertEqual(alloc.size, 512)
                self.assertEqual(alloc.page_size, 128)
                if arch.startswith("HY"):
                    self.assertEqual(alloc.dcp_rank, 1)
                    draft = pool(True)
                    cfg.is_draft_worker = True
                    kwargs.update(
                        token_to_kv_pool=draft, token_to_kv_pool_allocator=alloc
                    )
                    self.assertIs(build(cfg, **kwargs), alloc)
                    self.assertIs(alloc.draft_pool, draft)

    def test_mtp_snapshot_uses_latest_relay_and_owns_storage(self):
        for overlap in (False, True):
            alloc = Allocator(size=512, page_size=128, kvcache=pool(), dcp_rank=0)
            alloc.register_draft_pool(pool(True))
            topk_p, topk_index, hidden = (
                torch.rand(5, 1),
                torch.arange(5)[:, None],
                torch.rand(5, 16),
            )
            slots = torch.tensor([4, 1])
            relay = NS(
                topk_p_buf=topk_p, topk_index_buf=topk_index, hidden_states_buf=hidden
            )
            spec = NS(
                future_indices=slots if overlap else None,
                topk_p=topk_p[slots],
                topk_index=topk_index[slots],
                hidden_states=hidden[slots],
            )
            batch = NS(
                spec_info=spec,
                reqs=[NS(output_dsa_topk_indices=torch.tensor([999])) for _ in slots],
            )
            alloc.snapshot_mtp_retraction_state(batch, relay)
            for i, req in enumerate(batch.reqs):
                torch.testing.assert_close(req.hidden_states_tensor, hidden[slots[i]])
                torch.testing.assert_close(req.output_topk_p, topk_p[slots[i]])
                torch.testing.assert_close(req.output_topk_index, topk_index[slots[i]])
                self.assertIsNone(req.output_dsa_topk_indices)
            saved = batch.reqs[0].hidden_states_tensor.clone()
            hidden.zero_()
            spec.hidden_states.zero_()
            torch.testing.assert_close(batch.reqs[0].hidden_states_tensor, saved)

    def test_scheduler_drains_pending_result_before_retraction_only_for_hy4(self):
        for hy in (False, True):
            events = []
            update = definitions(
                "managers/scheduler.py",
                ["update_running_batch"],
                {"TEST_RETRACT": False, "logger": NS(warning=lambda *a: None)},
                cls="Scheduler",
            )["update_running_batch"]
            allocator = NS(available_size=lambda: 128)
            if hy:
                allocator.snapshot_mtp_retraction_state = lambda *a: events.append(
                    "snapshot"
                )
            batch = NS(
                check_decode_mem=lambda: False,
                batch_size=lambda: 2,
                filter_batch=lambda: events.append("filter"),
                is_empty=lambda: False,
                retract_decode=lambda *a: (events.append("retract") or ([], 0.5, [])),
                prepare_for_decode=lambda: events.append("prepare"),
            )
            scheduler = NS(
                token_to_kv_pool_allocator=allocator,
                enable_overlap=True,
                result_queue=deque([("pending", "result")]),
                last_batch="pending",
                future_map=None,
                process_batch_result=lambda *a: events.append("commit"),
                device_module=NS(synchronize=lambda: events.append("sync")),
                tree_cache=NS(req_to_token_pool=NS()),
                new_token_ratio_tracker=NS(current=0.2),
                metrics_reporter=NS(enable_metrics=False),
                server_args=NS(),
            )
            update(scheduler, batch)
            if hy:
                self.assertEqual(
                    events,
                    ["filter", "commit", "sync", "filter", "snapshot", "retract", "prepare"],
                )
                self.assertFalse(scheduler.result_queue)
                self.assertIsNone(scheduler.last_batch)
            else:
                self.assertEqual(events, ["filter", "retract", "prepare"])
                self.assertEqual(len(scheduler.result_queue), 1)
                self.assertEqual(scheduler.last_batch, "pending")

    def _run_finished_request_update(self, finished_count, active_count, finish_pending):
        events = []
        estimate = definitions(
            "managers/schedule_batch.py",
            ["_new_tokens_required_next_decode_spec_v2"],
            {
                "get_alloc_reserve_per_decode": lambda: 3,
                "ceil_align": lambda n, p: (n + p - 1) // p * p,
            },
            cls="ScheduleBatch",
        )["_new_tokens_required_next_decode_spec_v2"]
        update = definitions(
            "managers/scheduler.py",
            ["update_running_batch"],
            {"TEST_RETRACT": False},
            cls="Scheduler",
        )["update_running_batch"]
        reqs = [
            NS(done=True, kv=None, kv_committed_len=128)
            for _ in range(finished_count)
        ]
        reqs += [
            NS(done=False, kv=NS(kv_allocated_len=128), kv_committed_len=128)
            for _ in range(active_count)
        ]
        batch = NS(reqs=reqs, batch_is_full=True)
        available = [0 if finish_pending else 1024]

        def filter_batch():
            events.append("filter")
            batch.reqs = [r for r in batch.reqs if not r.done]

        def check_decode_mem():
            events.append("check")
            # Run the actual estimator, including its dereference of req.kv.
            return estimate(batch, batch.reqs, 128) <= available[0]

        def process_result(*args):
            events.append("commit")
            req = batch.reqs[0]
            req.done, req.kv = True, None
            available[0] = 1024

        batch.batch_size = lambda: len(batch.reqs)
        batch.is_empty = lambda: not batch.reqs
        batch.filter_batch = filter_batch
        batch.check_decode_mem = check_decode_mem
        batch.prepare_for_decode = lambda: events.append("prepare")
        scheduler = NS(
            token_to_kv_pool_allocator=NS(
                snapshot_mtp_retraction_state=lambda *a: self.fail(
                    "completed requests must not trigger a retraction snapshot"
                ),
            ),
            enable_overlap=True,
            result_queue=deque([("pending", "result")] if finish_pending else []),
            last_batch="pending" if finish_pending else None,
            process_batch_result=process_result,
            device_module=NS(synchronize=lambda: events.append("sync")),
            new_token_ratio_tracker=NS(decay_step=lambda: events.append("decay")),
        )
        self.assertIs(update(scheduler, batch), batch)
        self.assertFalse(batch.batch_is_full)
        self.assertEqual(len(batch.reqs), active_count - int(finish_pending))
        self.assertFalse(scheduler.result_queue)
        self.assertIsNone(scheduler.last_batch)
        return events

    def test_finished_request_is_filtered_before_memory_check(self):
        events = self._run_finished_request_update(1, 0, False)
        self.assertEqual(events, ["filter"])

    def test_mixed_batch_checks_only_live_kv(self):
        events = self._run_finished_request_update(1, 1, False)
        self.assertEqual(events, ["filter", "check", "check", "decay", "prepare"])

    def test_pending_completion_empties_batch_before_second_memory_check(self):
        events = self._run_finished_request_update(0, 1, True)
        self.assertEqual(events, ["filter", "check", "commit", "sync", "filter"])

    def test_pending_completion_frees_memory_without_retraction(self):
        events = self._run_finished_request_update(0, 2, True)
        self.assertEqual(
            events,
            ["filter", "check", "commit", "sync", "filter", "check", "decay", "prepare"],
        )

    def test_compact_pd_is_opt_in_hy_target_only(self):
        memory = NS(enable_hisparse=False, enable_hierarchical_cache=False,
                    enable_unified_cache_external_linker=False)
        flag = NS(get=lambda: enabled)
        should = definitions(
            "mem_cache/kv_cache_configurator.py",
            ["_should_elide_dsa_index_k"],
            {
                "get_memory": lambda: memory,
                "get_disagg": lambda: NS(disaggregation_mode=mode),
                "is_hy_v4": is_hy,
                "envs": NS(SGLANG_HY4_COMPACT_PD_INDEX_K=flag),
            },
        )["_should_elide_dsa_index_k"]
        for mode in ("null", "prefill", "decode"):
            for enabled in (False, True):
                for arch in ("HYV4ForCausalLM", "GlmMoeDsaForCausalLM"):
                    config = NS(architectures=[arch])
                    self.assertEqual(
                        should(is_draft_worker=False, hf_config=config),
                        mode == "null" or (enabled and arch.startswith("HY")),
                    )
                    self.assertFalse(should(is_draft_worker=True, hf_config=config))
        memory.enable_hierarchical_cache = True
        self.assertFalse(
            should(
                is_draft_worker=False, hf_config=NS(architectures=["HYV4ForCausalLM"])
            )
        )

    def test_pd_compact_mapping_includes_mtp_and_rejects_old_dense_prefill(self):
        pair = definitions(
            "disaggregation/utils.py", ["build_transfer_entry_pairs"], {"deque": deque}
        )["build_transfer_entry_pairs"]
        # Global target IDs plus draft offset=num_hidden_layers, not compact count.
        compact, dense = [0, 4, 8, 12], list(range(13))
        self.assertEqual(pair(compact, compact, 4, 4), [(0, 0), (1, 1), (2, 2), (3, 3)])
        self.assertEqual(pair(compact, dense, 4, 13), [(0, 0), (1, 4), (2, 8), (3, 12)])
        with self.assertRaisesRegex(RuntimeError, "missing a transfer entry"):
            pair(dense, compact, 13, 4)

    def test_padding_budget_matches_target_and_draft_allocations(self):
        method = definitions(
            "model_executor/pool_configurator.py",
            ["_compute_hyv4_index_padding_overhead"],
            cls="DefaultPoolConfigurator",
        )["_compute_hyv4_index_padding_overhead"]
        for dcp in (1, 2):
            for mtp in (False, True):
                for mode in (Mode.INT8_SCALED, Mode.BF16):
                    per_token = 132 if mode == Mode.INT8_SCALED else 256
                    workspace = 256.0625 if mode == Mode.INT8_SCALED else 0
                    compute = lambda **kw: math.ceil(
                        (per_token * kw["num_layers"] + workspace) * dcp
                    )
                    cfg = NS(_compute_dsa_indexer_cell_size=compute)
                    kvc = NS(
                        page_size=64,
                        is_draft_worker=False,
                        spec_algorithm=NS(is_eagle=lambda: mtp),
                        spec_aux_config=NS(eagle_draft_num_layers=1),
                    )
                    expected = 64 * dcp * (per_token * 3 + workspace)
                    if mtp:
                        expected += 64 * dcp * (per_token + workspace)
                    # Cell-level rounding is conservative by less than one physical page per pool.
                    got = method(cfg, kvc, 3)
                    self.assertGreaterEqual(got, expected)
                    self.assertLess(got - expected, 128)

    def test_sink_inplace_matches_old_fp32_formula_without_modifying_input(self):
        torch.manual_seed(9)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for n in (0, 1, 9):
                output = torch.randn(n, 4, 16, dtype=dtype)[..., ::2]
                original = output.clone()
                counts = torch.arange(n) % 3
                lse = torch.randn(n, 4) * 40
                sink = torch.randn(4) * 20
                for base2 in (False, True):
                    actual, merged = hy4.apply_hyv4_sink(
                        output, lse, sink, counts, dcp_size=2, lse_base2=base2
                    )
                    valid = counts[:, None] > 0
                    local_lse = torch.where(
                        valid, lse.float() * (math.log(2) if base2 else 1), -torch.inf
                    )
                    reference_lse = torch.logaddexp(
                        local_lse, sink.float() - math.log(2)
                    )
                    clean = torch.where(valid[..., None], output, 0.0)
                    expected = (
                        clean.float() * torch.exp(local_lse - reference_lse)[..., None]
                    ).to(dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(output, original, rtol=0, atol=0)
                    torch.testing.assert_close(
                        merged,
                        reference_lse * (math.log2(math.e) if base2 else 1),
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
