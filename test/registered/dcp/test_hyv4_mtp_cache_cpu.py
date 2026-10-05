"""Reproduce HY4 INT8 target / virtual-page INT8 draft cache allocation on CPU."""

import ast
from contextlib import nullcontext
import math
import sys
import unittest
from enum import Enum
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from test_hyv4_dcp_cpu import ROOT, definitions


class Mode(str, Enum):
    BF16 = "bf16"
    FP8_SCALED = "fp8_scaled"
    INT8_SCALED = "int8_scaled"


is_hy = definitions("configs/model_config.py", ["_hf_arch", "is_hy_v4"])["is_hy_v4"]


def pool_class(resolver):
    """Run the real DSA constructor, replacing only allocation and hardware."""
    tree = ast.parse((ROOT / "mem_cache/memory_pool.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DSATokenToKVPool"
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]

    class Base:
        def __init__(
            self,
            size,
            page,
            dtype,
            kv,
            rope,
            layers,
            device,
            saver,
            start,
            end,
            **kwargs
        ):
            self.page_size, self.dtype = page, dtype
            self.start_layer, self.layer_num = start or 0, layers

        def _create_index_key_cache(self):
            return NS(buffer=[]) if self.use_scaled_index_k_cache else None

        def _initialize_int8_index_k_workspace(self):
            self.index_k_dequant_workspace = torch.empty(0)
            self.index_k_page_claims = torch.empty(0)

        def _finalize_allocation_log(self, size):
            pass

    ns = dict(
        torch=torch,
        MLATokenToKVPool=Base,
        IndexKCacheMode=Mode,
        resolve_index_k_cache_mode=resolver,
        _is_hip=True,
        _is_hcu=True,
        logger=NS(info=lambda *a: None),
        index_k_cache_bytes_per_token=lambda _: 132,
        get_parallel=lambda: NS(dcp_enabled=True, attn_dcp_size=2),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "DSATokenToKVPool.__init__", "exec"),
        ns,
    )
    return ns["DSATokenToKVPool"]


# Exercise real packed storage and CPU quantization without importing GPU modules.
quant = definitions(
    "layers/attention/dsa/hcu_int8_index_k_cache.py",
    [
        "create_index_k_int8_aliases",
        "_validate_quantize_inputs",
        "_quantize_and_store_index_k_int8_reference",
        "quantize_and_store_index_k_int8",
    ],
    {
        "INDEX_K_PAGE_SIZE": 64,
        "INDEX_K_HEAD_DIM": 128,
        "INDEX_K_BYTES_PER_TOKEN": 132,
        "INDEX_K_EPSILON": 1e-6,
    },
)
Cache = definitions(
    "mem_cache/index_key_cache.py", ["IndexKeyCache"], {"nullcontext": nullcontext}
)["IndexKeyCache"]
pool_methods = definitions(
    "mem_cache/memory_pool.py",
    ["set_index_k_int8_buffer", "get_index_k_cache_transfer_abi"],
    quant,
    cls="DSATokenToKVPool",
)


class PackedPool:
    set_index_k_int8_buffer = pool_methods["set_index_k_int8_buffer"]
    get_index_k_cache_transfer_abi = pool_methods["get_index_k_cache_transfer_abi"]

    def __init__(self, kv_page=128):
        self.page_size, self.index_page_size = kv_page, 64
        self.index_head_dim = self.quant_block_size = 128
        self.index_k_cache_mode = Mode.INT8_SCALED
        self.use_int8_index_k_cache = True
        self.custom_mem_pool = self.layer_transfer_counter = None
        self.indexer_layer_num = 1
        self.index_k_with_scale_buffer_dtype = torch.uint8
        self.device = "cpu"
        self.cpu_offloading_chunk_size = 128
        self.index_num_pages = (512 + 128) // 64
        self.index_key_cache = Cache(self, 512)
        self.index_k_int8_aliases = [
            quant["create_index_k_int8_aliases"](self.index_key_cache.buffer[0])
        ]

    def _get_indexer_cache_index(self, layer_id):
        return 0

    def get_state_buf_infos(self):
        return self.index_key_cache.state_buf_infos()

    def get_state_layer_ids(self):
        return [0]

    def get_index_k_with_scale_buffer(self, layer_id):
        return self.index_key_cache.get_buffer(layer_id)


class TestHYV4MTPCache(unittest.TestCase):
    def test_pool_builder_keeps_same_int8_format_for_virtual_draft(self):
        resolver = Mock(return_value=Mode.INT8_SCALED)
        Pool = pool_class(resolver)
        build = definitions(
            "mem_cache/kv_cache_configurator.py",
            ["_build_dsa_kv_pool"],
            {
                "DSATokenToKVPool": Pool,
                "get_memory": lambda: NS(enable_hisparse=False),
                "get_exec": lambda: NS(features=NS(enable_memory_saver=False)),
                "get_schedule": lambda: NS(page_size=64),
                "get_parallel": lambda: NS(attn_dcp_size=dcp),
                "get_dsa_full_indexer_layer_ids": lambda config, start, end: list(
                    range(start, end)
                ),
                "_should_elide_dsa_index_k": lambda **kwargs: False,
                "get_dsa_index_head_dim": lambda _: 128,
                "calculate_mla_kv_cache_dim": lambda **kw: 576,
                "is_hy_v4": is_hy,
                "logger": NS(info=lambda *a: None),
            },
            cls="KVCacheConfigurator",
        )["_build_dsa_kv_pool"]
        cp = NS(get_glm_dsa_cp_layer_shard_info=lambda _: (None, 1))
        with patch.dict(sys.modules, {"sglang.srt.layers.cp.utils": cp}):
            for arch, draft, dcp, expected in (
                ("HYV4ForCausalLM", False, 2, Mode.INT8_SCALED),
                ("HYV4ForCausalLMNextN", True, 2, Mode.INT8_SCALED),
                ("HYV4ForCausalLMNextN", True, 1, Mode.INT8_SCALED),
                ("GlmMoeDsaForCausalLM", False, 2, Mode.INT8_SCALED),
            ):
                with self.subTest(arch=arch, draft=draft, dcp=dcp):
                    scale = dcp if draft else 1
                    kvc = NS(
                        model_config=NS(
                            hf_config=NS(architectures=[arch]),
                            kv_lora_rank=512,
                            qk_rope_head_dim=64,
                        ),
                        layer_info=NS(
                            start_layer=0, end_layer=1, num_effective_layers=1
                        ),
                        is_draft_worker=draft,
                        loc_space_scale=scale,
                        pool_page_size=64 * scale,
                        graph_kv_padding_capacity=64 * scale,
                        kv_cache_dtype=torch.float8_e4m3fn,
                        device="cpu",
                        server_args=NS(),
                    )
                    result = build(kvc, max_total_num_tokens=256 * scale)
                    self.assertIs(result.index_k_cache_mode, expected)
                    self.assertEqual(
                        result.use_int8_index_k_cache, expected is Mode.INT8_SCALED
                    )
                    self.assertEqual(result.index_page_size, 64)
                    if draft and dcp == 2:
                        self.assertEqual(result.index_buf_size, 512)
                        self.assertEqual(result.page_size, 128)
        # Default remains unchanged: GLM's existing BF16 virtual draft works,
        # and unsupported scaled virtual pages still fail the original guard.
        kwargs = dict(
            size=512,
            page_size=128,
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
        )
        with self.assertRaisesRegex(AssertionError, "Virtual DCP draft pages"):
            Pool(**kwargs)
        # The opt-in changes layout support, never the configured cache format.
        self.assertIs(
            Pool(**kwargs, allow_int8_virtual_index_pages=True).index_k_cache_mode,
            Mode.INT8_SCALED,
        )
        resolver.return_value = Mode.FP8_SCALED
        with self.assertRaisesRegex(AssertionError, "Virtual DCP draft pages"):
            Pool(**kwargs, allow_int8_virtual_index_pages=True)
        resolver.return_value = Mode.BF16
        self.assertIs(Pool(**kwargs).index_k_cache_mode, Mode.BF16)
        self.assertIs(
            Pool(**kwargs, allow_int8_virtual_index_pages=True).index_k_cache_mode,
            Mode.BF16,
        )

    def test_budget_uses_same_draft_index_format_as_pool(self):
        cp = NS(
            get_glm_dsa_cp_layer_shard_info=lambda _: (None, 1),
            get_layer_shard_range=lambda rank, size, n: (0, n),
        )
        kv_module = NS(_should_elide_dsa_index_k=lambda **kw: False)
        cache = NS(
            index_k_cache_bytes_per_token=lambda m: 256 if m is Mode.BF16 else 132,
            index_k_workspace_bytes_per_token=lambda m: (
                256.0625 if m is Mode.INT8_SCALED else 0
            ),
        )
        resolve = Mock(return_value=Mode.INT8_SCALED)
        ns = definitions(
            "model_executor/pool_configurator.py",
            ["_compute_dsa_indexer_cell_size"],
            {
                "get_dsa_index_head_dim": lambda _: 128,
                "IndexKCacheMode": Mode,
                "resolve_index_k_cache_mode": resolve,
                "_is_hcu": True,
                "is_hcu_native_fp8_supported": lambda: False,
                "is_hy_v4": is_hy,
                "math": math,
                "get_parallel": lambda: NS(dcp_enabled=True, attn_dcp_size=2),
                "_get_dsa_indexer_cache_token_multiplier": lambda _: 2,
                **vars(cache),
            },
            cls="DefaultPoolConfigurator",
        )
        compute = ns["_compute_dsa_indexer_cell_size"]
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.layers.cp.utils": cp,
                "sglang.srt.mem_cache.kv_cache_configurator": kv_module,
            },
        ):
            for arch in ("HYV4ForCausalLM", "GlmMoeDsaForCausalLM"):
                kvc = NS(
                    model_config=NS(hf_config=NS(architectures=[arch])),
                    kv_cache_dtype=torch.float8_e4m3fn,
                    page_size=64,
                    is_draft_worker=False,
                    layer_info=NS(start_layer=0, end_layer=10),
                )
                self.assertEqual(
                    compute(None, kvc=kvc, num_layers=10),
                    math.ceil((132 * 10 + 256.0625) * 2),
                )
                expected = math.ceil((132 + 256.0625) * 2)
                self.assertEqual(
                    compute(None, kvc=kvc, num_layers=1, allocate_all_layers=True),
                    expected,
                )
                kvc.is_draft_worker = True
                kvc.layer_info.end_layer = 1
                self.assertEqual(compute(None, kvc=kvc, num_layers=1), expected)

    def test_int8_virtual_pages_store_read_and_padding(self):
        target, draft = PackedPool(64), PackedPool(128)
        # Both physical page boundaries and the final padding page.
        loc = torch.tensor([0, 63, 64, 127, 128, 255, 511, 512, 575, 576, 639])
        key = torch.linspace(-2, 2, len(loc) * 128).reshape(-1, 128).bfloat16()
        for pool in (target, draft):
            pool.set_index_k_int8_buffer(0, loc, key)
            k, scales = pool.index_k_int8_aliases[0]
            recovered = (
                k[loc // 64, loc % 64].float() * scales[loc // 64, loc % 64, None]
            )
            self.assertLess((recovered - key.float()).abs().max().item(), 0.01)
            self.assertEqual(pool.index_key_cache.buffer[0].shape, (10, 64 * 132))
        torch.testing.assert_close(
            target.index_key_cache.buffer[0], draft.index_key_cache.buffer[0]
        )
        methods = definitions(
            "layers/attention/dsa/dsa_indexer.py",
            ["_get_index_k_read_buffer", "_get_hcu_int8_paged_index_k_cache"],
            cls="Indexer",
        )
        indexer = NS(
            head_dim=128, _get_index_k_read_buffer=methods["_get_index_k_read_buffer"]
        )
        read = methods["_get_hcu_int8_paged_index_k_cache"](indexer, draft, 0)
        self.assertEqual(read.shape, (10, 64, 1, 132))
        self.assertEqual(read.data_ptr(), draft.index_key_cache.buffer[0].data_ptr())

    def test_int8_virtual_pages_move_preserves_keys_and_scales(self):
        pool = PackedPool()
        src = torch.tensor([63, 64, 127, 128, 511])
        dst = torch.tensor([256, 319, 320, 383, 384])
        key = torch.arange(5 * 128).reshape(5, 128).bfloat16()
        pool.set_index_k_int8_buffer(0, src, key)
        k, scales = pool.index_k_int8_aliases[0]
        before = pool.index_key_cache.buffer[0].clone()
        pool.index_key_cache.move(dst, src)
        torch.testing.assert_close(k[dst // 64, dst % 64], k[src // 64, src % 64])
        torch.testing.assert_close(
            scales[dst // 64, dst % 64], scales[src // 64, src % 64]
        )
        torch.testing.assert_close(pool.index_key_cache.buffer[0][:4], before[:4])

    def test_int8_virtual_pages_offload_restores_both_halves(self):
        pool = PackedPool()
        buf = pool.index_key_cache.buffer[0]
        buf.copy_(torch.arange(buf.numel()).reshape(buf.shape).to(torch.uint8))
        before = buf.clone()
        # Two nonadjacent 128-token KV pages contain four index-K pages.
        indices = torch.cat((torch.arange(128, 256), torch.arange(384, 512)))
        with patch.object(torch.cuda, "synchronize"):
            saved = pool.index_key_cache.cpu_copy(indices)
            buf.zero_()
            pool.index_key_cache.load_cpu_copy(saved, indices)
        pages = torch.tensor([2, 3, 6, 7])
        torch.testing.assert_close(buf[pages], before[pages])
        self.assertEqual(buf[[0, 1, 4, 5, 8]].count_nonzero().item(), 0)

    def test_pd_registration_reproduces_old_error_and_accepts_int8_draft(self):
        class OtherPool:
            pass

        modules = {
            "sglang.srt.disaggregation.base.conn": NS(
                StateType=NS(DSA="dsa", MAMBA="mamba")
            ),
            "sglang.srt.hardware_backend.npu.memory_pool_npu": NS(
                NPUMLATokenToKVPool=OtherPool
            ),
            "sglang.srt.mem_cache.base_swa_memory_pool": NS(BaseSWAKVPool=OtherPool),
            "sglang.srt.mem_cache.deepseek_v4_memory_pool": NS(
                DeepSeekV4TokenToKVPool=OtherPool
            ),
            "sglang.srt.mem_cache.swa_memory_pool": NS(SWAKVPool=OtherPool),
            "sglang.srt.mem_cache.memory_pool": NS(
                DSATokenToKVPool=PackedPool,
                HybridLinearKVPool=OtherPool,
                MHATokenToKVPoolMXFP8=OtherPool,
                MiniMaxSparseKVPool=OtherPool,
            ),
        }
        ns = definitions(
            "disaggregation/utils.py",
            ["setup_state_kv_args", "append_state_component"],
            {"is_npu": lambda: False},
        )
        setup = ns["setup_state_kv_args"]
        target, draft = PackedPool(64), PackedPool(128)
        args = NS()
        with patch.dict(sys.modules, modules):
            draft.index_k_cache_mode = Mode.BF16
            with self.assertRaisesRegex(
                ValueError, "Target and draft DSA index-K cache transfer ABIs differ"
            ):
                setup(args, target, draft, total_kv_layers=1)
            draft.index_k_cache_mode = Mode.INT8_SCALED
            setup(args, target, draft, total_kv_layers=1)
        # Eliding shared layers must retain global IDs and the true target
        # layer count as draft offset, independent of the compact entry count.
        compact_args = NS()
        target.index_key_cache.buffer.extend(
            [torch.zeros_like(target.index_key_cache.buffer[0]) for _ in range(2)]
        )
        target.indexer_layer_num = 3
        target.get_state_layer_ids = lambda: [0, 4, 8]
        with patch.dict(sys.modules, modules):
            setup(compact_args, target, draft, total_kv_layers=12)
        self.assertEqual(compact_args.state_layer_ids, [[0, 4, 8, 12]])
        self.assertEqual(compact_args.state_item_lens, [[8448] * 4])
        self.assertEqual(args.state_types, ["dsa"])
        self.assertEqual(args.state_layer_ids, [[0, 1]])
        self.assertEqual(args.state_item_lens, [[8448, 8448]])
        self.assertEqual(
            args.state_data_formats, [target.get_index_k_cache_transfer_abi()]
        )
        validate = definitions(
            "disaggregation/common/conn.py", ["validate_dsa_state_transfer_abi"]
        )["validate_dsa_state_transfer_abi"]
        validate(
            target.get_index_k_cache_transfer_abi(),
            args.state_data_formats[0],
            [8448],
            args.state_item_lens[0],
        )
        with self.assertRaisesRegex(RuntimeError, "page layout mismatch"):
            validate(
                args.state_data_formats[0], args.state_data_formats[0], [8448], [16896]
            )

    def test_hy4_draft_kv_budget_accounts_for_replicated_dcp_slots(self):
        init = definitions(
            "model_executor/pool_configurator.py",
            ["__init__"],
            {
                "mambaish_config": lambda _: None,
                "is_deepseek_dsa": lambda _: True,
                "is_hy_v4": is_hy,
                "get_parallel": lambda: NS(dcp_enabled=True, attn_dcp_size=2),
            },
            cls="DefaultPoolConfigurator",
        )["__init__"]
        cp = NS(get_glm_dsa_layer_split_effective_num_layers=lambda kvc, n: n)
        with patch.dict(sys.modules, {"sglang.srt.layers.cp.utils": cp}):
            for arch, expected in (
                ("HYV4ForCausalLM", 8512),
                ("GlmMoeDsaForCausalLM", 7912),
            ):
                kvc = NS(
                    kv_cache_dtype_str="fp8_e4m3",
                    layer_info=NS(num_effective_layers=10),
                    model_config=NS(hf_config=NS(architectures=[arch])),
                    server_args=NS(max_total_tokens=256, enable_dsa_cache_layer_split=False),
                    is_draft_worker=False,
                    spec_algorithm=NS(
                        is_eagle=lambda: True, is_dflash_family=lambda: False
                    ),
                    spec_aux_config=NS(eagle_draft_num_layers=1),
                )
                configurator = NS(
                    _compute_cell_size=lambda *a: 6800,
                    _compute_graph_padding_overhead=lambda *a: 0,
                    _compute_hyv4_index_padding_overhead=lambda *a: 83968,
                    _compute_dsa_indexer_cell_size=lambda **kw: (
                        512 if kw.get("allocate_all_layers") else 800
                    ),
                )
                init(configurator, kvc)
                self.assertEqual(configurator._cell_size, expected)
                self.assertEqual(
                    configurator._fixed_overhead_bytes,
                    83968 if arch.startswith("HY") else 0,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
