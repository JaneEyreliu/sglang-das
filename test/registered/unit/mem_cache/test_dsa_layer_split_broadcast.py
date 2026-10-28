"""Multi-GPU integration test for LayerSplitDSATokenToKVPool owner-broadcast.

Spawns ``world`` processes forming a single attention-CP group, builds a tiny
``LayerSplitDSATokenToKVPool`` on each rank, writes a rank-distinct value into
every owned layer, then verifies that reading ANY layer (owned or not) returns
the *owning* rank's bytes -- i.e. the owner-broadcast in
``_get_broadcastable_kv_buffer`` / ``prefetch_kv_buffer`` surfaces correct
contents. Also exercises the DSA indexer broadcast and the async prefetch path.

Registered as a base-c 4-gpu-b200 unit test; uses up to 4 GPUs and skips when
fewer than 2 are visible. Run directly on 2+ GPUs:
    CUDA_VISIBLE_DEVICES=0,1 python -m pytest \
        test/registered/unit/mem_cache/test_dsa_layer_split_broadcast.py
"""

import os
import unittest

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-c", runner_config="4-gpu-b200")

LAYER_NUM = 4
PAGE_SIZE = 64
KV_LORA_RANK = 512
QK_ROPE = 64
INDEX_HEAD_DIM = 128
SIZE = PAGE_SIZE * 3  # a few pages
PORT = 29711


def _run(rank: int, world: int, port: int, compact_index: bool = False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    torch.cuda.set_device(rank)

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import get_parallel

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world,
        attention_context_model_parallel_size=world,
    )

    from sglang.srt.mem_cache.dsa_cache_layer_split import (
        LayerSplitDSATokenToKVPool,
    )

    cp_rank = get_parallel().attn_cp_rank
    cp_size = get_parallel().attn_cp_size
    assert cp_size == world

    pool = LayerSplitDSATokenToKVPool(
        SIZE,
        page_size=PAGE_SIZE,
        kv_lora_rank=KV_LORA_RANK,
        dtype=torch.bfloat16,
        qk_rope_head_dim=QK_ROPE,
        layer_num=LAYER_NUM,
        device=f"cuda:{rank}",
        index_head_dim=INDEX_HEAD_DIM,
        enable_memory_saver=False,
        kv_cache_dim=KV_LORA_RANK + QK_ROPE,
        layer_shard_rank=cp_rank,
        layer_shard_size=cp_size,
    )

    # Owner writes a layer-distinct constant into each owned kv_buffer layer.
    for layer_id in range(LAYER_NUM):
        if pool._is_layer_owned(layer_id):
            pool.kv_buffer[layer_id].fill_(float(layer_id + 1))

    torch.cuda.synchronize()
    torch.distributed.barrier()

    # Every rank reads every layer; broadcast must surface the owner's value.
    ok = True
    for layer_id in range(LAYER_NUM):
        buf = pool._get_broadcastable_kv_buffer(layer_id)
        expected = float(layer_id + 1)
        got = buf.float().mean().item()
        if abs(got - expected) > 1e-3:
            print(f"[rank {rank}] layer {layer_id}: expected {expected}, got {got}")
            ok = False
    assert ok, f"rank {rank} read stale/incorrect broadcast contents"

    # Indexer buffer owner-broadcast: owner writes a layer-distinct value, then
    # every rank must read it back for every layer.
    for layer_id in range(LAYER_NUM):
        # Use the format-independent storage: HCU BF16 pools intentionally
        # have no scaled Index-K accessor, while FP8 pools use packed bytes.
        store_buf = pool.index_key_cache.buffer[
            pool._get_indexer_cache_index(layer_id)
        ]
        if pool._is_layer_owned(layer_id):
            store_buf.fill_(layer_id + 10)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    for layer_id in range(LAYER_NUM):
        # invalidate any cached remote copy so the read forces a fresh broadcast
        pool.invalidate_index_buffer_for_layer(layer_id)
        buf = pool._get_broadcastable_index_buffer(layer_id)
        expected = layer_id + 10
        got = buf.float().mean().item()
        if abs(got - expected) > 1e-3:
            print(f"[rank {rank}] index layer {layer_id}: exp {expected}, got {got}")
            ok = False
    assert ok, f"rank {rank} read stale/incorrect index broadcast contents"

    # Async prefetch path: prefetch layer, then read must return owner value.
    for layer_id in range(LAYER_NUM):
        pool.remote_kv_layer_id = None  # force a fresh broadcast
        pool.prefetch_kv_buffer(layer_id)
        buf = pool._get_broadcastable_kv_buffer(layer_id)
        got = buf.float().mean().item()
        if abs(got - float(layer_id + 1)) > 1e-3:
            print(f"[rank {rank}] prefetch layer {layer_id}: got {got}")
            ok = False
    assert ok, f"rank {rank} prefetch path returned incorrect contents"

    if compact_index:
        from sglang.srt.mem_cache.dsa_cache_layer_split import MainKVPagePlan

        class BatchMarker:
            pass

        os.environ["SGLANG_HCU_COMPACT_INDEX_BROADCAST"] = "1"
        cache = pool.index_key_cache
        for history in ([], [1], [1, 2], [2], [], [1, 2]):
            marker = BatchMarker()
            page_ids = torch.tensor(history, dtype=torch.long, device=f"cuda:{rank}")
            all_pages = torch.tensor(
                sorted(set(history) | {3}), dtype=torch.long, device=f"cuda:{rank}"
            )
            pool.configure_main_kv_page_plan(
                MainKVPagePlan(page_ids, all_pages), marker
            )
            for layer_id in range(LAYER_NUM):
                cache.finalize_pending(set_remote_layer_id=False)
                cache.remote_layer_id = None
                cache.remote_buffer.fill_(17)
                cache.remote_buffer[0].zero_()
                reference = (
                    torch.arange(
                        cache.remote_buffer.numel(),
                        dtype=torch.int64,
                        device=cache.remote_buffer.device,
                    )
                    .remainder(251)
                    .reshape(cache.remote_buffer.shape)
                    .to(cache.remote_buffer.dtype)
                )
                if pool._is_layer_owned(layer_id):
                    cache.buffer[pool._get_indexer_cache_index(layer_id)].copy_(
                        reference
                    )
                expected = cache.remote_buffer.clone()
                expected[page_ids] = reference[page_ids]
                pool.prefetch_index_buffer(layer_id)
                actual = pool._get_broadcastable_index_buffer(layer_id)
                assert torch.equal(actual, expected), (
                    rank, layer_id, history
                )
        pool.configure_main_kv_page_plan(None, BatchMarker())
        os.environ["SGLANG_HCU_COMPACT_INDEX_BROADCAST"] = "0"

    print(f"[rank {rank}] OK: all {LAYER_NUM} layers read correct owner contents")
    torch.distributed.barrier()


class TestLayerSplitDSABroadcast(CustomTestCase):
    def test_owner_broadcast(self):
        world = min(4, torch.cuda.device_count())
        if world < 2:
            self.skipTest("LayerSplitDSATokenToKVPool broadcast test needs >= 2 GPUs")
        mp.spawn(_run, args=(world, PORT), nprocs=world, join=True)

    def test_compact_index_history_broadcast(self):
        world = min(4, torch.cuda.device_count())
        if world < 2:
            self.skipTest("LayerSplit compact index broadcast test needs >= 2 GPUs")
        mp.spawn(_run, args=(world, PORT + 1, True), nprocs=world, join=True)


if __name__ == "__main__":
    unittest.main()
