"""Gate TP8 numerical and graph coordination regressions.

CPU: python -m pytest -q <this file>
Eight GPUs: HYV4_GATE_TP_TEST_CUDA=1 python -m pytest -q <this file>
The GPU test loads no model and captures only small gate projections.
"""

import os
import tempfile
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.layers.hyv4_gate_tp import GateTPBatch, GateTPGraphPlan
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


class _Linear:
    def __init__(self, rank, world_size, weight):
        self.tp_rank = rank
        self.tp_size = world_size
        self.reduce_results = False
        self.output_size = weight.shape[0]
        self.weight = weight.chunk(world_size, dim=1)[rank].contiguous()

    def __call__(self, x, *, skip_all_reduce):
        assert skip_all_reduce
        return torch.nn.functional.linear(x, self.weight), None


def _input(rank, rows, width):
    generator = torch.Generator().manual_seed(100 + rank)
    return torch.randn(rows, width, generator=generator)


def _cpu_worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    try:
        group = SimpleNamespace(
            cpu_group=dist.group.WORLD,
            device_group=dist.group.WORLD,
            world_size=world_size,
            rank_in_group=rank,
        )
        generator = torch.Generator().manual_seed(42)
        weight = torch.randn(13, 32, generator=generator)
        linear = _Linear(rank, world_size, weight)
        for lengths in (
            [0, 1, 0, 3, 0, 0, 2, 0],
            [0] * 8,
            [2] + [0] * 7,
            [3, 0, 9, 0, 6, 0, 0, 0],
        ):  # MTP verify token rows
            x = _input(rank, lengths[rank], 32)
            batch = GateTPBatch.create(x.shape[0], group)
            actual = batch.project(x, linear)
            torch.testing.assert_close(actual, torch.nn.functional.linear(x, weight))

        # One sparse peer forces all ranks onto the same captured variant.
        plan = GateTPGraphPlan.synchronize(
            rank % 4, True, "sparse" if rank == 5 else "dense", group
        )
        assert plan == GateTPGraphPlan(3, True, "sparse")
        # One ineligible peer (EXTEND, out-of-range, embedding override, ...)
        # forces every peer back to eager, even those with an empty batch.
        plan = GateTPGraphPlan.synchronize(0, rank != 6, None, group)
        assert plan == GateTPGraphPlan(0, False, None)
        plan = GateTPGraphPlan.synchronize(0, True, "dense", group)
        assert plan == GateTPGraphPlan(0, True, "dense")
    finally:
        dist.destroy_process_group()


def _gpu_worker(rank, world_size, rendezvous):
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        comm = PyNcclCommunicator(dist.group.WORLD, device=rank)
        assert comm.available
        group = SimpleNamespace(
            cpu_group=dist.group.WORLD,
            pynccl_comm=comm,
            world_size=world_size,
            rank_in_group=rank,
        )
        generator = torch.Generator().manual_seed(42)
        weight = torch.randn(64, 128, generator=generator).cuda().bfloat16()
        linear = _Linear(rank, world_size, weight)
        # Model autotune precedes the runner's graph_capture() context.
        # The gate must enable PyNCCL for these capture-layout warmups too.
        warmup = torch.zeros(1, 128, device=rank, dtype=torch.bfloat16)
        GateTPBatch.create(1, group, capture=True).project(warmup, linear)
        torch.cuda.synchronize()
        assert comm.disabled
        captures = {}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with comm.change_state(enable=True):
            for bucket in (1, 2, 3, 4, 6, 12):
                x = torch.zeros(bucket, 128, device=rank, dtype=torch.bfloat16)
                batch = GateTPBatch.create(bucket, group, capture=True)
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        batch.project(x, linear)
                stream.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = batch.project(x, linear)
                captures[bucket] = (graph, x, output, batch)

            # Change bucket, then revisit it with different data/empty ranks.
            for lengths in (
                [1] + [0] * 7,
                [0, 3, 0, 1, 0, 2, 0, 0],
                [0] * 8,
                [0, 2, 0, 0, 0, 0, 1, 0],
                [1] * 8,
                [3, 0, 9, 0, 6, 0, 0, 0],
                [0] * 8,
                [3] + [0] * 7,
            ):
                plan = GateTPGraphPlan.synchronize(lengths[rank], True, None, group)
                bucket = next(n for n in (1, 2, 3, 4, 6, 12) if n >= plan.batch_size)
                graph, static_x, output, _ = captures[bucket]
                x = _input(rank, lengths[rank], 128).cuda().bfloat16()
                static_x.zero_()
                static_x[: len(x)].copy_(x)
                graph.replay()
                torch.cuda.synchronize()
                # Match the eager TP accumulation: BF16 partial GEMMs, FP32 sum.
                expected = sum(
                    torch.nn.functional.linear(a, b).float()
                    for a, b in zip(x.chunk(8, dim=1), weight.chunk(8, dim=1))
                ).bfloat16()
                torch.testing.assert_close(output[: len(x)], expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    output[len(x) :], torch.zeros_like(output[len(x) :])
                )
    finally:
        dist.destroy_process_group()


def test_eight_rank_eager_and_graph_plan():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_cpu_worker, args=(8, "file://" + directory + "/rdzv"), nprocs=8)


def test_capture_layout_does_not_collect_cpu_counts():
    group = SimpleNamespace(world_size=8, rank_in_group=0)
    with patch.object(dist, "all_gather", side_effect=AssertionError("CPU in capture")):
        batch = GateTPBatch.create(4, group, capture=True)
    assert batch.local_tokens == batch.padded_tokens == 4
    with pytest.raises(ValueError, match="positive bucket"):
        GateTPBatch.create(0, group, capture=True)


def test_graph_runner_coordinates_fallback_and_bucket():
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    runner = object.__new__(DecodeCudaGraphRunner)
    runner.hyv4_gate_tp = True
    runner.require_mlp_tp_gather = False
    runner.dsa_dual_graph = True
    runner.dsa_index_topk = 8
    runner.model_runner = SimpleNamespace(tp_group=object())
    batch = SimpleNamespace(
        batch_size=0,
        hyv4_gate_tp_graph_plan=None,
        seq_lens_cpu=torch.tensor([]),
        forward_mode=SimpleNamespace(is_cuda_graph=lambda: True),
    )
    runner.can_run_graph = lambda fb: runner._decode_graph_batch_size(fb) <= 8
    plan = GateTPGraphPlan(5, True, "sparse")
    with patch.object(GateTPGraphPlan, "synchronize", return_value=plan):
        assert runner.prepare_hyv4_gate_graph(batch)
    assert (
        runner._pad_to_bucket(runner._decode_graph_batch_size(batch), [1, 2, 4, 8]) == 8
    )
    assert runner._resolve_dsa_variant(batch) == "sparse"
    # A stale plan must not make the next local eligibility decision pass.
    batch.batch_size = 16
    with patch.object(
        GateTPGraphPlan, "synchronize", return_value=GateTPGraphPlan(16, False, None)
    ) as sync:
        assert not runner.prepare_hyv4_gate_graph(batch)
        assert sync.call_args.args[1] is False
    # Draft subclasses do not execute the parent runner's __init__.
    del runner.hyv4_gate_tp
    batch.hyv4_gate_tp_graph_plan = None
    assert runner._decode_graph_batch_size(batch) == 16


def test_gate_server_args_graph_modes():
    from sglang.srt.server_args import ServerArgs

    args = SimpleNamespace(
        hyv4_linear_gate_tp_size=8,
        enable_dp_attention=True,
        dp_size=32,
        tp_size=32,
        pp_size=1,
        attn_cp_size=1,
        nnodes=4,
        disaggregation_mode="decode",
        enable_torch_compile=False,
        disable_cuda_graph_padding=False,
        speculative_algorithm=None,
        enable_two_batch_overlap=False,
        enable_pdmux=False,
        ep_join_mode=None,
        enable_lora=False,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend="full"),
            prefill=SimpleNamespace(backend="disabled"),
        ),
    )
    ServerArgs._check_hyv4_gate_tp(args)
    args.cuda_graph_config.decode.backend = "disabled"
    ServerArgs._check_hyv4_gate_tp(args)
    args.cuda_graph_config.decode.backend = "breakable"
    with pytest.raises(ValueError, match="full decode"):
        ServerArgs._check_hyv4_gate_tp(args)
    args.cuda_graph_config.decode.backend = "full"
    args.disable_cuda_graph_padding = True
    with pytest.raises(ValueError, match="batch padding"):
        ServerArgs._check_hyv4_gate_tp(args)


@pytest.mark.skipif(
    os.environ.get("HYV4_GATE_TP_TEST_CUDA") != "1",
    reason="requires an explicitly enabled eight-GPU test",
)
def test_eight_gpu_capture_replay():
    assert torch.cuda.device_count() >= 8
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_gpu_worker, args=(8, "file://" + directory + "/rdzv"), nprocs=8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
