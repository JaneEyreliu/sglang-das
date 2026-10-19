"""HYV4 linear_gate CP sharding and two-slot weight prefetch."""

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
_RUNTIME: Optional["GateCPShardRuntime"] = None


def create_cp_sharded_gate_weight(
    full_weight: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    expected_output_size: Optional[int] = None,
    expected_input_size: Optional[int] = None,
) -> torch.Tensor:
    """Return one contiguous output-row shard of a full gate matrix."""
    if full_weight.ndim != 2:
        raise ValueError(f"HYV4 gate weight must be 2-D, got {full_weight.ndim}-D")
    if cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"invalid CP rank/size: rank={cp_rank}, size={cp_size}")
    output_size, input_size = full_weight.shape
    if expected_output_size is not None and output_size != expected_output_size:
        raise ValueError(
            f"HYV4 gate output size {output_size} != expected {expected_output_size}"
        )
    if expected_input_size is not None and input_size != expected_input_size:
        raise ValueError(
            f"HYV4 gate input size {input_size} != expected {expected_input_size}"
        )
    if output_size % cp_size:
        raise ValueError(
            f"HYV4 gate output size {output_size} is not divisible by CP size {cp_size}"
        )
    shard_size = output_size // cp_size
    return full_weight.narrow(0, cp_rank * shard_size, shard_size).contiguous()


def _init_gate_cp_group():
    from sglang.srt.distributed.parallel_state import (
        get_attn_cp_group,
        get_world_group,
        init_model_parallel_group,
    )
    from sglang.srt.runtime_context import get_parallel

    cp_group = get_attn_cp_group()
    attn_tp_size = get_parallel().attn_tp_size
    span = attn_tp_size * cp_group.world_size
    group_ranks = [
        list(range(base + i, base + i + span, attn_tp_size))
        for base in range(0, torch.distributed.get_world_size(), span)
        for i in range(attn_tp_size)
    ]
    group = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        torch.distributed.get_backend(get_world_group().device_group),
        use_custom_allreduce=False,
        use_message_queue_broadcaster=False,
        group_name="hyv4_gate_cp",
        use_mscclpp_allreduce=False,
        use_torch_symm_mem_allreduce=False,
    )
    if group.ranks != cp_group.ranks:
        raise RuntimeError(
            f"HYV4 gate CP partition {group.ranks} does not match attention CP "
            f"partition {cp_group.ranks}"
        )
    return group


class GateCPShardRuntime:
    """Shared communicator, stream, events, and two full-weight buffers."""

    def __init__(self, local_weight: torch.Tensor):
        if local_weight.ndim != 2 or not local_weight.is_cuda:
            raise ValueError("HYV4 CP gate weights must be a CUDA 2-D tensor")
        self.group = _init_gate_cp_group()
        self.comm = self.group.pynccl_comm
        self.cp_size = self.group.world_size
        if self.cp_size <= 1:
            raise ValueError("HYV4 CP gate runtime requires CP size > 1")
        self.local_shape = tuple(local_weight.shape)
        self.dtype = local_weight.dtype
        self.device = local_weight.device
        full_shape = (local_weight.shape[0] * self.cp_size, local_weight.shape[1])
        self.stream = torch.cuda.Stream(device=self.device)
        self.buffers = [
            torch.empty(full_shape, dtype=local_weight.dtype, device=self.device)
            for _ in range(2)
        ]
        self.ready = [torch.cuda.Event() for _ in range(2)]
        self.free = [torch.cuda.Event() for _ in range(2)]
        current = torch.cuda.current_stream(device=self.device)
        for event in self.free:
            event.record(current)
        self.owner = [None, None]
        self.weight_mode: Optional[bool] = None
        self.prefetched = False

    def _all_gather(self, output: torch.Tensor, input_: torch.Tensor):
        input_.record_stream(self.stream)
        if self.comm is not None and getattr(self.comm, "available", False):
            with self.comm.change_state(enable=True):
                self.comm.all_gather(output, input_)
        else:
            self.group.all_gather_into_tensor(output, input_)

    def prefetch(self, linear_gate, slot: int):
        if self.owner[slot] is linear_gate:
            return
        if slot not in (0, 1):
            raise ValueError(f"invalid gate buffer slot {slot}")
        if not self.prefetched:
            # Activation mode drives this same communicator from the compute
            # stream. Without this edge the first side-stream gather of a
            # forward can start while earlier compute-stream gathers (the
            # previous forward or graph replay) are still in flight, and two
            # streams on one RCCL comm may interleave differently per rank.
            self.stream.wait_stream(torch.cuda.current_stream(device=self.device))
        self.stream.wait_event(self.free[slot])
        with torch.cuda.stream(self.stream):
            self._all_gather(self.buffers[slot], linear_gate.weight)
            self.ready[slot].record(self.stream)
            self.buffers[slot].record_stream(self.stream)
        self.owner[slot] = linear_gate
        self.prefetched = True

    def acquire(self, linear_gate, slot: int) -> torch.Tensor:
        self.prefetch(linear_gate, slot)
        compute = torch.cuda.current_stream(device=self.device)
        compute.wait_event(self.ready[slot])
        self.buffers[slot].record_stream(compute)
        return self.buffers[slot]

    def release(self, slot: int):
        self.free[slot].record(torch.cuda.current_stream(device=self.device))
        self.owner[slot] = None

    def gate_from_activation(self, local_weight: torch.Tensor, x: torch.Tensor):
        n_local = local_weight.shape[0]
        lead = x.shape[:-1]
        if x.numel() == 0:
            return x.new_empty((*lead, n_local * self.cp_size))
        local = F.linear(x, local_weight).reshape(-1, n_local)
        gathered = local.new_empty((self.cp_size, local.shape[0], n_local))
        self._all_gather(gathered, local)
        return gathered.permute(1, 0, 2).reshape(*lead, n_local * self.cp_size)


def get_gate_cp_shard_runtime(local_weight: torch.Tensor) -> GateCPShardRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = GateCPShardRuntime(local_weight)
        shard_mib = local_weight.numel() * local_weight.element_size() / 2**20
        logger.info(
            "Rank %d: linear_gate CP-sharded weight %s (%.1f MiB/layer, "
            "saves %.1f MiB/layer, CP=%d, two full buffers=%.1f MiB",
            torch.distributed.get_rank(),
            tuple(local_weight.shape),
            shard_mib,
            shard_mib * (_RUNTIME.cp_size - 1),
            _RUNTIME.cp_size,
            2 * shard_mib * _RUNTIME.cp_size,
        )
    elif (
        tuple(local_weight.shape) != _RUNTIME.local_shape
        or local_weight.dtype != _RUNTIME.dtype
    ):
        raise ValueError("all HYV4 CP gate layers must have identical local shape/dtype")
    return _RUNTIME


class GateCPShardChain:
    """Sharded linear_gate layers for one PP stage, in execution order."""

    def __init__(self, attns: List[torch.nn.Module]):
        if not attns:
            raise ValueError("gate CP chain requires at least one attention layer")
        self.runtime = attns[0]._gate_cp_runtime
        self.gates = [attn.linear_gate for attn in attns]
        self.in_features = self.gates[0].weight.shape[1]
        for idx, attn in enumerate(attns):
            if attn._gate_cp_runtime is not self.runtime:
                raise ValueError("all layers in a gate CP chain must share runtime")
            attn._gate_cp_chain = self
            attn._gate_cp_index = idx

    def begin(self, tokens_differ: bool, num_tokens: int):
        rt = self.runtime
        weight_mode = tokens_differ or num_tokens >= self.in_features
        if torch.cuda.is_current_stream_capturing():
            if tokens_differ:
                raise RuntimeError("HYV4 CP gate cannot capture CP-split batches")
            weight_mode = False
        if rt.prefetched:
            torch.cuda.current_stream(device=rt.device).wait_stream(rt.stream)
        rt.owner = [None, None]
        rt.weight_mode = weight_mode
        rt.prefetched = False
        if weight_mode:
            for idx in range(min(2, len(self.gates))):
                rt.prefetch(self.gates[idx], idx % 2)

    def end(self):
        rt = self.runtime
        if rt.prefetched:
            torch.cuda.current_stream(device=rt.device).wait_stream(rt.stream)
        rt.weight_mode = None
        rt.prefetched = False
        rt.owner = [None, None]

    def gate(self, idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        rt = self.runtime
        linear_gate = self.gates[idx]
        if rt.weight_mode is False:
            return rt.gate_from_activation(linear_gate.weight, hidden_states)
        slot = idx % 2
        weight = rt.acquire(linear_gate, slot)
        out = F.linear(hidden_states, weight)
        rt.release(slot)
        if idx + 2 < len(self.gates):
            rt.prefetch(self.gates[idx + 2], slot)
        return out


def build_gate_cp_chain(attns) -> Optional[GateCPShardChain]:
    attns = [a for a in attns if getattr(a, "gate_cp_shard_enabled", False)]
    return GateCPShardChain(attns) if attns else None
