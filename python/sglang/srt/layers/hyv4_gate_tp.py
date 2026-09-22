"""Node-local input-sharded HYV4 gate projection for DP attention.

The batch layout and A2A workspaces are shared by all layers of one forward.
Only activations move: requests, KV cache and attention heads stay DP-local.
Eager uses torch collectives; graph capture uses the subgroup PyNCCL transport.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class GateTPGraphPlan:
    batch_size: int
    can_run: bool
    dsa_variant: Optional[str]

    @classmethod
    def synchronize(
        cls,
        local_batch_size: int,
        can_run: bool,
        dsa_variant: Optional[str],
        group: Any,
    ) -> "GateTPGraphPlan":
        """Coordinate BEFORE replay, including ranks falling back to eager.

        Gate A2A requires equal graph geometry within each node. Use the TP/DP
        world's maximum so EP peers also agree on graph vs eager. A single
        sparse DSA request selects the sparse graph everywhere; otherwise
        peers could replay graphs from different collective captures.
        """
        variant_code = {None: -1, "dense": 0, "sparse": 1}[dsa_variant]
        local = torch.tensor(
            [local_batch_size, int(can_run), variant_code], dtype=torch.int64
        )
        gathered = [torch.empty_like(local) for _ in range(group.world_size)]
        dist.all_gather(gathered, local, group=group.cpu_group)
        rows = torch.stack(gathered).tolist()
        variants = [row[2] for row in rows]
        if min(variants) == -1 and max(variants) != -1:
            raise RuntimeError("HYV4 gate TP peers disagree on DSA graph variants")
        return cls(
            batch_size=max(row[0] for row in rows),
            can_run=all(row[1] for row in rows),
            dsa_variant={-1: None, 0: "dense", 1: "sparse"}[max(variants)],
        )


@dataclass
class GateTPBatch:
    group: Any
    local_tokens: int
    padded_tokens: int
    rank: int
    world_size: int
    capture: bool = False
    _send: Optional[torch.Tensor] = None
    _recv: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls, local_tokens: int, group: Any, *, capture: bool = False
    ) -> "GateTPBatch":
        if capture:
            # All peers capture the same positive bucket. Replay selects a
            # common bucket outside the graph; no CPU collective belongs here.
            if local_tokens <= 0:
                raise ValueError("HYV4 gate graph capture requires a positive bucket")
            padded_tokens = local_tokens
        else:
            # Count actual tensor rows, including scheduler padding, once per
            # eager forward. Empty ranks must still compute for active peers.
            count = torch.tensor([local_tokens], dtype=torch.int64)
            counts = [torch.empty_like(count) for _ in range(group.world_size)]
            dist.all_gather(counts, count, group=group.cpu_group)
            padded_tokens = max(int(item.item()) for item in counts)
        return cls(
            group=group,
            local_tokens=local_tokens,
            padded_tokens=padded_tokens,
            capture=capture,
            rank=group.rank_in_group,
            world_size=group.world_size,
        )

    def project(self, hidden_states: torch.Tensor, linear: Any) -> torch.Tensor:
        tokens, hidden_size = hidden_states.shape
        if tokens != self.local_tokens or hidden_size % self.world_size:
            raise ValueError("HYV4 gate TP input does not match its batch layout")
        if (
            linear.tp_size != self.world_size
            or linear.tp_rank != self.rank
            or linear.reduce_results
        ):
            raise ValueError("HYV4 gate TP requires an unreduced row-parallel linear")
        if self.padded_tokens == 0:
            # Every member observed the same all-empty batch and skips together.
            return hidden_states.new_empty((0, linear.output_size))

        shard_width = hidden_size // self.world_size
        shape = (self.world_size, self.padded_tokens, shard_width)
        if (
            self._send is None
            or self._send.shape != shape
            or self._send.dtype != hidden_states.dtype
            or self._send.device != hidden_states.device
        ):
            self._send = hidden_states.new_empty(shape)
            self._recv = hidden_states.new_empty(shape)

        # send[dst, token, feature] -> recv[src, token, feature].
        # A rank with no local tokens sends zeros but still computes for peers.
        self._send.zero_()
        self._send[:, :tokens].copy_(
            hidden_states.reshape(tokens, self.world_size, shard_width).permute(1, 0, 2)
        )
        if self.capture:
            comm = self.group.pynccl_comm
            if comm is None or not comm.available:
                raise RuntimeError("HYV4 gate graph requires a PyNCCL communicator")
            # PyNCCL's equal-split send/recv implementation takes flat buffers.
            # Kernel autotune can run under model_capture_mode before the
            # runner enters graph_capture(), so enable the transport explicitly.
            with comm.change_state(enable=True):
                comm.all_to_all_single(self._recv.view(-1), self._send.view(-1))
        else:
            dist.all_to_all_single(
                self._recv, self._send, group=self.group.device_group
            )
        partial, _ = linear(
            self._recv.reshape(self.world_size * self.padded_tokens, shard_width),
            skip_all_reduce=True,
        )
        # Sum in FP32 for BF16/FP16 GEMM outputs, before the nonlinear sigmoid.
        summed = (
            partial.float()
            if partial.dtype in (torch.bfloat16, torch.float16)
            else partial
        )
        if self.capture:
            with comm.change_state(enable=True):
                comm.all_reduce(summed)
        else:
            dist.all_reduce(summed, op=dist.ReduceOp.SUM, group=self.group.device_group)
        start = self.rank * self.padded_tokens
        # Own just the local output; do not retain the full node-wide buffer.
        return summed[start : start + tokens].to(hidden_states.dtype).clone()
