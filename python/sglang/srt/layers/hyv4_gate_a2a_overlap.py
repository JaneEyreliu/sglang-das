"""Run HYV4 attention gate beside DCP2/4 A2A, with an explicit fork and join."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable, Optional

import torch


class _HipGraphForkBarrier:
    """Separate an accumulated predecessor from the two fork branches.

    The deployed HCU runtime serializes sibling accumulated nodes when their
    predecessor is also accumulated. A plain empty predecessor avoids that
    replay behavior while retaining the original two-queue layout and all
    ready/done ordering. It adds no kernel or tensor access.
    """

    def __init__(self):
        import ctypes as C

        self.C = C
        self.lib = C.CDLL("libamdhip64.so")
        ptr = C.c_void_p
        self.info = self.lib.hipStreamGetCaptureInfo_v2
        self.info.argtypes = (
            ptr,
            C.POINTER(C.c_int),
            C.POINTER(C.c_ulonglong),
            C.POINTER(ptr),
            C.POINTER(C.POINTER(ptr)),
            C.POINTER(C.c_size_t),
        )
        self.info.restype = C.c_int
        self.add = self.lib.hipGraphAddEmptyNode
        self.add.argtypes = (C.POINTER(ptr), ptr, C.POINTER(ptr), C.c_size_t)
        self.add.restype = C.c_int
        self.update = self.lib.hipStreamUpdateCaptureDependencies
        self.update.argtypes = (ptr, C.POINTER(ptr), C.c_size_t, C.c_uint)
        self.update.restype = C.c_int

    def add_barrier(self, stream):
        C = self.C
        status, capture_id = C.c_int(), C.c_ulonglong()
        graph, deps, count = C.c_void_p(), C.POINTER(C.c_void_p)(), C.c_size_t()
        rc = self.info(
            C.c_void_p(stream.cuda_stream), C.byref(status), C.byref(capture_id),
            C.byref(graph), C.byref(deps), C.byref(count),
        )
        if rc:
            raise RuntimeError(f"HYV4 gate fork capture-info failed: HIP {rc}")
        if status.value not in (0, 1):
            raise RuntimeError(f"HYV4 gate fork capture is invalid: status {status.value}")
        if status.value == 0 or not count.value:
            return
        node = C.c_void_p()
        rc = self.add(C.byref(node), graph, deps, count)
        if rc:
            raise RuntimeError(f"HYV4 gate fork empty-node failed: HIP {rc}")
        deps = (C.c_void_p * 1)(node.value)
        # Replace the frontier with a node that depends on the entire old
        # frontier. Both real branches now depend on this plain empty node.
        # hipStreamSetCaptureDependencies = 1; no old dependency is dropped.
        rc = self.update(
            C.c_void_p(stream.cuda_stream), deps, C.c_size_t(1), C.c_uint(1)
        )
        if rc:
            raise RuntimeError(f"HYV4 gate fork frontier update failed: HIP {rc}")


_HIP_GRAPH_API = None


def _get_hip_graph_api():
    global _HIP_GRAPH_API
    if _HIP_GRAPH_API is None:
        _HIP_GRAPH_API = _HipGraphForkBarrier()
    return _HIP_GRAPH_API


class GateA2AState:
    """Per-layer events on the model's dedicated, shared gate stream.

    Created at model initialization, never inside forward or graph capture.
    Layer calls are sequential; tensors and results remain per invocation.
    """

    def __init__(self, stream: torch.cuda.Stream):
        self.stream = stream
        self._graph_api = (
            _get_hip_graph_api()
            if (
                getattr(getattr(torch, "version", None), "hip", None)
                and os.getenv("HIP_GRAPH_ACCUMULATE_DISPATCH", "1") != "0"
            )
            else None
        )
        self.ready = torch.cuda.Event(enable_timing=False)
        self.done = torch.cuda.Event(enable_timing=False)

    def prepare_fork(self, caller):
        if self._graph_api is not None and torch.cuda.is_current_stream_capturing():
            self._graph_api.add_barrier(caller)


class DeferredHyv4Gate:
    """One forward's gate, resolved serially or beside the original collective."""

    def __init__(
        self,
        hidden_states: torch.Tensor,
        project: Callable[[torch.Tensor], torch.Tensor],
        state: GateA2AState,
    ):
        self.hidden_states = hidden_states
        self.project = project
        self.dtype = hidden_states.dtype
        self.state = state
        self.result: Optional[torch.Tensor] = None

    def resolve(self) -> torch.Tensor:
        if self.result is None:
            # Also handles the non-raw-LSE fallback on the caller's stream.
            self.result = self.project(self.hidden_states)
        return self.result

    @contextmanager
    def overlap(self):
        """Fork before pack, yield to pack/A2A, and join before consumption.

        The gate is submitted BEFORE calling the collective, so even a
        host-blocking transport cannot defer its submission. The side stream
        waits only for the pre-pack caller frontier, never for A2A completion.
        Capture records this DAG; replay need not preserve stream identities.
        """
        if self.result is not None:
            yield self.result
            return

        state = self.state
        caller = torch.cuda.current_stream(self.hidden_states.device)
        state.prepare_fork(caller)
        # Snapshot the caller BEFORE the collective is submitted.
        state.ready.record(caller)
        # Protect both directions of cross-stream tensor use, including eager
        # execution and errors. No copies or extra arithmetic are introduced.
        self.hidden_states.record_stream(state.stream)
        done_recorded = False
        try:
            with torch.cuda.stream(state.stream):
                state.stream.wait_event(state.ready)
                try:
                    self.result = self.project(self.hidden_states)
                finally:
                    state.done.record(state.stream)
                    done_recorded = True
            self.result.record_stream(caller)
            yield self.result
        finally:
            # Also joins a partially submitted gate if projection or A2A raises.
            # This is a stream dependency, not a host/device synchronize().
            if done_recorded:
                caller.wait_event(state.done)
