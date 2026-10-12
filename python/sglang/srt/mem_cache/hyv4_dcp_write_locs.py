"""Forward-scoped reuse of HYV4's HCU DCP KV write locations.

The cache is deliberately not attached to ForwardBatch or the KV pool: both
objects survive multiple speculative steps and graph captures. Each model
forward starts a new scope, so its first layer always records a fresh producer
in the captured graph. Replay executes that producer with the live input.
The original, widened out_cache_loc remains available to the replicated index-K
cache and is never modified by this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


_write_loc_cache: ContextVar[dict | None] = ContextVar(
    "hyv4_dcp_forward_write_locs", default=None
)


@contextmanager
def hyv4_dcp_write_loc_scope():
    """Isolate target, draft steps, nested calls, and failed forwards."""
    token = _write_loc_cache.set({})
    try:
        yield
    finally:
        _write_loc_cache.reset(token)


def scoped_hyv4_dcp_write_locs(forward):
    """Create the optional scope around exactly one model forward."""

    @wraps(forward)
    def wrapped(*args, **kwargs):
        from sglang.srt.environ import envs

        if not envs.SGLANG_HY4_DCP_FUSED_KV_WRITE_LOCS.get():
            return forward(*args, **kwargs)
        with hyv4_dcp_write_loc_scope():
            return forward(*args, **kwargs)

    return wrapped


def _tensor_version(loc):
    try:
        return loc._version
    except RuntimeError:
        # Inference tensors have no version counter. Like out_cache_loc in the
        # existing per-layer path, their values must stay constant within one
        # model forward; a new draft step always opens a separate scope.
        return None


def _build_remapped_locs(loc, **kwargs):
    from sglang.kernels.ops.kvcache.hyv4_dcp_write_locs import (
        fused_hyv4_dcp_write_locs,
    )

    return fused_hyv4_dcp_write_locs(loc, **kwargs)


def remap_hyv4_dcp_write_locs(
    loc: torch.Tensor,
    *,
    dcp_size: int,
    dcp_rank: int,
    pool_size: int,
    padding_capacity: int,
) -> torch.Tensor:
    """Fuse once per forward and reuse only the same unchanged input tensor.

    Outside a scope this still executes the fused producer on every call;
    there is never an implicit persistent cache. Holding the input in the cache
    entry prevents object-id reuse while the scope is alive. Pool geometry and
    owner rank are part of the key, including for multiple pools in a forward.
    """
    kwargs = dict(
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        pool_size=pool_size,
        padding_capacity=padding_capacity,
    )
    cache = _write_loc_cache.get()
    if cache is None:
        return _build_remapped_locs(loc, **kwargs)
    key = (
        id(loc),
        _tensor_version(loc),
        tuple(loc.shape),
        tuple(loc.stride()),
        dcp_size,
        dcp_rank,
        pool_size,
        padding_capacity,
    )
    entry = cache.get(key)
    if entry is None:
        result = _build_remapped_locs(loc, **kwargs)
        cache[key] = (loc, result)
        return result
    return entry[1]
