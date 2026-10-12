"""Static-shaped HCU DCP KV write-location remapping in one launch."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _hyv4_dcp_write_locs_kernel(
    loc_ptr,
    out_ptr,
    loc_stride: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(loc_ptr + rows * loc_stride, rows < NUM_ROWS, other=0).to(tl.int64)
    # Match torch.remainder / floor division even for negative test inputs.
    # Production cache locations are nonnegative virtual addresses.
    remainder = values % DCP_SIZE
    owner = tl.where(remainder < 0, remainder + DCP_SIZE, remainder)
    # Triton integer division/remainder truncate toward zero. Correct the
    # quotient separately, avoiding values - owner overflow at INT64_MIN.
    local = values // DCP_SIZE - (remainder < 0).to(tl.int64)
    result = tl.where(owner == DCP_RANK, local, POOL_SIZE + rows)
    tl.store(out_ptr + rows, result, rows < NUM_ROWS)


def fused_hyv4_dcp_write_locs(
    loc: torch.Tensor,
    *,
    dcp_size: int,
    dcp_rank: int,
    pool_size: int,
    padding_capacity: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return local owned slots or a distinct reserved slot per non-owner row.

    This implements the existing torch remainder/divide/arange/add/where
    expression exactly. It does not compact rows, change the input, or infer
    whether an owned location is a real token or an allocator dummy location.
    """
    if loc.device.type != "cuda" or loc.ndim != 1:
        raise ValueError("HCU DCP write locations must be a one-dimensional GPU tensor")
    if loc.dtype not in (torch.int32, torch.int64):
        raise ValueError("HCU DCP write locations must use int32 or int64")
    if dcp_size < 1 or not 0 <= dcp_rank < dcp_size:
        raise ValueError("Invalid DCP size or rank")
    if pool_size < 0 or padding_capacity < loc.numel():
        raise ValueError("KV pool padding capacity must cover all captured rows")
    if out is None:
        out = torch.empty(loc.shape, dtype=loc.dtype, device=loc.device)
    if (
        out.shape != loc.shape
        or out.dtype != loc.dtype
        or out.device != loc.device
        or not out.is_contiguous()
    ):
        raise ValueError("Output must be contiguous with the input shape, dtype and device")
    if (
        loc.numel() != 0
        and out.untyped_storage().data_ptr() == loc.untyped_storage().data_ptr()
    ):
        raise ValueError("Remapping must not overwrite the global input locations")
    if loc.numel():
        block = 256
        _hyv4_dcp_write_locs_kernel[(triton.cdiv(loc.numel(), block),)](
            loc,
            out,
            loc.stride(0),
            NUM_ROWS=loc.numel(),
            DCP_SIZE=dcp_size,
            DCP_RANK=dcp_rank,
            POOL_SIZE=pool_size,
            BLOCK=block,
            num_warps=4,
        )
    return out
