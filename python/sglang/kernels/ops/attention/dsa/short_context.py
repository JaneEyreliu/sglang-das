"""Logical select-all TopK for a host-proven short-context DSA graph."""

import torch
import triton
import triton.language as tl


@triton.jit
def _logical_select_all_kernel(
    LENGTHS, OUTPUT, STRIDE: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    length = tl.load(LENGTHS + row * STRIDE)
    # Consume the current per-query causal length on every graph replay, not
    # the request's final length. The caller proves all lengths <= TOPK.
    tl.store(
        OUTPUT + row * TOPK + column,
        tl.where(column < length, column, -1),
        mask=column < TOPK,
    )


def logical_select_all(lengths: torch.Tensor, topk: int) -> torch.Tensor:
    """Return logical IDs [0, causal_length) and -1 padding, without logits.

    Contract: every input length is in [0, topk]. Ownership filtering and page
    lookup remain downstream, preserving IndexTopKShareState across layers.
    """
    if lengths.ndim != 1 or topk <= 0:
        raise ValueError("select-all requires one-dimensional lengths and topk > 0")
    result = torch.empty(
        (lengths.numel(), topk), dtype=torch.int32, device=lengths.device
    )
    if lengths.numel():
        _logical_select_all_kernel[(lengths.numel(),)](
            lengths, result, lengths.stride(0), topk, triton.next_power_of_2(topk)
        )
    return result
