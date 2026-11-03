from itertools import accumulate
from typing import List, Optional

import torch
import triton
import triton.language as tl


def transform_index_page_table_prefill(**kwargs):
    return transform_index_page_table_prefill_fast(**kwargs)


def transform_index_page_table_decode(**kwargs):
    return transform_index_page_table_decode_fast(**kwargs)


@triton.jit
def transform_index_page_table_dcp_kernel(
    page_table_ptr,
    result_ptr,
    counts_ptr,
    row_stride: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_TOPK)
    slots = tl.load(
        page_table_ptr + row * row_stride + offsets, mask=offsets < TOPK, other=-1
    )
    owned = (slots >= 0) & (slots % DCP_SIZE == DCP_RANK)
    packed_offsets = tl.cumsum(owned.to(tl.int32), axis=0) - 1
    count = tl.sum(owned.to(tl.int32), axis=0)
    tl.store(result_ptr + row * TOPK + packed_offsets, slots // DCP_SIZE, mask=owned)
    tl.store(result_ptr + row * TOPK + count + offsets, -1, mask=offsets < TOPK - count)
    tl.store(counts_ptr + row, count)


def transform_index_page_table_dcp(
    page_table: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows, topk = page_table.shape
    result = torch.empty_like(page_table)
    counts = torch.empty(num_rows, dtype=torch.int32, device=page_table.device)
    if num_rows == 0 or topk == 0:
        return result, counts.zero_()
    assert page_table.stride(1) == 1
    transform_index_page_table_dcp_kernel[(num_rows,)](
        page_table,
        result,
        counts,
        page_table.stride(0),
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        TOPK=topk,
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=4,
    )
    return result, counts


@triton.jit
def prepare_trtllm_nope_sparse_metadata_kernel(
    page_table_ptr: torch.Tensor,
    topk_lens_ptr: torch.Tensor,
    row_stride: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_TOPK)
    valid = offsets < TOPK
    indices = tl.load(
        page_table_ptr + row * row_stride + offsets,
        mask=valid,
        other=-1,
    )
    topk_len = tl.sum((valid & (indices >= 0)).to(tl.int32), axis=0)

    # TRTLLM-GEN's native H512 dynamic sparse kernel produces NaNs for an
    # empty row. CUDA-graph padding rows are never consumed, so point them at
    # a valid dummy token and run a one-element attention instead.
    is_empty = topk_len == 0
    tl.store(page_table_ptr + row * row_stride, 0, mask=is_empty)
    tl.store(topk_lens_ptr + row, tl.maximum(topk_len, 1))


def prepare_trtllm_nope_sparse_metadata(
    page_table: torch.Tensor,
) -> torch.Tensor:
    """Build per-query active top-k lengths for native H512 TRTLLM-GEN MLA.

    ``page_table`` must contain packed valid token locations followed by ``-1``
    padding. The tensor is only modified for fully empty CUDA-graph padding
    rows, whose first entry is replaced with the valid dummy token location 0.
    """
    assert page_table.ndim == 2
    assert page_table.dtype == torch.int32
    assert page_table.is_contiguous()
    num_rows, topk = page_table.shape
    topk_lens = torch.empty(num_rows, dtype=torch.int32, device=page_table.device)
    block_topk = triton.next_power_of_2(topk)
    prepare_trtllm_nope_sparse_metadata_kernel[(num_rows,)](
        page_table,
        topk_lens,
        page_table.stride(0),
        TOPK=topk,
        BLOCK_TOPK=block_topk,
        num_warps=8,
    )
    return topk_lens


def _allocate_prefill_result(
    topk_indices: torch.Tensor,
    real_num_tokens: int,
    output_num_tokens: Optional[int],
    fill_padding: bool = True,
) -> torch.Tensor:
    topk_num_tokens = topk_indices.shape[0]
    if output_num_tokens is None:
        output_num_tokens = topk_num_tokens

    assert real_num_tokens <= topk_num_tokens, (
        f"sum(extend_lens_cpu) ({real_num_tokens}) exceeds "
        f"topk_indices rows ({topk_num_tokens})"
    )
    assert topk_num_tokens <= output_num_tokens, (
        f"topk_indices rows ({topk_num_tokens}) exceeds "
        f"output_num_tokens ({output_num_tokens})"
    )

    result = torch.empty(
        (output_num_tokens, topk_indices.shape[1]),
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if fill_padding and real_num_tokens < output_num_tokens:
        result[real_num_tokens:].fill_(-1)
    return result


def _validate_out_valid_chunks(
    out_valid_chunks: torch.Tensor,
    rows: int,
    chunks: int,
    device: torch.device,
) -> None:
    assert out_valid_chunks.shape == (rows, chunks)
    assert out_valid_chunks.device == device
    assert out_valid_chunks.dtype in (torch.bool, torch.uint8, torch.int32)


def _validate_out_topk_lengths(
    out_topk_lengths: torch.Tensor,
    rows: int,
    device: torch.device,
) -> None:
    assert out_topk_lengths.shape == (rows,)
    assert out_topk_lengths.device == device
    assert out_topk_lengths.dtype == torch.int32
    assert out_topk_lengths.is_contiguous()


@triton.jit
def transform_index_page_table_compact_kernel(
    page_table_ptr,
    topk_indices_ptr,
    cu_seqlens_q_ptr,
    result_ptr,
    out_topk_lengths_ptr,
    out_valid_chunks_ptr,
    page_table_stride_0: tl.constexpr,
    page_table_stride_1: tl.constexpr,
    topk_indices_stride_0: tl.constexpr,
    topk_indices_stride_1: tl.constexpr,
    result_stride_0: tl.constexpr,
    result_stride_1: tl.constexpr,
    valid_stride_0: tl.constexpr,
    valid_stride_1: tl.constexpr,
    PAGE_TABLE_COLS: tl.constexpr,
    TOPK_NUM_ROWS: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr,
    PAGE_TABLE_IS_EXPANDED: tl.constexpr,
    HAS_VALID_CHUNKS: tl.constexpr,
    VALID_CHUNKS: tl.constexpr,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
):
    # A single CTA owns every slot and the count of this output row. Prefix
    # scatter and tail fill are disjoint, preserving order and duplicates.
    row = tl.program_id(0)
    offsets = tl.arange(0, 2048)
    row_is_valid = row < TOPK_NUM_ROWS
    page_table_row = row
    if HAS_CU_SEQLENS:
        live_rows = tl.load(cu_seqlens_q_ptr + NUM_REQUESTS)
        row_is_valid = row_is_valid & (row < live_rows)
        if not PAGE_TABLE_IS_EXPANDED:
            # Find the request for this query from the live GPU boundaries.
            # Repeated boundaries represent zero-query requests and must be
            # skipped. This also works when boundaries change on graph replay.
            left = 0
            right = NUM_REQUESTS
            while left < right:
                middle = (left + right) // 2
                query_end = tl.load(cu_seqlens_q_ptr + middle + 1)
                move_right = row >= query_end
                left = tl.where(move_right, middle + 1, left)
                right = tl.where(move_right, right, middle)
            page_table_row = left

    logical_indices = tl.load(
        topk_indices_ptr
        + row * topk_indices_stride_0
        + offsets * topk_indices_stride_1,
        mask=row_is_valid,
        other=-1,
    )
    valid = row_is_valid & (logical_indices >= 0) & (logical_indices < PAGE_TABLE_COLS)
    physical_indices = tl.load(
        page_table_ptr
        + page_table_row * page_table_stride_0
        + logical_indices * page_table_stride_1,
        mask=valid,
        other=-1,
    )
    valid = valid & (physical_indices >= 0)
    if dcp_size > 1:
        valid = valid & (physical_indices % dcp_size == dcp_rank)
        physical_indices = physical_indices // dcp_size
    valid_int = valid.to(tl.int32)
    destination = tl.cumsum(valid_int, 0) - 1
    valid_count = tl.sum(valid_int, 0)
    tl.store(
        result_ptr + row * result_stride_0 + destination * result_stride_1,
        physical_indices,
        mask=valid,
    )
    tl.store(
        result_ptr + row * result_stride_0 + offsets * result_stride_1,
        -1,
        mask=offsets >= valid_count,
    )
    tl.store(out_topk_lengths_ptr + row, valid_count)
    if HAS_VALID_CHUNKS:
        chunks = tl.arange(0, VALID_CHUNKS)
        tl.store(
            out_valid_chunks_ptr
            + row * valid_stride_0
            + chunks * valid_stride_1,
            valid_count > chunks * (2048 // VALID_CHUNKS),
        )


def _transform_index_page_table_compact(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: torch.Tensor,
    out_topk_lengths: torch.Tensor,
    out_valid_chunks: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    page_table_is_expanded: bool,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    _validate_out_topk_lengths(out_topk_lengths, result.shape[0], topk_indices.device)
    assert result.shape == (out_topk_lengths.shape[0], 2048)
    assert result.dtype == torch.int32 and result.device == topk_indices.device
    assert dcp_size > 0 and 0 <= dcp_rank < dcp_size
    if result.shape[0] == 0:
        return result
    transform_index_page_table_compact_kernel[(result.shape[0],)](
        page_table,
        topk_indices,
        cu_seqlens_q,
        result,
        out_topk_lengths,
        out_valid_chunks,
        page_table.stride(0),
        page_table.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(1),
        result.stride(0),
        result.stride(1),
        out_valid_chunks.stride(0) if out_valid_chunks is not None else 0,
        out_valid_chunks.stride(1) if out_valid_chunks is not None else 0,
        PAGE_TABLE_COLS=page_table.shape[1],
        TOPK_NUM_ROWS=topk_indices.shape[0],
        NUM_REQUESTS=cu_seqlens_q.shape[0] - 1 if cu_seqlens_q is not None else 0,
        HAS_CU_SEQLENS=cu_seqlens_q is not None,
        PAGE_TABLE_IS_EXPANDED=page_table_is_expanded,
        HAS_VALID_CHUNKS=out_valid_chunks is not None,
        VALID_CHUNKS=out_valid_chunks.shape[1] if out_valid_chunks is not None else 1,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        num_warps=4,
    )
    return result


@triton.jit
def transform_index_page_table_decode_kernel(
    page_table_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    page_size: tl.constexpr,
    page_table_row_stride: tl.constexpr,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
    out_valid_chunks_ptr=None,
    valid_stride_0: tl.constexpr = 0,
    HAS_VALID_CHUNKS: tl.constexpr = False,
):
    TOPK: tl.constexpr = 2048
    req_id = tl.program_id(0)
    page_table_ptr = page_table_ptr + req_id * page_table_row_stride
    topk_indices_ptr = topk_indices_ptr + req_id * TOPK
    result_ptr = result_ptr + req_id * TOPK

    offset = tl.arange(0, TOPK)  # topk should be 2048
    loaded_topk_indices = tl.load(topk_indices_ptr + offset)
    mask = loaded_topk_indices >= 0
    loaded_kv_indices = tl.load(page_table_ptr + loaded_topk_indices, mask=mask)
    if dcp_size > 1:
        mask = mask & (loaded_kv_indices % dcp_size == dcp_rank)
        loaded_kv_indices = loaded_kv_indices // dcp_size
    tl.store(result_ptr + offset, loaded_kv_indices, mask=mask)
    tl.store(result_ptr + offset, -1, mask=~mask)
    if HAS_VALID_CHUNKS:
        has_valid = tl.sum((mask & (loaded_kv_indices >= 0)).to(tl.int32), 0) > 0
        tl.store(out_valid_chunks_ptr + req_id * valid_stride_0, has_valid)


@triton.jit
def transform_index_page_table_decode_tiled_kernel(
    page_table_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    page_table_row_stride: tl.constexpr,
    topk_indices_stride_0: tl.constexpr,
    topk_indices_stride_1: tl.constexpr,
    result_stride_0: tl.constexpr,
    result_stride_1: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    """Width-generic form of the kernel above.

    The 2048 variant folds the row stride into a compile-time TOPK and covers a
    whole row with one unmasked `tl.arange`, which needs TOPK to be a power of
    two. k-pool widths are not: `index_topk + index_kpool - 1` is 2051 for
    GLM-5.3-Flash. Tile the row instead and carry the strides explicitly.
    """
    req_id = tl.program_id(0)
    topk_offsets = tl.program_id(1) * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    in_row = topk_offsets < TOPK

    loaded_topk_indices = tl.load(
        topk_indices_ptr
        + req_id * topk_indices_stride_0
        + topk_offsets * topk_indices_stride_1,
        mask=in_row,
        other=-1,
    )
    selected = in_row & (loaded_topk_indices >= 0)
    loaded_kv_indices = tl.load(
        page_table_ptr + req_id * page_table_row_stride + loaded_topk_indices,
        mask=selected,
        other=-1,
    )
    tl.store(
        result_ptr + req_id * result_stride_0 + topk_offsets * result_stride_1,
        loaded_kv_indices,
        mask=in_row,
    )


# Expanded EAGLE page tables are contiguous, so their row stride changes with
# the exact context length. Treating it as constexpr creates one cubin per
# observed length and grows the loaded-module set in long-lived processes.
@triton.jit(do_not_specialize=["page_table_stride_0"])
def transform_index_page_table_prefill_kernel(
    page_table_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    cu_seqlens_q_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    page_table_stride_0,
    page_table_stride_1: tl.constexpr,
    topk_indices_stride_0: tl.constexpr,
    topk_indices_stride_1: tl.constexpr,
    result_stride_0: tl.constexpr,
    result_stride_1: tl.constexpr,
    PAGE_TABLE_IS_EXPANDED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
    out_valid_chunks_ptr=None,
    valid_stride_0: tl.constexpr = 0,
    valid_stride_1: tl.constexpr = 0,
    NUM_REQUESTS: tl.constexpr = 0,
    OUTPUT_NUM_TOKENS: tl.constexpr = 0,
    HAS_VALID_CHUNKS: tl.constexpr = False,
):
    request_id = tl.program_id(0)
    query_offsets = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    topk_offsets = tl.program_id(2) * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)

    if HAS_VALID_CHUNKS:
        if request_id == NUM_REQUESTS:
            # This extra request program owns the padding rows. Read the live
            # endpoint from the graph's metadata and cover the tail in this
            # launch, without a separate fill or a racing atomic reduction.
            padding_start = tl.load(cu_seqlens_q_ptr + NUM_REQUESTS)
            padding_start += tl.program_id(1) * BLOCK_Q
            while padding_start < OUTPUT_NUM_TOKENS:
                token_indices = padding_start + tl.arange(0, BLOCK_Q)
                row_mask = token_indices < OUTPUT_NUM_TOKENS
                tl.store(
                    result_ptr
                    + token_indices[:, None] * result_stride_0
                    + topk_offsets[None, :] * result_stride_1,
                    -1,
                    mask=row_mask[:, None] & (topk_offsets[None, :] < TOPK),
                )
                tl.store(
                    out_valid_chunks_ptr
                    + token_indices * valid_stride_0
                    + tl.program_id(2) * valid_stride_1,
                    0,
                    mask=row_mask,
                )
                padding_start += tl.num_programs(1) * BLOCK_Q
            return

    query_start = tl.load(cu_seqlens_q_ptr + request_id)
    query_end = tl.load(cu_seqlens_q_ptr + request_id + 1)
    # Grid axis 1 spans the batch-max extend len; fully-masked blocks store nothing.
    if query_start + tl.program_id(1) * BLOCK_Q >= query_end:
        return
    token_indices = query_start + query_offsets
    mask = (token_indices[:, None] < query_end) & (topk_offsets[None, :] < TOPK)

    loaded_topk_indices = tl.load(
        topk_indices_ptr
        + token_indices[:, None] * topk_indices_stride_0
        + topk_offsets[None, :] * topk_indices_stride_1,
        mask=mask,
        other=-1,
    )
    valid_topk_mask = mask & (loaded_topk_indices >= 0)

    if PAGE_TABLE_IS_EXPANDED:
        page_table_rows = token_indices
    else:
        page_table_rows = token_indices * 0 + request_id
    loaded_kv_indices = tl.load(
        page_table_ptr
        + page_table_rows[:, None] * page_table_stride_0
        + loaded_topk_indices * page_table_stride_1,
        mask=valid_topk_mask,
        other=-1,
    )
    if dcp_size > 1:
        owned_mask = valid_topk_mask & (loaded_kv_indices % dcp_size == dcp_rank)
        loaded_kv_indices = tl.where(owned_mask, loaded_kv_indices // dcp_size, -1)
    tl.store(
        result_ptr
        + token_indices[:, None] * result_stride_0
        + topk_offsets[None, :] * result_stride_1,
        loaded_kv_indices,
        mask=mask,
    )
    if HAS_VALID_CHUNKS:
        has_valid = tl.sum((mask & (loaded_kv_indices >= 0)).to(tl.int32), 1) > 0
        tl.store(
            out_valid_chunks_ptr
            + token_indices * valid_stride_0
            + tl.program_id(2) * valid_stride_1,
            has_valid,
            mask=token_indices < query_end,
        )


def transform_index_page_table_decode_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    out_valid_chunks: Optional[torch.Tensor] = None,
    out_topk_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Transform the page table according to topk indices for sparse topk attention.
    Args:
        page_table: [qo_len, max_seqlen_k], the original page table
        topk_indices: [qo_len, topk], the topk indices for each query position
        out_valid_chunks: Optional [qo_len, 1] bool/uint8/int32 output. The same
            launch writes whether each row contains a nonnegative final index.
        out_topk_lengths: Optional int32 [qo_len] output. When supplied, the
            same launch stably compacts locally owned indices to a prefix,
            fills the tail with -1, and writes its actual length.
    Returns:
        transformed_page_table: [qo_len, topk], the transformed page table
        For out-of-bound indices in topk_indices, this should be filled with -1.
    """
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    qo_len = topk_indices.shape[0]
    topk = topk_indices.shape[1]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    if out_valid_chunks is not None:
        _validate_out_valid_chunks(out_valid_chunks, qo_len, 1, topk_indices.device)
    if out_topk_lengths is not None:
        return _transform_index_page_table_compact(
            page_table,
            topk_indices,
            result,
            out_topk_lengths,
            out_valid_chunks,
            None,
            True,
            dcp_size,
            dcp_rank,
        )
    if qo_len == 0:
        return result
    if topk == 2048:
        # Keep the single-program path for the unpooled width, which covers a
        # whole row per program with no masking.
        transform_index_page_table_decode_kernel[(qo_len,)](
            page_table,
            topk_indices,
            result,
            page_size,
            page_table_row_stride=page_table.stride(0),
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            out_valid_chunks_ptr=out_valid_chunks,
            valid_stride_0=(
                out_valid_chunks.stride(0) if out_valid_chunks is not None else 0
            ),
            HAS_VALID_CHUNKS=out_valid_chunks is not None,
        )
        return result

    # The tiled k-pool path has no DCP owner filter or row-validity output.
    assert dcp_size == 1 and out_valid_chunks is None, (
        "DCP/valid-chunk decode transforms require topk == 2048"
    )
    block_topk = 256
    transform_index_page_table_decode_tiled_kernel[
        (qo_len, triton.cdiv(topk, block_topk))
    ](
        page_table,
        topk_indices,
        result,
        page_table.stride(0),
        topk_indices.stride(0),
        topk_indices.stride(1),
        result.stride(0),
        result.stride(1),
        TOPK=topk,
        BLOCK_TOPK=block_topk,
        num_warps=4,
    )
    return result


def transform_index_page_table_prefill_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
    output_num_tokens: Optional[int] = None,
    page_table_is_expanded: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    out_valid_chunks: Optional[torch.Tensor] = None,
    out_topk_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Transform TopK and optionally report validity for each 256-index tile.

    ``out_valid_chunks`` has shape [output rows, 8], with bool/uint8/int32 dtype.
    All result and validity rows, including padding, are written in the same
    launch. A row is locally valid iff any of its eight flags is nonzero. The
    flags describe this transform's output, before any later KV remapping.
    Providing int32 ``out_topk_lengths[output rows]`` also stably compacts each
    row and reports its actual local length; padding rows have length zero.
    """
    assert page_size == 1
    assert topk_indices.ndim == 2
    assert topk_indices.shape[1] > 0
    real_num_tokens = sum(extend_lens_cpu)
    result = _allocate_prefill_result(
        topk_indices,
        real_num_tokens,
        output_num_tokens,
        fill_padding=out_valid_chunks is None and out_topk_lengths is None,
    )
    block_topk = 256
    num_chunks = triton.cdiv(topk_indices.shape[1], block_topk)
    if out_valid_chunks is not None:
        _validate_out_valid_chunks(
            out_valid_chunks, result.shape[0], num_chunks, topk_indices.device
        )
    if out_topk_lengths is not None:
        if cu_seqlens_q is None:
            cu_seqlens_q = torch.tensor(
                [0, *accumulate(extend_lens_cpu)],
                dtype=torch.int32,
                device=topk_indices.device,
            )
        return _transform_index_page_table_compact(
            page_table,
            topk_indices,
            result,
            out_topk_lengths,
            out_valid_chunks,
            cu_seqlens_q,
            page_table_is_expanded,
            dcp_size,
            dcp_rank,
        )
    if result.shape[0] == 0 or (real_num_tokens == 0 and out_valid_chunks is None):
        return result

    max_extend_len = max(extend_lens_cpu, default=0)
    block_q = 1 if max_extend_len == 1 else 2 if max_extend_len == 2 else 4
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.tensor(
            [0, *accumulate(extend_lens_cpu)],
            dtype=torch.int32,
            device=topk_indices.device,
        )
    num_requests = cu_seqlens_q.shape[0] - 1
    grid = (
        num_requests + int(out_valid_chunks is not None),
        max(1, triton.cdiv(max_extend_len, block_q)),
        num_chunks,
    )
    transform_index_page_table_prefill_kernel[grid](
        page_table,
        topk_indices,
        cu_seqlens_q,
        result,
        page_table.stride(0),
        page_table.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(1),
        result.stride(0),
        result.stride(1),
        PAGE_TABLE_IS_EXPANDED=page_table_is_expanded,
        TOPK=topk_indices.shape[1],
        BLOCK_Q=block_q,
        BLOCK_TOPK=block_topk,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        out_valid_chunks_ptr=out_valid_chunks,
        valid_stride_0=(
            out_valid_chunks.stride(0) if out_valid_chunks is not None else 0
        ),
        valid_stride_1=(
            out_valid_chunks.stride(1) if out_valid_chunks is not None else 0
        ),
        NUM_REQUESTS=num_requests,
        OUTPUT_NUM_TOKENS=result.shape[0],
        HAS_VALID_CHUNKS=out_valid_chunks is not None,
        num_warps=4,
    )
    return result
