from itertools import accumulate
from typing import List, Optional

import torch
import triton
import triton.language as tl


def transform_index_page_table_prefill(**kwargs):
    return transform_index_page_table_prefill_fast(**kwargs)


def transform_index_page_table_decode(**kwargs):
    return transform_index_page_table_decode_fast(**kwargs)


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
def transform_index_page_table_prefill_kernel(
    page_table_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    cu_seqlens_q_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    page_table_stride_0: tl.constexpr,
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
) -> torch.Tensor:
    """
    Transform the page table according to topk indices for sparse topk attention.
    Args:
        page_table: [qo_len, max_seqlen_k], the original page table
        topk_indices: [qo_len, topk], the topk indices for each query position
        out_valid_chunks: Optional [qo_len, 1] bool/uint8/int32 output. The same
            launch writes whether each row contains a nonnegative final index.
    Returns:
        transformed_page_table: [qo_len, topk], the transformed page table
        For out-of-bound indices in topk_indices, this should be filled with -1.
    """
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    assert topk_indices.shape[1] == 2048
    qo_len = topk_indices.shape[0]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    if out_valid_chunks is not None:
        _validate_out_valid_chunks(out_valid_chunks, qo_len, 1, topk_indices.device)
    if qo_len == 0:
        return result
    # Launch triton kernel
    grid = (qo_len,)
    transform_index_page_table_decode_kernel[grid](
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
) -> torch.Tensor:
    """Transform TopK and optionally report validity for each 256-index tile.

    ``out_valid_chunks`` has shape [output rows, 8], with bool/uint8/int32 dtype.
    All result and validity rows, including padding, are written in the same
    launch. A row is locally valid iff any of its eight flags is nonzero. The
    flags describe this transform's output, before any later KV remapping.
    """
    assert page_size == 1
    assert topk_indices.shape[1] == 2048
    real_num_tokens = sum(extend_lens_cpu)
    result = _allocate_prefill_result(
        topk_indices,
        real_num_tokens,
        output_num_tokens,
        fill_padding=out_valid_chunks is None,
    )
    block_topk = 256
    num_chunks = triton.cdiv(topk_indices.shape[1], block_topk)
    if out_valid_chunks is not None:
        _validate_out_valid_chunks(
            out_valid_chunks, result.shape[0], num_chunks, topk_indices.device
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


def transform_index_page_table_decode_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
    dcp_size: int = 1,
    dcp_rank: int = 0,
) -> torch.Tensor:
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert result.shape == topk_indices.shape
    torch.gather(
        page_table.to(result.dtype),
        dim=1,
        index=topk_indices.clamp(min=0),
        out=result,
    )
    result[topk_indices < 0] = -1
    if dcp_size > 1:
        owned_mask = (result >= 0) & (result % dcp_size == dcp_rank)
        result.copy_(torch.where(owned_mask, result // dcp_size, -1))
    return result


def transform_index_page_table_prefill_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
    output_num_tokens: Optional[int] = None,
    page_table_is_expanded: bool = False,
    dcp_size: int = 1,
    dcp_rank: int = 0,
) -> torch.Tensor:
    assert page_size == 1
    real_num_tokens = sum(extend_lens_cpu)
    result = _allocate_prefill_result(topk_indices, real_num_tokens, output_num_tokens)

    if page_table_is_expanded:
        if real_num_tokens > 0:
            transform_index_page_table_decode_ref(
                page_table[:real_num_tokens],
                topk_indices[:real_num_tokens],
                result=result[:real_num_tokens],
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
            )
        return result

    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        transform_index_page_table_decode_ref(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
        )
        offset += l
    return result


if __name__ == "__main__":
    bs, topk, max_seqlen = 10, 2048, 3000
    page_table = torch.randint(0, 100, (bs, max_seqlen), device="cuda")
    topk_indices = torch.full((bs, topk), -1, device="cuda")
    topk_indices[:, :1600] = torch.arange(1600).unsqueeze(0).repeat(bs, 1)
    ref_result = transform_index_page_table_decode_ref(page_table, topk_indices)
    result = transform_index_page_table_decode_fast(page_table, topk_indices)
    assert torch.all(result == ref_result)
    print("Passed")
