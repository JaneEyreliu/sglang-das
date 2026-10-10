"""HYV4 DCP2 post-attention kernels.

The transport is BF16/FP16 ``[2, B, H_local, D + 2]``. The final two
16-bit elements hold the *bits* of one natural-log FP32 LSE, not a converted
16-bit LSE. Empty local partitions are represented by zero output and -inf
LSE. The learnable zero-valued sink is added once, on the receiving rank.

All entry points accept preallocated destinations for graph replay. They do
not synchronize, inspect device tensor values, or communicate between ranks.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


def _transport_shape(buffer: torch.Tensor):
    if buffer.ndim != 4 or buffer.shape[0] != 2:
        raise ValueError("DCP2 transport must have shape [2, B, H_local, D + 2]")
    if buffer.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("DCP2 transport requires BF16 or FP16 output")
    if not buffer.is_contiguous():
        raise ValueError("DCP2 transport must be contiguous")
    _, batch, heads, packed_dim = buffer.shape
    dim = packed_dim - 2
    if dim <= 0 or dim % 2 or heads <= 0:
        raise ValueError("DCP2 transport needs a positive even D and positive heads")
    return batch, heads, dim


def _same_device(reference: torch.Tensor, *tensors: torch.Tensor):
    if reference.device.type != "cuda":
        raise ValueError("HYV4 DCP2 fused kernels require a CUDA/HIP device")
    if any(t.device != reference.device for t in tensors):
        raise ValueError("HYV4 DCP2 tensors must reside on the same device")


@triton.jit
def _sanitize_pack_kernel(
    output_ptr,
    lse_ptr,
    valid_ptr,
    send_ptr,
    send_lse_ptr,
    output_stride_b,
    output_stride_h,
    output_stride_d,
    lse_stride_b,
    lse_stride_h,
    valid_stride_b,
    valid_stride_c,
    B: tl.constexpr,
    H_LOCAL: tl.constexpr,
    D: tl.constexpr,
    VALID_ROWS: tl.constexpr,
    VALID_BUFFER_ROWS: tl.constexpr,
    VALID_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    chunks = tl.arange(0, BLOCK_C)
    chunk_valid = tl.load(
        valid_ptr + b * valid_stride_b + chunks * valid_stride_c,
        mask=(b < VALID_ROWS) & (b < VALID_BUFFER_ROWS) & (chunks < VALID_CHUNKS),
        other=0,
    )
    valid = (b < VALID_ROWS) & (tl.sum((chunk_valid != 0).to(tl.int32), 0) > 0)
    ds = tl.arange(0, BLOCK_D)
    values = tl.load(
        output_ptr + b * output_stride_b + h * output_stride_h + ds * output_stride_d,
        mask=valid & (ds < D),
        other=0.0,
    )
    lse = tl.load(
        lse_ptr + b * lse_stride_b + h * lse_stride_h,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    peer = h // H_LOCAL
    h_local = h % H_LOCAL
    row = (peer * B + b) * H_LOCAL + h_local
    tl.store(send_ptr + row * (D + 2) + ds, values, mask=ds < D)
    tl.store(send_lse_ptr + row * ((D + 2) // 2) + D // 2, lse)


def fused_sanitize_pack(
    raw_output: torch.Tensor,
    raw_lse: torch.Tensor,
    valid_chunks: torch.Tensor,
    send_combined: torch.Tensor,
    *,
    num_valid_rows: Optional[int] = None,
) -> None:
    """Sanitize empty/padded rows and pack FlashMLA partials in one launch.

    ``raw_output`` is [B_real, H_pad, D] or [B_real, 1, H_pad, D], with
    arbitrary strides. ``raw_lse`` is FP32 [B_real, H_pad] or
    [B_real, H_pad, 1] (also [B_real, 1, H_pad]). Only the first 2*H_local
    heads are transmitted. LSE must be natural-log and exclude the sink.

    ``valid_chunks`` is [B_valid, C] or [B_valid], containing nonzero flags
    for any locally owned, nonnegative KV address in that chunk. C is normally
    1 (decode) or 8 (tiled verify). The producer must overwrite all real rows.
    ``num_valid_rows`` is the static real prefix length, defaulting to B_real.
    All remaining rows of ``send_combined`` are explicitly overwritten.
    """
    batch, heads, dim = _transport_shape(send_combined)
    _same_device(send_combined, raw_output, raw_lse, valid_chunks)
    if raw_output.dtype != send_combined.dtype:
        raise ValueError("Raw output and transport dtypes must match")
    if raw_output.ndim == 4 and raw_output.shape[1] == 1:
        real_rows, _, padded_heads, raw_dim = raw_output.shape
        output_strides = (
            raw_output.stride(0),
            raw_output.stride(2),
            raw_output.stride(3),
        )
    elif raw_output.ndim == 3:
        real_rows, padded_heads, raw_dim = raw_output.shape
        output_strides = raw_output.stride()
    else:
        raise ValueError("Raw output must be [B, H, D] or [B, 1, H, D]")
    if raw_dim != dim or padded_heads < 2 * heads:
        raise ValueError(
            "Raw output head count or latent dimension does not match transport"
        )
    if raw_lse.dtype != torch.float32:
        raise ValueError("Raw LSE must be FP32 natural-log values")
    if raw_lse.ndim == 2:
        lse_rows, lse_heads = raw_lse.shape
        lse_strides = raw_lse.stride()
    elif raw_lse.ndim == 3 and raw_lse.shape[-1] == 1:
        lse_rows, lse_heads, _ = raw_lse.shape
        lse_strides = raw_lse.stride(0), raw_lse.stride(1)
    elif raw_lse.ndim == 3 and raw_lse.shape[1] == 1:
        lse_rows, _, lse_heads = raw_lse.shape
        lse_strides = raw_lse.stride(0), raw_lse.stride(2)
    else:
        raise ValueError("Raw LSE must be [B, H], [B, H, 1], or [B, 1, H]")
    if lse_heads < 2 * heads:
        raise ValueError("Raw LSE does not contain all DCP heads")
    if valid_chunks.ndim == 1:
        valid_rows, chunks = valid_chunks.shape[0], 1
        valid_strides = valid_chunks.stride(0), 0
    elif valid_chunks.ndim == 2:
        valid_rows, chunks = valid_chunks.shape
        valid_strides = valid_chunks.stride()
    else:
        raise ValueError("Validity flags must be [B] or [B, C]")
    if chunks <= 0:
        raise ValueError("Validity flags must contain at least one chunk")
    if num_valid_rows is None:
        num_valid_rows = real_rows
    if not isinstance(num_valid_rows, int) or not 0 <= num_valid_rows <= min(
        batch, real_rows, lse_rows, valid_rows
    ):
        raise ValueError(
            "num_valid_rows must fit the raw tensors, validity flags, and bucket"
        )
    if batch == 0:
        return
    _sanitize_pack_kernel[(batch, 2 * heads)](
        raw_output,
        raw_lse,
        valid_chunks,
        send_combined,
        send_combined.view(torch.float32),
        *output_strides,
        *lse_strides,
        *valid_strides,
        B=batch,
        H_LOCAL=heads,
        D=dim,
        VALID_ROWS=num_valid_rows,
        VALID_BUFFER_ROWS=valid_rows,
        VALID_CHUNKS=chunks,
        BLOCK_C=triton.next_power_of_2(chunks),
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=4,
    )


@triton.jit
def _sink_weights(lse0, lse1, sink):
    maximum = tl.maximum(tl.maximum(lse0, lse1), sink)
    # A disabled (-inf) sink and two empty ranks have a zero denominator.
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    w0 = tl.exp(lse0 - maximum)
    w1 = tl.exp(lse1 - maximum)
    ws = tl.exp(sink - maximum)
    denominator = w0 + w1 + ws
    safe_denominator = tl.where(denominator > 0, denominator, 1.0)
    return w0 / safe_denominator, w1 / safe_denominator


@triton.jit
def _sink_combine_kernel(
    recv_ptr,
    lse_ptr,
    sink_ptr,
    output_ptr,
    sink_stride,
    output_stride_b,
    output_stride_h,
    output_stride_d,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    row = b * H + h
    rank_rows = B * H
    lse0 = tl.load(lse_ptr + row * ((D + 2) // 2) + D // 2)
    lse1 = tl.load(lse_ptr + (rank_rows + row) * ((D + 2) // 2) + D // 2)
    sink = tl.load(sink_ptr + h * sink_stride).to(tl.float32)
    a0, a1 = _sink_weights(lse0, lse1, sink)
    ds = tl.arange(0, BLOCK_D)
    # Masked loads also prevent NaN * 0 if an empty input was not sanitized.
    o0 = tl.load(
        recv_ptr + row * (D + 2) + ds,
        mask=(ds < D) & (a0 != 0),
        other=0,
    ).to(tl.float32)
    o1 = tl.load(
        recv_ptr + (rank_rows + row) * (D + 2) + ds,
        mask=(ds < D) & (a1 != 0),
        other=0,
    ).to(tl.float32)
    combined = o0 * a0 + o1 * a1
    tl.store(
        output_ptr + b * output_stride_b + h * output_stride_h + ds * output_stride_d,
        combined,
        mask=ds < D,
    )


def _check_sink(buffer: torch.Tensor, sink: torch.Tensor, heads: int):
    _same_device(buffer, sink)
    if sink.ndim != 1 or sink.shape[0] != heads or not sink.is_floating_point():
        raise ValueError(
            "Sink must contain one floating-point logit per receiving head"
        )


def dcp2_sink_combine(
    recv_combined: torch.Tensor,
    local_sink: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Merge two local softmax states, adding one global zero-valued sink.

    LSE and finite sink logits use natural logs; -inf sink disables the sink.
    Empty states have -inf LSE. A fully empty row without a sink returns zero.
    Reduction is FP32, with one final cast to the transport/output dtype.
    """
    batch, heads, dim = _transport_shape(recv_combined)
    _check_sink(recv_combined, local_sink, heads)
    if out is None:
        out = torch.empty(
            (batch, heads, dim),
            device=recv_combined.device,
            dtype=recv_combined.dtype,
        )
    _same_device(recv_combined, out)
    if out.shape != (batch, heads, dim) or out.dtype != recv_combined.dtype:
        raise ValueError(
            "Merge output must be [B, H_local, D] with the transport dtype"
        )
    if batch:
        _sink_combine_kernel[(batch, heads)](
            recv_combined,
            recv_combined.view(torch.float32),
            local_sink,
            out,
            local_sink.stride(0),
            *out.stride(),
            B=batch,
            H=heads,
            D=dim,
            BLOCK_D=triton.next_power_of_2(dim),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _sink_project_gate_kernel(
    recv_ptr,
    lse_ptr,
    sink_ptr,
    weight_ptr,
    gate_ptr,
    output_ptr,
    sink_stride,
    weight_stride_h,
    weight_stride_d,
    weight_stride_v,
    gate_stride_b,
    gate_stride_h,
    gate_stride_v,
    output_stride_b,
    output_stride_v,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    PRESERVE_ROUNDING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    h = tl.program_id(0)
    bs = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    vs = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    rows = bs * H + h
    rank_rows = B * H
    lse0 = tl.load(
        lse_ptr + rows * ((D + 2) // 2) + D // 2,
        mask=bs < B,
        other=-float("inf"),
    )
    lse1 = tl.load(
        lse_ptr + (rank_rows + rows) * ((D + 2) // 2) + D // 2,
        mask=bs < B,
        other=-float("inf"),
    )
    sink = tl.load(sink_ptr + h * sink_stride).to(tl.float32)
    a0, a1 = _sink_weights(lse0, lse1, sink)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(tl.cdiv(D, BLOCK_K)):
        ds = k_start * BLOCK_K + ks
        o0 = tl.load(
            recv_ptr + rows[:, None] * (D + 2) + ds[None, :],
            mask=(bs[:, None] < B) & (ds[None, :] < D) & (a0[:, None] != 0),
            other=0,
        ).to(tl.float32)
        o1 = tl.load(
            recv_ptr + (rank_rows + rows[:, None]) * (D + 2) + ds[None, :],
            mask=(bs[:, None] < B) & (ds[None, :] < D) & (a1[:, None] != 0),
            other=0,
        ).to(tl.float32)
        merged = (o0 * a0[:, None] + o1 * a1[:, None]).to(recv_ptr.dtype.element_ty)
        weight = tl.load(
            weight_ptr
            + h * weight_stride_h
            + ds[:, None] * weight_stride_d
            + vs[None, :] * weight_stride_v,
            mask=(ds[:, None] < D) & (vs[None, :] < V),
            other=0,
        )
        acc = tl.dot(merged, weight, acc)
    gate = tl.load(
        gate_ptr
        + bs[:, None] * gate_stride_b
        + h * gate_stride_h
        + vs[None, :] * gate_stride_v,
        mask=(bs[:, None] < B) & (vs[None, :] < V),
        other=0,
    ).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    if PRESERVE_ROUNDING:
        # Match the observable BF16/FP16 stores of bmm and torch.sigmoid.
        acc = acc.to(output_ptr.dtype.element_ty).to(tl.float32)
        sigmoid = sigmoid.to(gate_ptr.dtype.element_ty).to(tl.float32)
    result = acc * sigmoid
    tl.store(
        output_ptr
        + bs[:, None] * output_stride_b
        + (h * V + vs[None, :]) * output_stride_v,
        result,
        mask=(bs[:, None] < B) & (vs[None, :] < V),
    )


def dcp2_sink_project_gate(
    recv_combined: torch.Tensor,
    local_sink: torch.Tensor,
    w_vc: torch.Tensor,
    gate: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    preserve_intermediate_rounding: bool = True,
) -> torch.Tensor:
    """Fuse DCP2 merge/sink, latent-to-value projection, sigmoid, and gate.

    ``w_vc`` is already-scaled [H_local, D, V], with arbitrary strides and
    the transport dtype. Quantized weights/scales and LoRA are not supported.
    ``gate`` is [B, H_local * V] or [B, H_local, V]. The result is flattened
    [B, H_local * V], ready for o_proj. The merged latent is rounded to the
    transport dtype before dot, matching standalone merge. By default the
    projection and sigmoid also preserve their original dtype roundings.
    """
    batch, heads, dim = _transport_shape(recv_combined)
    _check_sink(recv_combined, local_sink, heads)
    _same_device(recv_combined, w_vc, gate)
    if w_vc.ndim != 3 or w_vc.shape[:2] != (heads, dim) or w_vc.shape[2] <= 0:
        raise ValueError("V projection weights must be [H_local, D, V]")
    if w_vc.dtype != recv_combined.dtype or gate.dtype != recv_combined.dtype:
        raise ValueError(
            "Projection weights, gate, and transport must share a BF16/FP16 dtype"
        )
    value_dim = w_vc.shape[2]
    if gate.shape == (batch, heads * value_dim):
        gate_strides = gate.stride(0), value_dim * gate.stride(1), gate.stride(1)
    elif gate.shape == (batch, heads, value_dim):
        gate_strides = gate.stride()
    else:
        raise ValueError("Gate must be [B, H_local * V] or [B, H_local, V]")
    if out is None:
        out = torch.empty(
            (batch, heads * value_dim),
            device=recv_combined.device,
            dtype=recv_combined.dtype,
        )
    _same_device(recv_combined, out)
    if out.shape != (batch, heads * value_dim) or out.dtype != recv_combined.dtype:
        raise ValueError(
            "Projection output must be [B, H_local * V] with the transport dtype"
        )
    if batch:
        grid = (heads, triton.cdiv(batch, 16), triton.cdiv(value_dim, 32))
        _sink_project_gate_kernel[grid](
            recv_combined,
            recv_combined.view(torch.float32),
            local_sink,
            w_vc,
            gate,
            out,
            local_sink.stride(0),
            *w_vc.stride(),
            *gate_strides,
            *out.stride(),
            B=batch,
            H=heads,
            D=dim,
            V=value_dim,
            PRESERVE_ROUNDING=preserve_intermediate_rounding,
            BLOCK_M=16,
            BLOCK_N=32,
            BLOCK_K=64,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
