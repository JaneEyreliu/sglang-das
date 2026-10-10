"""HYV4-only launch checks and learnable-sink correction for FlashMLA."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Hyv4DcpRawLSE:
    """Per-call contract for HYV4's deferred DCP sink correction.

    Unlike a normal DSA LSE tensor, ``lse`` is natural-log and has no sink.
    The explicit carrier prevents it entering a base-2 or already-corrected
    merge. It owns the validity tensor for this layer/step (never global state).
    Output/LSE may retain FlashMLA head padding and contain only the real rows.
    """

    lse: torch.Tensor
    valid_chunks: torch.Tensor
    num_valid_rows: int
    num_total_rows: int
    num_heads: int


def validate_hyv4_launch(args, parallel):
    """Keep unsupported paths from silently skipping HYV4's gate or sink."""
    if not parallel.dcp_enabled:
        return
    if args.pp_size != 1:
        raise ValueError("HYV4 currently requires PP1")
    if args.speculative_algorithm is not None:
        if args.speculative_algorithm != "EAGLE":
            raise ValueError("HYV4 MTP currently requires EAGLE (NEXTN alias)")
        if getattr(args, "speculative_eagle_topk", None) != 1:
            raise ValueError("HYV4 MTP currently requires speculative_eagle_topk=1")
        for name in ("speculative_num_steps", "speculative_num_draft_tokens"):
            if (getattr(args, name, None) or 0) <= 0:
                raise ValueError(f"HYV4 MTP requires positive {name}")
    if args.enable_two_batch_overlap or args.enable_single_batch_overlap:
        raise ValueError("HYV4 iHC does not support SBO/TBO")
    graph = args.cuda_graph_config
    if graph is not None and (
        graph.prefill.backend != "disabled"
        or graph.decode.backend not in ("disabled", "full")
    ):
        raise ValueError(
            "HYV4 requires disabled prefill graphs and full/disabled decode graphs"
        )
    if args.attention_backend != "dsa" or any(
        impl != "flashmla_kv"
        for impl in (args.dsa_prefill_backend, args.dsa_decode_backend)
    ):
        raise ValueError("HYV4 requires DSA flashmla_kv for both prefill and decode")
    if parallel.attn_cp_size != 1:
        raise ValueError("HYV4 DCP adaptation currently requires attention CP1")
    if parallel.dcp_enabled:
        if parallel.attn_tp_size != parallel.attn_dcp_size:
            raise ValueError("HYV4 DCP requires attention TP size == DCP size")
        if tuple(parallel.dcp_group.ranks) != tuple(parallel.attn_tp_group.ranks):
            raise ValueError(
                "HYV4 DCP and attention TP groups must have identical ranks"
            )
        if parallel.dcp_comm_backend not in ("ag_rs", "a2a"):
            raise ValueError("HYV4 DCP requires ag_rs or a2a communication")
        if not args.enable_dp_attention or args.moe_dense_tp_size != 1:
            raise ValueError("HYV4 DCP requires DP attention and dense MLP TP1")


def apply_hyv4_sink(
    output: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    local_kv_counts: torch.Tensor,
    *,
    dcp_size: int = 1,
    lse_base2: bool = False,
):
    """Add one virtual zero-valued KV entry across all DCP partitions.

    FlashMLA returns an output normalized by the real keys and an LSE without
    a sink. Give each DCP rank exp(sink) / D of the virtual key's denominator,
    and update *both* output and LSE before the existing DCP combine. Thus the
    merge sums exactly one sink, including when a rank owns no selected KV.
    Inputs are [tokens, heads, value_dim], [tokens, heads], and [heads].
    """
    if dcp_size < 1 or sink.numel() != output.shape[1]:
        raise ValueError("HYV4 sink head count / DCP size does not match attention")
    valid = local_kv_counts[:, None] > 0
    local_lse = lse.float() * (math.log(2.0) if lse_base2 else 1.0)
    # Some kernels use +inf/NaN for empty rows. They contribute zero real KV.
    local_lse = torch.where(valid, local_lse, -torch.inf)
    sink_share = sink.float() - math.log(dcp_size)
    merged_lse = torch.logaddexp(local_lse, sink_share)
    scale = torch.exp(local_lse - merged_lse)
    output = torch.where(valid[..., None], output, 0.0)
    # where() above owns this tensor. TensorIterator computes BF16 * FP32 in
    # FP32 and casts on store, avoiding a full FP32 output temporary/copy.
    output.mul_(scale[..., None])
    if lse_base2:
        merged_lse = merged_lse * math.log2(math.e)
    return output.contiguous(), merged_lse.contiguous()
