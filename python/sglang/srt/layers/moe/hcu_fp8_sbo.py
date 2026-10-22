"""Pair32 FP8 packing and DeepEP completion-signal metadata on gfx938."""

from importlib import import_module

import torch


def get_hcu_fp8_pair32_packer(device):
    if device.type != "cuda":
        return None
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    if arch != "gfx938":
        return None
    return import_module("deepgemm.group_gemm_weight_pack").pack_fp8_weight_pair32


def run_hcu_fp8_pair32(
    a, b, out, masked_m, expected_m, *, overlap_args=None, meta_overlap_args=None
):
    gemm = import_module(
        "deepgemm.m_group_gemm_nt_masked"
    ).m_grouped_fp8_gemm_nt_masked_pair32
    enabled = overlap_args is not None
    if enabled:
        if meta_overlap_args is None:
            raise ValueError("FP8 pair32 overlap requires shared DeepEP metadata")
        overlap_args.start_event.record()
    # The native schema requires a Tensor even when signals are not read.
    signal = overlap_args.signal if enabled else masked_m
    block_m, threshold = gemm(
        a,
        b,
        out,
        masked_m,
        expected_m,
        enable_overlap=enabled,
        signal=signal,
    )
    if enabled:
        if block_m not in (64, 128) or threshold <= 0:
            raise ValueError(f"Invalid pair32 completion metadata: {block_m}, {threshold}")
        meta_overlap_args.update(block_m=block_m, threshold=threshold)
    return out
