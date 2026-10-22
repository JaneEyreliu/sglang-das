"""DeepGEMM HIPC W4A8 down-GEMM completion signals for HCU SBO."""

from importlib import import_module


def run_hcu_w4a8_hipc(
    a, b, out, masked_m, expected_m, *, overlap_args=None, meta_overlap_args=None
):
    gemm = import_module("deepgemm").m_grouped_w4a8_gemm_nt_masked_hipc
    enabled = overlap_args is not None
    if enabled:
        if meta_overlap_args is None:
            raise ValueError("W4A8 HIPC overlap requires shared DeepEP metadata")
        overlap_args.start_event.record()
    block_m, threshold = gemm(
        a,
        b,
        out,
        masked_m,
        expected_m,
        enable_overlap=enabled,
        signal=overlap_args.signal if enabled else None,
    )
    if enabled:
        if (
            block_m <= 0
            or (64 % block_m != 0 and block_m % 64 != 0)
            or threshold <= 0
        ):
            raise ValueError(f"Invalid W4A8 HIPC completion metadata: {block_m}, {threshold}")
        meta_overlap_args.update(block_m=block_m, threshold=threshold)
    return out
