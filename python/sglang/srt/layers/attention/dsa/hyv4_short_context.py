"""Host-only policy for opt-in HYV4 HCU select-all indexer graphs.

No device tensor is reduced or copied here. Missing/incomplete host metadata
always selects the full indexer. Allocation bounds are snapshots belonging to
one ForwardBatch, never cached by request/tensor identity across forwards.
"""

from numbers import Integral
from typing import Optional


def hyv4_short_context_enabled(config, *, is_hcu: bool) -> bool:
    from sglang.srt.configs.model_config import is_hy_v4
    from sglang.srt.environ import envs

    return (
        is_hcu
        and config is not None
        and is_hy_v4(config)
        and envs.SGLANG_HY4_HCU_DSA_SHORT_CONTEXT_K_ONLY.get()
    )


def request_kv_upper_bound(reqs, extra_tokens: int) -> Optional[int]:
    """Bound every query in a forward using already allocated request slots.

    Spec-v2 allocates against CPU committed lengths with an overlap reserve
    before constructing ForwardBatch. Add another reserve for the verify /
    draft-extend length adjustments after construction. Over-allocation only
    makes us leave the short graph earlier; it cannot enable a wrong graph.
    """
    if not reqs or not isinstance(extra_tokens, Integral) or extra_tokens < 0:
        return None
    bound = 0
    for req in reqs:
        allocated = getattr(getattr(req, "kv", None), "kv_allocated_len", None)
        committed = getattr(req, "kv_committed_len", None)
        if (
            not isinstance(allocated, Integral)
            or not isinstance(committed, Integral)
            or allocated <= 0
            or committed < 0
            or committed > allocated
        ):
            return None
        bound = max(bound, int(allocated))
    return bound + int(extra_tokens)


def select_hyv4_dsa_variant(
    forward_batch, index_topk: int, *, extra_tokens: int = 0
) -> str:
    """Choose a graph before replay; padding rows never participate.

    The allocation snapshot already includes speculative headroom. A real CPU
    seq_lens mirror, when available, additionally checks the actual mode's
    prospective query width. Never trust the capture runner's padded/stale
    mirror: callers pass the original runtime ForwardBatch.
    """
    bound = getattr(forward_batch, "hyv4_dsa_kv_len_upper_bound", None)
    if not isinstance(bound, Integral) or bound < 0:
        bound = None
    seq_lens = getattr(forward_batch, "seq_lens_cpu", None)
    if seq_lens is not None and getattr(seq_lens.device, "type", None) == "cpu":
        real_bs = getattr(forward_batch, "batch_size", 0)
        if real_bs > 0 and seq_lens.numel() >= real_bs:
            real_lens = seq_lens[:real_bs]
            if int(real_lens.min().item()) < 0:
                return "sparse"
            mirror_bound = int(real_lens.max().item()) + max(int(extra_tokens), 0)
            bound = mirror_bound if bound is None else max(bound, mirror_bound)
    return "dense" if bound is not None and bound <= index_topk else "sparse"
