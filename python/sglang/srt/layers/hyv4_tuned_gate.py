"""Opt-in gfx936 HY4 gate tuning; prepare once before CUDA/HIP graph capture.

HYV4_GATE_TUNING=off (default), torch (pretranspose only), or triton.
Only the measured BF16 attention-TP2 shape [8192, 6144] is eligible.
No quantization. Logical parameter shape and Parameter identity are preserved.
"""
import logging
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gate_mm(X, WT, Y, M: tl.constexpr):
    n = tl.program_id(0) * 128 + tl.arange(0, 128)
    m = tl.arange(0, 16)
    k = tl.arange(0, 128)
    acc = tl.full((16, 128), 0, tl.float32)
    for base in range(0, 6144, 128):
        kk = base + k
        a = tl.load(X + m[:, None] * 6144 + kk[None, :], m[:, None] < M, 0)
        b = tl.load(WT + kk[:, None] * 8192 + n[None, :])
        acc += tl.dot(a, b)
    tl.store(Y + m[:, None] * 8192 + n[None, :], acc, m[:, None] < M)


@torch.no_grad()
def prepare_model_gates(model, mode=None):
    mode = os.environ.get('HYV4_GATE_TUNING', 'off') if mode is None else mode
    if mode not in ('off', 'torch', 'triton'):
        raise ValueError('HYV4_GATE_TUNING must be off, torch, or triton')
    if mode == 'off':
        return 0
    prepared = 0
    for name, layer in model.named_modules():
        if not name.endswith('self_attn.linear_gate'):
            continue
        w = getattr(layer, 'weight', None)
        method = getattr(layer, 'quant_method', None)
        if (w is None or not w.is_cuda or w.dtype != torch.bfloat16
                or tuple(w.shape) != (8192, 6144) or w.requires_grad
                or getattr(layer, 'bias', None) is not None
                or (method is not None and type(method).__name__ != 'UnquantizedLinearMethod')):
            continue
        if not getattr(torch.cuda.get_device_properties(w.device), 'gcnArchName', '').startswith('gfx936'):
            continue
        with torch.cuda.device(w.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Prepare tuned gates before graph capture')
            if w.stride() != (1, 8192):
                # Replaces storage; no second persistent copy of the weight is kept.
                w.data = w.detach().t().contiguous().t()
            layer._hyv4_gate_tuning_mode = mode
            prepared += 1
    logging.getLogger(__name__).info('HYV4 gate tuning: mode=%s prepared=%d', mode, prepared)
    return prepared


def try_tuned_gate(layer, x):
    mode = getattr(layer, '_hyv4_gate_tuning_mode', None)
    if mode is None:
        return None
    w = layer.weight
    # Guard layout in case a subsequent weight reload replaced the prepared storage.
    if (tuple(w.shape) != (8192, 6144) or w.dtype != torch.bfloat16
            or w.stride() != (1, 8192) or x.dtype != torch.bfloat16
            or x.device != w.device):
        return None
    # Other shapes, including prefill, retain a standard PyTorch linear path.
    if mode == 'triton' and x.ndim == 2 and x.is_contiguous() and x.shape[0] in (1, 4, 8):
        if x.shape[1] != 6144:
            return None
        out = torch.empty((x.shape[0], 8192), device=x.device, dtype=x.dtype)
        _gate_mm[(64,)](x, w, out, x.shape[0], num_warps=4, num_stages=2,
                       matrix_instr_nonkdim=16, waves_per_eu=0)
        return out
    return F.linear(x, w)
