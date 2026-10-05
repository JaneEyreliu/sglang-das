"""BF16 Hadamard-128 rotation with FP32 butterflies, for gfx936 inference."""
import torch
import triton
import triton.language as tl


@triton.jit
def _hadamard128(X,Y,ROWS:tl.constexpr,SCALE:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0)*B+tl.arange(0,B)
    c=tl.arange(0,128)
    v=tl.load(X+r[:,None]*128+c[None,:],r[:,None]<ROWS,0).to(tl.float32)
    for shift in tl.static_range(7):
        bit=1<<shift
        indices=tl.broadcast_to((c^bit)[None,:],(B,128))
        other=tl.gather(v,indices,axis=1)
        v=tl.where((c[None,:]&bit)==0,v+other,other-v)
    if SCALE != 1.0:
        # Match the existing BF16 GEMM rounding before its separate scaling step.
        v=v.to(tl.bfloat16).to(tl.float32)
    tl.store(Y+r[:,None]*128+c[None,:],v*SCALE,r[:,None]<ROWS)


def hadamard128(x,scale=1.0,block_rows=4):
    if x.dtype!=torch.bfloat16 or x.shape[-1]!=128 or not x.is_cuda:
        raise ValueError('hadamard128 requires GPU BF16 with final dimension 128')
    if block_rows not in (1,2,4,8,16):
        raise ValueError('block_rows must be 1,2,4,8,16')
    xc=x.contiguous();out=torch.empty_like(xc);rows=x.numel()//128
    if rows:
        _hadamard128[(triton.cdiv(rows,block_rows),)](xc,out,rows,scale,block_rows,num_warps=4)
    return out
