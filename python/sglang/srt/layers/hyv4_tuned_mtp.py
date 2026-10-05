"""Opt-in inference-only MTP input projection tuning for gfx936/BF16.

HYV4_MTP_INPUT_TUNING=1 prepares the [6144,12288] weight before graph capture.
Pretransposed PyTorch GEMM is used except M=1, which uses the validated kernel
below to avoid the slow default NN solution. No extra persistent weight copy.
"""
import logging
import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mtp_single(X,WT,Y):
    n=tl.program_id(0)*128+tl.arange(0,128)
    m=tl.arange(0,16)
    k=tl.arange(0,128)
    acc=tl.full((16,128),0,tl.float32)
    for base in range(0,12288,128):
        kk=base+k
        a=tl.load(X+kk[None,:]+m[:,None]*12288,m[:,None]<1,0)
        b=tl.load(WT+kk[:,None]*6144+n[None,:])
        acc+=tl.dot(a,b)
    tl.store(Y+m[:,None]*6144+n[None,:],acc,m[:,None]<1)


@torch.no_grad()
def prepare_mtp_input(model, enabled=None):
    if enabled is None:enabled=os.environ.get('HYV4_MTP_INPUT_TUNING','0')=='1'
    if not enabled:return 0
    count=0
    for name,layer in model.named_modules():
        if not name.endswith('eh_proj'):continue
        w=getattr(layer,'weight',None)
        if (w is None or not w.is_cuda or w.dtype!=torch.bfloat16
                or tuple(w.shape)!=(6144,12288) or getattr(layer,'bias',None) is not None):continue
        if not getattr(torch.cuda.get_device_properties(w.device),'gcnArchName','').startswith('gfx936'):continue
        with torch.cuda.device(w.device):
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('Prepare MTP input before graph capture')
            if w.stride()!=(1,6144):w.data=w.detach().t().contiguous().t()
            layer._hyv4_mtp_input_tuned=True;count+=1
    logging.getLogger(__name__).info('HYV4 MTP input tuning: prepared=%d',count)
    return count


@torch.no_grad()
def try_tuned_mtp_input(layer,x):
    if not getattr(layer,'_hyv4_mtp_input_tuned',False):return None
    w=layer.weight
    if (tuple(w.shape)!=(6144,12288) or w.stride()!=(1,6144)
            or x.dtype!=torch.bfloat16 or w.dtype!=torch.bfloat16 or x.device!=w.device):return None
    if x.ndim==2 and x.shape==(1,12288) and x.is_contiguous():
        y=torch.empty((1,6144),device=x.device,dtype=x.dtype)
        _mtp_single[(48,)](x,w,y,num_warps=4,num_stages=2,matrix_instr_nonkdim=16)
        return y
    return F.linear(x,w)
