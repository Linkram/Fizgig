"""Hardware-selected faster GEMMs for frozen BF16 weights on RDNA2.

Keep BF16 model/activation storage; FP32 GEMMs avoid the slow BF16 path measured
on gfx1030 with torch 2.12/ROCm 7.15. Disable autocast locally so it cannot undo
the conversion. No global dtype change or persistent FP32 weight cache.

The experimental scaled_fp16 alternative normalizes operands before conversion
and restores their range afterward. It was slower than FP32 in the block test
and is retained only for reproducible comparisons. Both paths change rounding.
"""
import functools
import os

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


@functools.lru_cache(maxsize=None)
def _is_rdna2(device_index):
    from fizgig.utils.gpu_backend import is_rocm
    if not is_rocm():
        return False
    props = torch.cuda.get_device_properties(device_index)
    return getattr(props, "gcnArchName", "").split(":")[0].startswith("gfx103")


def enabled(x):
    return (os.environ.get("FIZGIG_RDNA2_LINEAR", "auto") in ("auto", "1", "fp32", "scaled_fp16")
            and x.device.type == "cuda" and x.dtype == torch.bfloat16
            and _is_rdna2(x.device.index))


def stream_nf4_enabled(device):
    """Stream by default on RDNA2; retain an explicit loader override for debugging."""
    override = os.environ.get("FIZGIG_STREAM_NF4")
    if override is not None:
        return override == "1"
    device = torch.device(device)
    return device.type == "cuda" and _is_rdna2(device.index)


def matmul(a, b):
    if os.environ.get("FIZGIG_RDNA2_LINEAR", "0") == "scaled_fp16":
        return scaled_mm(a, b)
    with torch.autocast(device_type=a.device.type, enabled=False):
        return (a.float() @ b.float()).to(a.dtype)


def scaled_mm(a, b):
    """A[M,K] @ B[K,N], with row/column scales and FP32 rescaling.

    Each normalized operand is bounded by one. Limit K to 32768, leaving room
    below FP16's maximum even in the worst-case dot product. Accumulation uses
    the backend's normal FP16 GEMM; the output is restored to A's dtype.
    """
    if a.shape[-1] > 32768:
        return a @ b
    with torch.autocast(device_type=a.device.type, enabled=False):
        af, bf = a.float(), b.float()
        sa = af.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        sb = bf.abs().amax(dim=-2, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        ah = (af / sa).to(torch.float16)
        bh = (bf / sb).to(torch.float16)
        del af, bf
        return ((ah @ bh).float() * sa * sb).to(a.dtype)


class _FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(weight)
        ctx.shape = x.shape
        return matmul(x.reshape(-1, x.shape[-1]), weight.t()).reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        dx = matmul(grad.reshape(-1, grad.shape[-1]), weight)
        return dx.reshape(ctx.shape), None


def frozen_linear(x, weight, bias=None):
    if not enabled(x) or weight.requires_grad or weight.dtype != x.dtype:
        return F.linear(x, weight, bias)
    y = _FrozenLinear.apply(x, weight)
    return y if bias is None else y + bias
