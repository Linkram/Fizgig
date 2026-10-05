"""Opt-in RDNA2 attention: analytical backward without a checkpointed SDPA replay.

No dropout, causal masking or fused-kernel claims. The Krea 2 torch path supplies
its validity mask and already expands GQA. Float32 math preserves BF16 storage.
"""
import math
import torch


def _probabilities(q, k, mask):
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k = q.to(dtype).contiguous(), k.to(dtype).contiguous()
    scores = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(q.shape[-1]))
    if mask is not None:
        if mask.dtype == torch.bool:
            scores.masked_fill_(~mask, float('-inf'))
        else:
            scores = scores + mask
    probabilities = torch.softmax(scores, dim=-1)
    # SDPA defines fully masked rows to produce zero rather than NaN.
    return q, k, torch.nan_to_num(probabilities, nan=0.0)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask):
        ctx.save_for_backward(q, k, v, mask)
        with torch.autocast(device_type=q.device.type, enabled=False):
            _, _, probabilities = _probabilities(q, k, mask)
            return (probabilities @ v.to(probabilities.dtype).contiguous()).to(q.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        q, k, v, mask = ctx.saved_tensors
        with torch.autocast(device_type=q.device.type, enabled=False):
            qf, kf, probabilities = _probabilities(q, k, mask)
            vf, gf = v.to(probabilities.dtype).contiguous(), grad.to(probabilities.dtype).contiguous()
            dv = probabilities.transpose(-2, -1) @ gf
            dp = gf @ vf.transpose(-2, -1)
            ds = probabilities * (dp - (dp * probabilities).sum(dim=-1, keepdim=True))
            ds *= 1 / math.sqrt(q.shape[-1])
            dq = ds @ kf
            dk = ds.transpose(-2, -1) @ qf
            return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None


def analytical_attention(q, k, v, attn_mask=None, heads_per_chunk=2):
    if attn_mask is not None and attn_mask.requires_grad:
        raise ValueError('This experimental path does not support differentiable attention masks')
    outputs = []
    for start in range(0, q.shape[1], heads_per_chunk):
        stop = start + heads_per_chunk
        mask = attn_mask
        if mask is not None and mask.ndim >= 3 and mask.shape[-3] == q.shape[1]:
            mask = mask[..., start:stop, :, :]
        outputs.append(_Attention.apply(q[:,start:stop], k[:,start:stop], v[:,start:stop], mask))
    return torch.cat(outputs, dim=1)
