"""Rejected checkpoint experiment: caused shared-memory paging on the RX 6800.

Preserved for reproducibility. It is deliberately absent from the driver and
the benchmark's default variant list.
"""
import torch
from torch.utils.checkpoint import checkpoint
from fizgig.krea2.model import SingleStreamBlock, RMSNorm


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, eps):
        ctx.save_for_backward(x, scale)
        ctx.eps = eps
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            inv = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
            return (xf * inv * (scale.float()+1)).to(x.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        x, scale = ctx.saved_tensors
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf, gf = x.float(), grad.float()
            inv = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + ctx.eps)
            gs = gf * (scale.float()+1)
            dx = (gs - xf * (xf*gs).mean(dim=-1,keepdim=True) * inv.square()) * inv
            ds = (gf*xf*inv).sum(dim=tuple(range(x.ndim-1)))
            return dx.to(x.dtype), ds.to(scale.dtype), None


def norm_forward(self, x):
    return _Norm.apply(x, self.scale, self.eps)


def mlp_checkpoint_forward(self, x, vec, freqs, attn_params=None):
    prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
    x = x + pregate * self.attn((1+prescale)*self.prenorm(x)+preshift, freqs, attn_params)
    def mlp_part(value, scale, shift, gate):
        return gate * self.mlp((1+scale)*self.postnorm(value)+shift)
    if self.training and torch.is_grad_enabled():
        return x + checkpoint(mlp_part, x, postscale, postshift, postgate, use_reentrant=False)
    return x + mlp_part(x, postscale, postshift, postgate)


def install(model):
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.forward = norm_forward.__get__(module, type(module))
        if isinstance(module, SingleStreamBlock) and module in model.blocks:
            module.forward = mlp_checkpoint_forward.__get__(module, type(module))
            module._handles_checkpointing = True
