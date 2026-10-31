"""Fuse attention residual, compact MLP RMSNorm and their joint backward.

The full fp32 residual is still returned for the next skip connection. Only
the token-wise MLP norm is compact. Its saved fp16 input and exact fp32 rstd
use the same storage policy as ck_rms, without gather/scatter temporaries.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ck_residual_norm_fwd_kernel(R, A, W, INVERSE, FULL, NORM, SAVED, RSTD,
                                 EPS, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    offsets = row * H + columns
    residual = tl.load(R + offsets, columns < H, other=0).to(tl.float32)
    attention = tl.load(A + offsets, columns < H, other=0).to(tl.float32)
    value = residual + attention
    tl.store(FULL + offsets, value, columns < H)
    packed = tl.load(INVERSE + row)
    if packed >= 0:
        weight = tl.load(W + columns, columns < H, other=0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(value * value, 0) / H + EPS)
        normalized = weight * (value * rstd)
        packed_offsets = packed * H + columns
        tl.store(NORM + packed_offsets, normalized, columns < H)
        tl.store(SAVED + packed_offsets, value.to(tl.float16), columns < H)
        tl.store(RSTD + packed, rstd)


@triton.jit
def _ck_residual_norm_bwd_kernel(GFULL, GNORM, W, INVERSE, SAVED, RSTD, DR, DA,
                                 H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    packed = tl.load(INVERSE + row)
    packed_offsets = tl.maximum(packed, 0) * H + columns
    mask = (columns < H) & (packed >= 0)
    value = tl.load(SAVED + packed_offsets, mask, other=0).to(tl.float32)
    dy = tl.load(GNORM + packed_offsets, mask, other=0).to(tl.float32)
    weight = tl.load(W + columns, columns < H, other=0).to(tl.float32)
    rstd = tl.load(RSTD + tl.maximum(packed, 0), packed >= 0, other=0)
    dot = tl.sum(value * weight * dy, 0)
    derivative = rstd * weight * dy - (rstd * rstd * rstd / H) * value * dot
    offsets = row * H + columns
    full_grad = tl.load(GFULL + offsets, columns < H, other=0).to(tl.float32)
    total = full_grad + derivative
    tl.store(DR + offsets, total, columns < H)
    tl.store(DA + offsets, total, columns < H)


class ResidualCompactNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, attention, weight, eps, rows, inverse):
        residual, attention = residual.contiguous(), attention.contiguous()
        hidden = residual.shape[-1]
        full = torch.empty_like(residual)
        norm = torch.empty((rows.numel(), hidden), device=residual.device,
                            dtype=torch.float16)
        saved = torch.empty_like(norm)
        rstd = torch.empty(rows.numel(), device=residual.device, dtype=torch.float32)
        _ck_residual_norm_fwd_kernel[(residual.numel() // hidden,)](
            residual, attention, weight, inverse, full, norm, saved, rstd, eps,
            H=hidden, BLOCK=triton.next_power_of_2(hidden), num_warps=4,
        )
        ctx.save_for_backward(weight, inverse, saved, rstd)
        ctx.shape, ctx.attention_dtype = residual.shape, attention.dtype
        return full, norm

    @staticmethod
    def backward(ctx, grad_full, grad_norm):
        weight, inverse, saved, rstd = ctx.saved_tensors
        grad_full, grad_norm = grad_full.contiguous(), grad_norm.contiguous()
        dr = torch.empty(ctx.shape, device=grad_full.device, dtype=torch.float32)
        da = torch.empty_like(dr, dtype=ctx.attention_dtype)
        hidden = dr.shape[-1]
        _ck_residual_norm_bwd_kernel[(dr.numel() // hidden,)](
            grad_full, grad_norm, weight, inverse, saved, rstd, dr, da,
            H=hidden, BLOCK=triton.next_power_of_2(hidden), num_warps=4,
        )
        return dr, da, None, None, None, None


def residual_compact_norm(residual, attention, norm, layout):
    if (residual.dtype != torch.float32 or attention.dtype not in (torch.float16, torch.float32)
            or not residual.is_cuda or attention.shape != residual.shape
            or not getattr(norm, "_ck_output_fp16", False)
            or norm.weight.requires_grad or norm.weight.numel() != residual.shape[-1]
            or residual.shape[-1] % 512 != 0):
        return None
    return ResidualCompactNorm.apply(
        residual, attention, norm.weight, norm.variance_epsilon, layout[0], layout[2])
