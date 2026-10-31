"""Fused SwiGLU (silu(gate) * up) in Triton, forward and backward.

This is a 2-D tiled adaptation of the starter-kit 1-D example. Its forward and
backward kernels are explicitly marked as examples and do not count toward
the challenge's own-kernel time requirement. The math is the same fp32
formula the example/stock path uses:

    out   = gate * sigmoid(gate) * up
    dup   = grad * gate * sigmoid(gate)
    dgate = grad * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))

and the result is rounded once to the output dtype.  The Qwen2 MLP patch is
kept at the bottom; if the activation is not SiLU or the tensor is not on CUDA,
the stock forward is used unchanged.
"""

import torch
import triton
import triton.language as tl
from .ck_activation import PACK_BLOCK, pack_activations

# 2-D tile: two rows x 2048 columns = 4096 elements per program.  The hidden
# size of Qwen2.5-1.5B is 8960, so the column grid has a masked tail.
_SW_BM = 2
_SW_BN = 2048


@triton.jit
def example_swiglu_fwd_kernel(GATE, UP, OUT, M, N,
                          BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offs = rows[:, None] * N + cols[None, :]

    gate = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + offs, mask=mask, other=0.0).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(OUT + offs, out.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def example_swiglu_bwd_kernel(GRAD, GATE, UP, SGATE, SUP, DGATE, DUP, M, N,
                          BM: tl.constexpr, BN: tl.constexpr,
                          COMPRESSED: tl.constexpr, PACK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offs = rows[:, None] * N + cols[None, :]

    grad = tl.load(GRAD + offs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + offs, mask=mask, other=0.0).to(tl.float32)
    if COMPRESSED:
        sg = tl.load(SGATE + offs // PACK, mask=mask, other=0.0)
        su = tl.load(SUP + offs // PACK, mask=mask, other=0.0)
        gate *= sg
        up *= su

    sig = tl.sigmoid(gate)
    dup = grad * gate * sig
    dgate = grad * up * sig * (1.0 + gate * (1.0 - sig))
    tl.store(DGATE + offs, dgate.to(DGATE.dtype.element_ty), mask=mask)
    tl.store(DUP + offs, dup.to(DUP.dtype.element_ty), mask=mask)


class SwiGLU(torch.autograd.Function):
    """silu(gate) * up in one launch each way instead of several PyTorch ops."""

    @staticmethod
    def forward(ctx, gate, up, compress):
        gate = gate.contiguous()
        up = up.to(gate.dtype).contiguous()
        rows = gate.numel() // gate.shape[-1]
        cols = gate.shape[-1]
        out = torch.empty_like(gate)
        grid = (triton.cdiv(rows, _SW_BM), triton.cdiv(cols, _SW_BN))
        example_swiglu_fwd_kernel[grid](
            gate, up, out, rows, cols,
            BM=_SW_BM, BN=_SW_BN, num_warps=8, num_stages=2,
        )
        ctx.compressed = bool(compress and gate.dtype == torch.float16)
        ctx.dtype = gate.dtype
        if ctx.compressed:
            ctx.save_for_backward(*pack_activations(gate, up))
        else:
            ctx.save_for_backward(gate, up, gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up, sg, su = ctx.saved_tensors
        grad = grad.to(ctx.dtype).contiguous()
        rows = gate.numel() // gate.shape[-1]
        cols = gate.shape[-1]
        dgate = torch.empty_like(gate, dtype=ctx.dtype)
        dup = torch.empty_like(up, dtype=ctx.dtype)
        grid = (triton.cdiv(rows, _SW_BM), triton.cdiv(cols, _SW_BN))
        example_swiglu_bwd_kernel[grid](
            grad, gate, up, sg, su, dgate, dup, rows, cols,
            BM=_SW_BM, BN=_SW_BN, COMPRESSED=ctx.compressed, PACK=PACK_BLOCK,
            num_warps=8, num_stages=2,
        )
        return dgate, dup, None


def swiglu(gate, up, compress=False):
    return SwiGLU.apply(gate, up, compress)


def patch_qwen2_mlp():
    """Send Qwen2MLP.forward through the fused kernel. A class patch works before or after Soup
    builds the model: Python looks the method up on every call."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not x.is_cuda:
            return original(self, x)
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x), compress=True))

    modeling_qwen2.Qwen2MLP.forward = forward
