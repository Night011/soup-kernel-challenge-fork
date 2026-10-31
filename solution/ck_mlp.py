"""Paired frozen MLP projections with a manually chained backward.

Idea: combine gate/up matrices sharing an input, as described in NVIDIA's
TensorRT-LLM post-load fusion documentation. This is our own training wrapper;
no TensorRT-LLM code or kernels are imported. The activation kernels remain
marked as starter examples and are excluded from the own-kernel share.
"""

import torch
import triton

from . import ck_nf4
from .ck_activation import PACK_BLOCK


class _PairedFrozenMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed_gate, packed_up, packed_down, compress):
        from .swiglu import example_swiglu_fwd_kernel, _SW_BM, _SW_BN

        shape, hidden = x.shape, x.shape[-1]
        rows = x.numel() // hidden
        wgu = ck_nf4.paired_mlp_weights(packed_gate, packed_up)
        width = wgu.shape[0] // 2
        gu = torch.mm(x.view(rows, hidden), wgu.t())
        del wgu
        gate, up = gu[:, :width], gu[:, width:]
        activation = torch.empty((rows, width), dtype=x.dtype, device=x.device)
        grid = (triton.cdiv(rows, _SW_BM), triton.cdiv(width, _SW_BN))
        example_swiglu_fwd_kernel[grid](
            gate, up, activation, rows, width, 2 * width,
            BM=_SW_BM, BN=_SW_BN, num_warps=8, num_stages=2,
        )
        down = ck_nf4.decode_nf4(packed_down, packed_down.quant_state)
        result = torch.mm(activation, down.t())
        del down, activation
        recompute = compress
        if recompute:
            # Frozen projections can be replayed exactly from the much
            # narrower norm input instead of retaining both wide activations.
            saved_gate, saved_up, sg, su = x, x, x, x
        else:
            saved_gate, saved_up, sg, su = gate, up, gate, up
        ctx.save_for_backward(saved_gate, saved_up, sg, su, packed_gate, packed_up, packed_down)
        ctx.pair_states = (packed_gate.quant_state, packed_up.quant_state)
        ctx.down_state = packed_down.quant_state
        ctx.shape, ctx.width, ctx.compress = shape, width, False
        ctx.recompute = recompute
        return result.view(shape)

    @staticmethod
    def backward(ctx, grad):
        from .swiglu import example_swiglu_bwd_kernel, _SW_BM, _SW_BN

        gate, up, sg, su, packed_gate, packed_up, packed_down = ctx.saved_tensors
        hidden, width = ctx.shape[-1], ctx.width
        grad = grad.contiguous().view(-1, hidden)
        rows = grad.shape[0]
        wgu = None
        if ctx.recompute:
            wgu = ck_nf4.paired_mlp_weights(packed_gate, packed_up, ctx.pair_states)
            gu = torch.mm(gate.view(rows, hidden), wgu.t())
            gate, up = gu[:, :width], gu[:, width:]
        down = ck_nf4.decode_nf4(packed_down, ctx.down_state)
        grad_activation = torch.mm(grad, down)
        del down
        dgu = torch.empty((rows, 2 * width), dtype=grad.dtype, device=grad.device)
        grid = (triton.cdiv(rows, _SW_BM), triton.cdiv(width, _SW_BN))
        example_swiglu_bwd_kernel[grid](
            grad_activation, gate, up, sg, su, dgu[:, :width], dgu[:, width:],
            rows, width, width if ctx.compress else 2 * width, 2 * width,
            BM=_SW_BM, BN=_SW_BN, COMPRESSED=ctx.compress, PACK=PACK_BLOCK,
            num_warps=8, num_stages=2,
        )
        del gate, up, grad_activation
        if ctx.recompute:
            del gu
        if wgu is None:
            wgu = ck_nf4.paired_mlp_weights(packed_gate, packed_up, ctx.pair_states)
        dx = torch.mm(dgu, wgu)
        return dx.view(ctx.shape), None, None, None, None


def paired_frozen_mlp(mlp, x, compress=True):
    if not x.is_cuda or x.dtype != torch.float16 or not x.is_contiguous() or x.numel() == 0:
        return None
    weights = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        projection = getattr(mlp, name, None)
        weight = getattr(projection, "weight", None)
        state = getattr(weight, "quant_state", None)
        if (state is None or weight.requires_grad or hasattr(projection, "lora_A")
                or getattr(projection, "bias", None) is not None
                or state.dtype != torch.float16 or state.quant_type != "nf4"
                or state.blocksize != 64
                or (state.nested and (state.state2.blocksize != 256 or state.state2.nested))):
            return None
        weights.append(weight)
    gate, up, down = weights
    if (tuple(gate.quant_state.shape) != tuple(up.quant_state.shape)
            or gate.quant_state.shape[1] != x.shape[-1]
            or tuple(down.quant_state.shape) != (x.shape[-1], gate.quant_state.shape[0])):
        return None
    if not ck_nf4.can_pair_weights(gate, up):
        return None
    return _PairedFrozenMLP.apply(x, gate, up, down, compress)
