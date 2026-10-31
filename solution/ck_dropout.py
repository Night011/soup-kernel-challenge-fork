"""Pack native PyTorch dropout masks without changing their random stream.

Native dropout still generates the exact fp32 output and mask. Our forward
combines the following projection's fp16 cast with lossless mask packing.
Backward unpacks the bits and writes the original fp32 input gradient.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ck_dropout_pack_kernel(VALUE, MASK, OUT, BITS, N, INVERSE, H,
                            COMPACT: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    i = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(VALUE + i, i < N, other=0.0)
    keep = tl.load(MASK + i, i < N, other=0).to(tl.int32)
    if COMPACT:
        row = tl.load(INVERSE + i // H, i < N, other=-1)
        tl.store(OUT + tl.maximum(row, 0) * H + i % H,
                 value.to(tl.float16), (i < N) & (row >= 0))
    else:
        tl.store(OUT + i, value.to(tl.float16), i < N)
    groups = tl.reshape(keep, (BLOCK // 8, 8))
    shifts = tl.arange(0, 8)
    encoded = tl.sum(groups << shifts[None, :], axis=1)
    byte = block * (BLOCK // 8) + tl.arange(0, BLOCK // 8)
    tl.store(BITS + byte, encoded.to(tl.uint8), byte * 8 < N)


@triton.jit
def _ck_dropout_bwd_kernel(GRAD, BITS, DX, N, SCALE, INVERSE, H,
                           COMPACT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    encoded = tl.load(BITS + i // 8, i < N, other=0).to(tl.int32)
    keep = (encoded >> (i % 8)) & 1
    if COMPACT:
        row = tl.load(INVERSE + i // H, i < N, other=-1)
        grad = tl.load(GRAD + tl.maximum(row, 0) * H + i % H,
                       (i < N) & (row >= 0), other=0.0).to(tl.float32)
    else:
        grad = tl.load(GRAD + i, i < N, other=0.0).to(tl.float32)
    dx = grad * keep.to(tl.float32) * SCALE
    tl.store(DX + i, dx, i < N)


class _CompactDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, layout=None):
        values, mask = torch.ops.aten.native_dropout(x, p, True)
        shape = (layout[0].numel(), x.shape[-1]) if layout is not None else x.shape
        out = torch.empty(shape, device=x.device, dtype=torch.float16)
        bits = torch.empty(triton.cdiv(x.numel(), 8), dtype=torch.uint8, device=x.device)
        inverse = layout[2] if layout is not None else bits
        _ck_dropout_pack_kernel[(triton.cdiv(x.numel(), 1024),)](
            values, mask, out, bits, x.numel(), inverse, x.shape[-1],
            COMPACT=layout is not None, BLOCK=1024, num_warps=4,
        )
        ctx.save_for_backward(bits, inverse)
        ctx.compact = layout is not None
        ctx.shape, ctx.scale = x.shape, 1.0 / (1.0 - p)
        return out

    @staticmethod
    def backward(ctx, grad):
        bits, inverse = ctx.saved_tensors
        grad = grad.contiguous()
        dx = torch.empty(ctx.shape, dtype=torch.float32, device=grad.device)
        _ck_dropout_bwd_kernel[(triton.cdiv(dx.numel(), 1024),)](
            grad, bits, dx, dx.numel(), ctx.scale, inverse, ctx.shape[-1],
            COMPACT=ctx.compact, BLOCK=1024, num_warps=4,
        )
        return (dx, None, None)[:len(ctx.needs_input_grad)]


def compact_lora_dropout(module, x, layout=None):
    p = getattr(module, "p", None)
    if (p is not None and 0.0 < p < 1.0 and getattr(module, "training", False)
            and not getattr(module, "inplace", False)
            and x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
            and torch.is_autocast_enabled()
            and torch.get_autocast_dtype("cuda") == torch.float16):
        return _CompactDropout.apply(x, float(p), layout)
    out = module(x)
    if layout is not None:
        out = out.reshape(-1, out.shape[-1]).index_select(0, layout[0])
    return out
