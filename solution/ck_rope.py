"""K2 — our own fused rotary-embedding pass (Triton).

Patches ``transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb``.
For q of shape (B, H, S, D) and cos/sin of shape (B, S, D) (fp32, with the
standard duplicated halves) the kernel writes
``y_d = x_d*c_d - x_{d+H}*s_d`` and ``y_{d+H} = x_{d+H}*c_d + x_d*s_d``
for the first half ``d < D/2`` — in fp32 math, fp16 out, matching the fp32
intermediate of the reference implementation.

Backward uses the saved cos/sin only:
``dx_d = dy_d*c_d + dy_{d+H}*s_d``, ``dx_{d+H} = dy_{d+H}*c_d - dy_d*s_d``.
"""

import torch
import triton
import triton.language as tl

_ROPE_ROWS = 16  # sequence positions handled by one program


@triton.jit
def _ck_rope_fwd_kernel(X, COS, SIN, Y, S, H, RD: tl.constexpr, BR: tl.constexpr):
    p0 = tl.program_id(0)  # b*Hn + h
    p1 = tl.program_id(1)
    s = p1 * BR + tl.arange(0, BR)
    smask = s < S
    b = p0 // H
    d = tl.arange(0, RD // 2)
    xrow = (p0 * S + s) * RD
    crow = ((b * S) + s) * RD
    x1 = tl.load(X + xrow[:, None] + d[None, :], mask=smask[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(X + xrow[:, None] + d[None, :] + RD // 2, mask=smask[:, None], other=0.0).to(tl.float32)
    c = tl.load(COS + crow[:, None] + d[None, :], mask=smask[:, None], other=0.0).to(tl.float32)
    sn = tl.load(SIN + crow[:, None] + d[None, :], mask=smask[:, None], other=0.0).to(tl.float32)
    y1 = x1 * c - x2 * sn
    y2 = x2 * c + x1 * sn
    tl.store(Y + xrow[:, None] + d[None, :], y1.to(tl.float16), mask=smask[:, None])
    tl.store(Y + xrow[:, None] + d[None, :] + RD // 2, y2.to(tl.float16), mask=smask[:, None])


@triton.jit
def _ck_rope_bwd_kernel(DX, COS, SIN, GX, S, H, RD: tl.constexpr, BR: tl.constexpr):
    p0 = tl.program_id(0)
    p1 = tl.program_id(1)
    s = p1 * BR + tl.arange(0, BR)
    smask = s < S
    b = p0 // H
    d = tl.arange(0, RD // 2)
    xrow = (p0 * S + s) * RD
    crow = ((b * S) + s) * RD
    dy1 = tl.load(DX + xrow[:, None] + d[None, :], mask=smask[:, None], other=0.0).to(tl.float32)
    dy2 = tl.load(DX + xrow[:, None] + d[None, :] + RD // 2, mask=smask[:, None], other=0.0).to(tl.float32)
    c = tl.load(COS + crow[:, None] + d[None, :], mask=smask[:, None], other=0.0).to(tl.float32)
    sn = tl.load(SIN + crow[:, None] + d[None, :], mask=smask[:, None], other=0.0).to(tl.float32)
    gx1 = dy1 * c + dy2 * sn
    gx2 = dy2 * c - dy1 * sn
    tl.store(GX + xrow[:, None] + d[None, :], gx1.to(tl.float16), mask=smask[:, None])
    tl.store(GX + xrow[:, None] + d[None, :] + RD // 2, gx2.to(tl.float16), mask=smask[:, None])


def _rope_one(x, cos, sin, back, grad=None):
    """x: (B, H, S, RD) fp16 (made contiguous); cos/sin: (B, S, RD) fp32 or fp16."""
    B, H, S, RD = x.shape
    src = grad if back else x
    if not src.is_contiguous():
        src = src.contiguous()
    out = torch.empty_like(src)
    grid = (B * H, (S + _ROPE_ROWS - 1) // _ROPE_ROWS)
    kern = _ck_rope_bwd_kernel if back else _ck_rope_fwd_kernel
    kern[grid](src, cos, sin, out, S, H, RD=RD, BR=_ROPE_ROWS, num_warps=4)
    return out


class CKRopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        qy = _rope_one(q, cos, sin, back=False)
        ky = _rope_one(k, cos, sin, back=False)
        ctx.save_for_backward(cos, sin)
        return qy, ky

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        gx_q = _rope_one(dq, cos, sin, back=True, grad=dq)
        gx_k = _rope_one(dk, cos, sin, back=True, grad=dk)
        return gx_q, gx_k, None, None


def _rope_cos3(cos):
    """Callers usually pass (B, S, RD); accept an optional (B, 1, S, RD) too."""
    if cos.dim() == 3:
        return cos
    if cos.dim() == 4 and cos.shape[1] == 1:
        return cos[:, 0]
    return None


def ck_rope(q, k, cos, sin):
    if not (q.is_cuda and q.dtype == torch.float16 and k.dtype == torch.float16):
        return None
    if q.dim() != 4 or q.shape != k.shape:
        return None
    cos, sin = _rope_cos3(cos), _rope_cos3(sin)
    if cos is None or sin is None or cos.dtype != sin.dtype:
        return None
    if cos.dtype not in (torch.float16, torch.float32):
        return None
    if cos.shape[0] != q.shape[0] or cos.shape[1] != q.shape[2] \
            or cos.shape[-1] != q.shape[-1] or cos.shape != sin.shape:
        return None
    rd = q.shape[-1]
    if rd % 64 != 0:
        return None
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()
    return CKRopeFn.apply(q, k, cos, sin)


def patch_qwen2_rope():
    import transformers.models.qwen2.modeling_qwen2 as qm

    orig = qm.apply_rotary_pos_emb
    if getattr(orig, "_ck_is_ours", False):
        return

    def patched(q, k, cos, sin, unsqueeze_dim=1):
        try:
            out = ck_rope(q, k, cos, sin)
        except Exception:
            import traceback
            traceback.print_exc()
            out = None
        if out is not None:
            return out
        return orig(q, k, cos, sin, unsqueeze_dim)

    patched._ck_is_ours = True
    qm.apply_rotary_pos_emb = patched
