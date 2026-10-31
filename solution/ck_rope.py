"""K2 — fused rotary position embedding (RoPE) forward and backward in Triton.

The patch replaces ``transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb``.
Qwen2 passes q as ``(B, Hq, S, D)`` and k as ``(B, Hk, S, D)``; both are views
produced by ``view(B, S, H, D).transpose(1, 2)``, so their last dimension is
contiguous while the sequence and head strides are not.  The kernel takes those
strides explicitly, which lets grouped-query attention (Hq != Hk) run without
the ``.contiguous()`` copies that made the first GQA attempt slower than the
stock path.

For ``d < D/2`` the rotation is

    y[d]     = x[d] * cos[d] - x[d + D/2] * sin[d]
    y[d+D/2] = x[d+D/2] * cos[d] + x[d] * sin[d]

and the backward is its transpose:

    dx[d]     = dy[d] * cos[d] + dy[d+D/2] * sin[d]
    dx[d+D/2] = dy[d+D/2] * cos[d] - dy[d] * sin[d]

cos/sin are ``(B, S, D)`` (or a broadcast ``(1, S, D)``); the kernel accepts
their batch/sequence strides, so an expanded table is not copied.  The math is
fp32 and the result is rounded once to the output dtype, which stays inside the
fp16 tolerance of the reference.
"""

import torch
import triton
import triton.language as tl

_ROPE_ROWS = 16          # sequence positions handled by one program
_ROPE_WARNED = False     # print the fallback diagnostic only once


@triton.jit
def _ck_rope_fwd_kernel(
    X, COS, SIN, Y,
    S, H,
    XB, XH, XS,
    CB, CS,
    SB, SS,
    RD: tl.constexpr, BR: tl.constexpr,
):
    pid = tl.program_id(0)          # b * H + h
    blk = tl.program_id(1)
    b = pid // H
    h = pid - b * H
    s = blk * BR + tl.arange(0, BR)
    smask = s < S
    d = tl.arange(0, RD // 2)

    x_base = b * XB + h * XH + s * XS
    x1 = tl.load(X + x_base[:, None] + d[None, :],
                 mask=smask[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(X + x_base[:, None] + d[None, :] + RD // 2,
                 mask=smask[:, None], other=0.0).to(tl.float32)

    c_base = b * CB + s * CS
    c = tl.load(COS + c_base[:, None] + d[None, :],
                mask=smask[:, None], other=0.0).to(tl.float32)
    sn = tl.load(SIN + c_base[:, None] + d[None, :],
                 mask=smask[:, None], other=0.0).to(tl.float32)

    y1 = x1 * c - x2 * sn
    y2 = x2 * c + x1 * sn

    # The output is a fresh contiguous (B, H, S, RD) buffer, so its strides
    # are fixed and do not need to be passed separately.
    y_base = ((b * H + h) * S + s) * RD
    tl.store(Y + y_base[:, None] + d[None, :],
             y1.to(Y.dtype.element_ty), mask=smask[:, None])
    tl.store(Y + y_base[:, None] + d[None, :] + RD // 2,
             y2.to(Y.dtype.element_ty), mask=smask[:, None])


@triton.jit
def _ck_rope_bwd_kernel(
    DY, COS, SIN, DX,
    S, H,
    GB, GH, GS,
    CB, CS,
    SB, SS,
    RD: tl.constexpr, BR: tl.constexpr,
):
    pid = tl.program_id(0)
    blk = tl.program_id(1)
    b = pid // H
    h = pid - b * H
    s = blk * BR + tl.arange(0, BR)
    smask = s < S
    d = tl.arange(0, RD // 2)

    g_base = b * GB + h * GH + s * GS
    dy1 = tl.load(DY + g_base[:, None] + d[None, :],
                  mask=smask[:, None], other=0.0).to(tl.float32)
    dy2 = tl.load(DY + g_base[:, None] + d[None, :] + RD // 2,
                  mask=smask[:, None], other=0.0).to(tl.float32)

    c_base = b * CB + s * CS
    c = tl.load(COS + c_base[:, None] + d[None, :],
                mask=smask[:, None], other=0.0).to(tl.float32)
    sn = tl.load(SIN + c_base[:, None] + d[None, :],
                 mask=smask[:, None], other=0.0).to(tl.float32)

    dx1 = dy1 * c + dy2 * sn
    dx2 = dy2 * c - dy1 * sn

    x_base = ((b * H + h) * S + s) * RD
    tl.store(DX + x_base[:, None] + d[None, :],
             dx1.to(DX.dtype.element_ty), mask=smask[:, None])
    tl.store(DX + x_base[:, None] + d[None, :] + RD // 2,
             dx2.to(DX.dtype.element_ty), mask=smask[:, None])


def _table3(t):
    """Accept (B, S, D) or the optional (B, 1, S, D) layout."""
    if t.dim() == 3:
        return t
    if t.dim() == 4 and t.shape[1] == 1:
        return t[:, 0]
    return None


def _dense_table(t, batch, seq, dim):
    """Normalise cos/sin to a (batch, seq, dim) table with last stride 1.

    ``expand`` keeps a zero batch stride, which the kernel reads correctly, so
    a broadcast table does not turn into a per-layer copy.
    """
    if t is None or t.dim() != 3:
        return None
    if t.shape[0] == 1 and batch != 1:
        t = t.expand(batch, -1, -1)
    if t.shape[0] != batch or t.shape[1] != seq or t.shape[2] != dim:
        return None
    if t.stride(-1) != 1:
        t = t.contiguous()
    return t


def _rope_one(src, cos, sin, backward):
    """Run one direction of the rotation for q or k.

    ``src`` may be a transposed (B, H, S, D) view; only its last stride has to
    be 1.  Backward gradients are made contiguous if they arrive with a
    different last stride, because silently returning None there would drop the
    RoPE gradient.
    """
    if src is None:
        return None
    if not src.is_cuda or src.dtype not in (torch.float16, torch.float32):
        return None
    if src.dim() != 4 or src.shape[-1] % 64 != 0:
        return None
    if src.stride(-1) != 1:
        if not backward:
            return None
        src = src.contiguous()

    batch, heads, seq, dim = src.shape
    out = torch.empty(src.shape, dtype=src.dtype, device=src.device)
    xb, xh, xs, _ = src.stride()
    cb, cs, _ = cos.stride()
    sb, ss, _ = sin.stride()

    grid = (batch * heads, triton.cdiv(seq, _ROPE_ROWS))
    kernel = _ck_rope_bwd_kernel if backward else _ck_rope_fwd_kernel
    kernel[grid](
        src, cos, sin, out,
        seq, heads,
        xb, xh, xs,
        cb, cs,
        sb, ss,
        RD=dim, BR=_ROPE_ROWS, num_warps=4,
    )
    return out


class CKRopeFn(torch.autograd.Function):
    """q/k rotation with the exact inverse-rotation backward."""

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q_out = _rope_one(q, cos, sin, backward=False)
        k_out = _rope_one(k, cos, sin, backward=False)
        if q_out is None or k_out is None:
            # ck_rope validated the layout before apply(); this only catches a
            # late change (for example a different tensor subclass).
            raise RuntimeError("CK RoPE layout is no longer supported")
        ctx.save_for_backward(cos, sin)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        gq = _rope_one(dq, cos, sin, backward=True)
        gk = _rope_one(dk, cos, sin, backward=True)
        return gq, gk, None, None


def ck_rope(q, k, cos, sin):
    """Triton path for the Qwen2 layout, or None to keep the stock function."""
    if not (q.is_cuda and k.is_cuda):
        return None
    if q.dtype not in (torch.float16, torch.float32) \
            or k.dtype not in (torch.float16, torch.float32):
        return None
    if q.dim() != 4 or k.dim() != 4:
        return None
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] \
            or q.shape[3] != k.shape[3]:
        return None
    batch, _, seq, dim = q.shape
    if dim % 64 != 0:
        return None
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        return None

    cos, sin = _table3(cos), _table3(sin)
    if cos is None or sin is None:
        return None
    if cos.dtype != sin.dtype or cos.dtype not in (torch.float16, torch.float32):
        return None
    cos = _dense_table(cos, batch, seq, dim)
    sin = _dense_table(sin, batch, seq, dim)
    if cos is None or sin is None:
        return None
    return CKRopeFn.apply(q, k, cos, sin)


def patch_qwen2_rope():
    import transformers.models.qwen2.modeling_qwen2 as qm

    orig = qm.apply_rotary_pos_emb
    if getattr(orig, "_ck_is_ours", False):
        return

    def patched(q, k, cos, sin, unsqueeze_dim=1):
        global _ROPE_WARNED
        try:
            out = ck_rope(q, k, cos, sin)
        except Exception as exc:  # noqa: BLE001 - a kernel failure must not stop training
            if not _ROPE_WARNED:
                _ROPE_WARNED = True
                import traceback
                traceback.print_exc()
                print(f"[ck] rope fallback: {exc.__class__.__name__}: {exc}",
                      flush=True)
            out = None
        if out is not None:
            return out
        return orig(q, k, cos, sin, unsqueeze_dim)

    patched._ck_is_ours = True
    qm.apply_rotary_pos_emb = patched
