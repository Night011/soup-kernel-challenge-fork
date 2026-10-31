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


class CKRopeFn(torch.autograd.Function):
    """q/k rotation with the exact inverse-rotation backward."""

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q_out, k_out = _rope_pair(q, k, cos, sin, backward=False)
        ctx.save_for_backward(cos, sin)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        gq, gk = _rope_pair(dq, dk, cos, sin, backward=True)
        return gq, gk, None, None


@triton.jit
def _ck_rope_pair_kernel(Q, K, COS, SIN, OQ, OK, S, HQ, HK,
                          QB, QH, QS, KB, KH, KS, CB, CS, SB, SS,
                          RD: tl.constexpr, BR: tl.constexpr,
                          BACKWARD: tl.constexpr):
    combined = tl.program_id(0)
    batch = combined // (HQ + HK)
    head = combined % (HQ + HK)
    if head < HQ:
        source, destination = Q, OQ
        source_base = batch * QB + head * QH
        source_stride = QS
        output_base = (batch * HQ + head) * S * RD
    else:
        source, destination = K, OK
        source_base = batch * KB + (head - HQ) * KH
        source_stride = KS
        output_base = (batch * HK + head - HQ) * S * RD
    sequence = tl.program_id(1) * BR + tl.arange(0, BR)
    dimension = tl.arange(0, RD // 2)
    source_offset = source_base + sequence[:, None] * source_stride + dimension[None, :]
    mask = sequence[:, None] < S
    first = tl.load(source + source_offset, mask, other=0).to(tl.float32)
    second = tl.load(source + source_offset + RD // 2, mask, other=0).to(tl.float32)
    cosine = tl.load(COS + batch * CB + sequence[:, None] * CS + dimension[None, :],
                     mask, other=0).to(tl.float32)
    sine = tl.load(SIN + batch * SB + sequence[:, None] * SS + dimension[None, :],
                   mask, other=0).to(tl.float32)
    if BACKWARD:
        out_first = first * cosine + second * sine
        out_second = second * cosine - first * sine
    else:
        out_first = first * cosine - second * sine
        out_second = second * cosine + first * sine
    output_offset = output_base + sequence[:, None] * RD + dimension[None, :]
    tl.store(destination + output_offset, out_first, mask)
    tl.store(destination + output_offset + RD // 2, out_second, mask)


def _rope_pair(q, k, cos, sin, backward):
    if q.dtype != k.dtype:
        raise RuntimeError("Joint RoPE requires equal q/k dtypes")
    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_out = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    batch, hq, sequence, dimension = q.shape
    hk = k.shape[1]
    _ck_rope_pair_kernel[(batch * (hq + hk), triton.cdiv(sequence, _ROPE_ROWS))](
        q, k, cos, sin, q_out, k_out, sequence, hq, hk,
        *q.stride()[:3], *k.stride()[:3], *cos.stride()[:2], *sin.stride()[:2],
        RD=dimension, BR=_ROPE_ROWS, BACKWARD=backward, num_warps=4,
    )
    return q_out, k_out


def ck_rope(q, k, cos, sin):
    """Triton path for the Qwen2 layout, or None to keep the stock function."""
    if not (q.is_cuda and k.is_cuda):
        return None
    if q.dtype not in (torch.float16, torch.float32) \
            or k.dtype not in (torch.float16, torch.float32):
        return None
    if q.dtype != k.dtype:
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
