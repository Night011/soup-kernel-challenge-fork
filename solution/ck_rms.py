"""K1 — our own fused RMSNorm forward/backward (Triton).

Replaces ``transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm.forward``
with an autograd pair backed by two ``@triton.jit`` kernels:

* ``_ck_rms_fwd``  — one launch per row-block: variance, rsqrt and the
  ``weight * x * rsqrt(mean(x^2) + eps)`` product, emitted directly in fp16 so
  the downstream projections see exactly the values the fp16 autocast cast of
  the fp32 reference output would produce (residual stream stays fp32).
* ``_ck_rms_bwd``  — ``dx_k = r*w_k*dy_k - (r^3 / H) * x_k * sum_d(w_d*dy_d*x_d)``
  with ``r = rsqrt(mean(x^2)+eps)`` recomputed from the saved input.

Guards: CUDA only, contiguous input in fp16/fp32 (the Soup T4 flow has an fp16
residual stream with an fp16 weight — fp32 straight-loading is also accepted),
frozen weight, hidden size divisible by the kernel tile — otherwise the
original forward is used unchanged.
"""

import torch
import triton
import triton.language as tl

_RMS_BLOCK = 512
_RMS_WARNED = False


@triton.jit
def _ck_rms_fwd_kernel(X, W, Y, eps, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * H
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(H // BLOCK):
        x = tl.load(X + base + i * BLOCK + offs).to(tl.float32)
        acc += x * x
    var = tl.sum(acc, axis=0) / H
    r = tl.rsqrt(var + eps)
    for i in tl.static_range(H // BLOCK):
        d = i * BLOCK + offs
        x = tl.load(X + base + d).to(tl.float32)
        w = tl.load(W + d).to(tl.float32)
        y = w * (x * r)
        tl.store(Y + base + d, y.to(Y.dtype.element_ty))


# y_j = w_j * x_j * r, где r = (mean(x^2) + eps)^(-1/2).
# Производная по x_k:
#   dL/dx_k = w_k * dy_k * r
#             - (r^3 / H) * x_k * sum_j (w_j * dy_j * x_j).
# Второе слагаемое — это dr/dx_k = -(r^3 / H) * x_k, умноженное на сумму
# по всем j. Поэтому первый цикл копит sum(x^2) и sum(w*dy*x), а второй
# цикл уже пишет dx. Оба цикла идут по H с шагом BLOCK, чтобы не держать
# весь вектор строки в регистрах.
@triton.jit
def _ck_rms_bwd_kernel(X, DY, W, DX, eps, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * H
    offs = tl.arange(0, BLOCK)
    acc_xx = tl.zeros([BLOCK], dtype=tl.float32)
    acc_gx = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(H // BLOCK):
        d = i * BLOCK + offs
        x = tl.load(X + base + d).to(tl.float32)
        dy = tl.load(DY + base + d).to(tl.float32)
        w = tl.load(W + d).to(tl.float32)
        acc_xx += x * x
        acc_gx += x * w * dy
    m = tl.sum(acc_xx, axis=0) / H
    s = tl.sum(acc_gx, axis=0)
    r = tl.rsqrt(m + eps)
    # coef = r^3 / H; второе слагаемое — вклад производной r по x_k.
    coef = r * r * r / H
    for i in tl.static_range(H // BLOCK):
        d = i * BLOCK + offs
        x = tl.load(X + base + d).to(tl.float32)
        dy = tl.load(DY + base + d).to(tl.float32)
        w = tl.load(W + d).to(tl.float32)
        dx = r * w * dy - coef * x * s
        tl.store(DX + base + d, dx.to(DX.dtype.element_ty))


class CKRMSNormFn(torch.autograd.Function):
    """y = w * x * rsqrt(mean(x^2)+eps); output in the input's dtype, backward dx same."""

    @staticmethod
    def forward(ctx, x, weight, eps, output_fp16=False):
        rows = x.numel() // weight.numel()
        hidden = weight.numel()
        y = torch.empty_like(x, dtype=torch.float16 if output_fp16 else x.dtype).view(rows, hidden)
        _ck_rms_fwd_kernel[(rows,)](
            x, weight, y, eps, H=hidden, BLOCK=_RMS_BLOCK, num_warps=4, num_stages=2
        )
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return y.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        rows = x.numel() // weight.numel()
        hidden = weight.numel()
        grad = grad_out.contiguous().view(rows, hidden)
        dx = torch.empty_like(x).view(rows, hidden)
        _ck_rms_bwd_kernel[(rows,)](
            x, grad, weight, dx, ctx.eps, H=hidden, BLOCK=_RMS_BLOCK,
            num_warps=4, num_stages=2,
        )
        return (dx.view_as(x), None, None, None)[:len(ctx.needs_input_grad)]


def ck_rmsnorm(x, weight, eps, output_fp16=False):
    """Triton path when it applies (returns tensor), else None."""
    if not x.is_cuda or not x.is_contiguous():
        return None
    if x.dtype not in (torch.float16, torch.float32):
        return None
    hidden = weight.numel()
    if weight.dtype not in (torch.float16, torch.float32):
        return None
    if hidden != x.shape[-1] or hidden % _RMS_BLOCK != 0 or weight.requires_grad:
        return None
    if not weight.is_contiguous():
        return None
    return CKRMSNormFn.apply(x, weight, eps, output_fp16)


def patch_qwen2_rmsnorm():
    """Class-level patch of Qwen2RMSNorm.forward (guarded, idempotent)."""
    import transformers.models.qwen2.modeling_qwen2 as qm

    orig = qm.Qwen2RMSNorm.forward
    if getattr(orig, "_ck_is_ours", False):
        return

    def rms_forward(self, hidden_states):
        global _RMS_WARNED
        try:
            out = ck_rmsnorm(
                hidden_states, self.weight, self.variance_epsilon,
                getattr(self, "_ck_output_fp16", False),
            )
        except Exception as exc:  # noqa: BLE001 - never break the stock model
            if not _RMS_WARNED:
                _RMS_WARNED = True
                import traceback
                traceback.print_exc()
                print(f"[ck] rmsnorm fallback: {exc.__class__.__name__}: {exc}",
                      flush=True)
            out = None
        if out is not None:
            return out
        return orig(self, hidden_states)

    rms_forward._ck_is_ours = True
    qm.Qwen2RMSNorm.forward = rms_forward
