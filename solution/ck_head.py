"""K3 — our own chunked lm_head + CE + entropy + accuracy (Triton).

The reference path materializes a full fp16 logits tensor (4 x 512 x 151936)
plus fp32 copies for log_softmax / cross-entropy / entropy — a few GB of
transient buffers per step.  We keep only one 256-row logits piece alive.

Per forward step (B*S rows, vocab V, hidden K):

1. ``_ck_cast_kernel``       — frozen fp32 lm_head weight -> cached fp16 copy
                               (values are bit-identical to the autocast-cast
                               weight the fp16 reference matmul uses).
2. ``_ck_lse_store_kernel``  — one program per row, four streaming passes over
                               the row: max/argmax, sum(exp), copy the fp16
                               logits into the chunked logits buffer, and the
                               entropy sum.  It produces logsumexp, the first-
                               wins argmax, the per-token NLL and the raw
                               entropy without a full fp32 logits tensor.
3. ``_ck_dlogits_kernel``    — reads the stored fp16 logits piece and turns it
                               into the fp16 softmax gradient
                               ((p - onehot) / den), zeros at ignored rows.
4. ``dH = dlogits_piece @ W16`` stays a plain fp16 cuBLAS matmul, piece by
   piece, so piece state is freed immediately.

The large forward and backward GEMMs are cuBLAS: a handwritten large-vocab
Triton GEMM is much slower on a T4 and its compile time alone breaks the wall
clock.  Our Triton kernels own the full-vocab softmax/metrics work, which is
the part the reference does with several extra fp32 passes.

den is the cross-entropy denominator the Trainer already computes
(num_items_in_batch), so loss value and every gradient scale match the
reference exactly.  Per-token losses are exposed through a CKHeadStep so the
rewind recorder can produce row losses without logits.
"""

import torch
import triton
import triton.language as tl

# _BM rows x _BN columns — tile of the elementwise dlogits pass.
# _CH — rows per head piece (a 256-row logits piece is the only large temp).
# _BLOCK — vocab elements per streaming step in the LSE kernel.
_BM = 64
_BN = 64
_BK = 64
_CH = 256
_BLOCK = 256

_W16 = {"key": None, "tensor": None}


@triton.jit
def _ck_cast_kernel(SRC, DST, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK
    o = i + tl.arange(0, BLOCK)
    m = o < n
    tmp = tl.load(SRC + o, mask=m, other=0.0)
    tl.store(DST + o, tmp.to(tl.float16), mask=m)


@triton.jit
def _ck_lse_store_kernel(LOGITS, TGT, LSE, AMAX, NLL, ENT, LOGF, V,
                         BLOCK: tl.constexpr):
    """Per-row logsumexp + argmax + NLL + entropy, and a copy of the logits.

    Four passes over one row keep the kernel simple and numerically identical
    to the reference fp32 logsumexp: max/argmax, sum(exp), logits copy, and
    the entropy reduction.  LOGF is the chunked fp16 logits buffer consumed by
    the backward dlogits kernel, so the backward needs no second GEMM.
    """
    r = tl.program_id(0)
    row = LOGITS + r * V

    mx = float("-inf")
    amax = 0
    for v0 in range(0, V, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(row + offs, mask=mask, other=float("-inf")).to(tl.float32)
        bmax = tl.max(x, axis=0)
        bidx = tl.argmax(x, axis=0)
        upd = bmax > mx
        mx = tl.where(upd, bmax, mx)
        amax = tl.where(upd, v0 + bidx, amax)

    sm = 0.0
    for v0 in range(0, V, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(row + offs, mask=mask, other=float("-inf")).to(tl.float32)
        sm += tl.sum(tl.exp(x - mx), axis=0)

    lse = mx + tl.log(sm)
    tgt = tl.load(TGT + r)
    valid = tgt != -100
    tgt_safe = tl.where(valid, tgt, 0)
    tlog = tl.load(row + tgt_safe).to(tl.float32)
    nll = tl.where(valid, lse - tlog, 0.0)

    for v0 in range(0, V, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(row + offs, mask=mask, other=0.0)
        tl.store(LOGF + r * V + offs, x, mask=mask)

    ent = 0.0
    for v0 in range(0, V, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(row + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.where(mask, tl.exp(x - lse), 0.0)
        ent += tl.sum(p * (lse - x), axis=0)

    tl.store(LSE + r, lse)
    tl.store(AMAX + r, amax)
    tl.store(NLL + r, nll)
    tl.store(ENT + r, ent)


@triton.jit
def _ck_dlogits_kernel(LOGITS, TGT, LSE, DL, nrows, inv_den, V,
                       BM: tl.constexpr, BN: tl.constexpr):
    rb = tl.program_id(0)
    nt = tl.program_id(1)
    rows = rb * BM + tl.arange(0, BM)
    rmask = rows < nrows
    vw = nt * BN + tl.arange(0, BN)
    lg = tl.load(LOGITS + rows[:, None] * V + vw[None, :],
                 mask=rmask[:, None], other=0.0).to(tl.float32)
    lse = tl.load(LSE + rows, mask=rmask, other=0.0)
    p = tl.exp(lg - lse[:, None])
    tgt = tl.load(TGT + rows, mask=rmask, other=-100)
    labelled = (tgt != -100) & rmask
    onehot = (vw[None, :] == tgt[:, None]).to(tl.float32)
    d = tl.where(labelled[:, None], (p - onehot) * inv_den, 0.0)
    tl.store(DL + rows[:, None] * V + vw[None, :],
             d.to(tl.float16), mask=rmask[:, None])


def _w16_for(weight):
    """Cached fp16 copy of the frozen fp32 lm_head weight (our cast kernel)."""
    key = (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype)
    if _W16["key"] == key:
        return _W16["tensor"]
    numel = weight.numel()
    w16 = torch.empty(numel, dtype=torch.float16, device=weight.device)
    step = 8192
    _ck_cast_kernel[((numel + step - 1) // step,)](
        weight.reshape(-1), w16, numel, BLOCK=step, num_warps=8
    )
    w16 = w16.view_as(weight)
    _W16["key"] = key
    _W16["tensor"] = w16
    return w16


class CKHeadStep:
    """What the rewind recorder consumes for one micro-batch (#1693).

    ``token_loss`` is (B, T) float32: position t holds the cross-entropy of
    ``labels[:, t + 1]`` and 0.0 where that label is -100 (or there is none).
    """

    def __init__(self, token_loss):
        self.token_loss = token_loss


class CKHeadHook:
    """Holds per-step values and forwards metrics/logits-free state to TRL."""

    def __init__(self):
        self.trainer = None
        self.step = None

    def take_step(self):
        """The chunked step of the last labelled forward, once (Soup's protocol)."""
        step, self.step = self.step, None
        return step

    def store_step(self, trainer, chunked):
        try:
            from soup_cli.utils.stream_loss_chunk import STEP_ATTR
            setattr(trainer, STEP_ATTR, chunked)
        except Exception:
            pass

    def on_backward_done(self, trainer, entropy, accuracy):
        if trainer is None:
            return
        m = getattr(trainer, "_metrics", None)
        if m is None or "train" not in m:
            return
        m["train"]["entropy"].append(entropy)
        m["train"]["mean_token_accuracy"].append(accuracy)


def _engage_head_ok(weight, V, K):
    return bool(
        weight.is_cuda
        and weight.dtype == torch.float32
        and weight.is_contiguous()
        and not weight.requires_grad
        and V % _BN == 0
        and K % _BK == 0
    )


def ck_chunked_lm_loss(hidden, labels, weight, attention_mask, num_items, hook):
    """Returns (loss fp32 tensor with grad, CKHeadStep) or None to fall back."""
    B, S, K = hidden.shape
    V = weight.shape[0]
    if not _engage_head_ok(weight, V, K):
        return None
    if not hidden.is_contiguous():
        hidden = hidden.contiguous()
    npos = B * S
    X = hidden.view(npos, K)
    if X.dtype != torch.float16:
        # the kbit preparation upcasts the body to fp32; autocast feeds the
        # head matmul an fp16 copy, and so do we (gradient flows via the cast)
        X = X.to(torch.float16)
    labels_flat = labels.reshape(-1)
    tgt = torch.empty(npos, dtype=torch.int32, device=hidden.device)
    tgt[:-1] = labels_flat[1:]
    tgt.view(B, S)[:, -1] = -100
    if attention_mask is not None and attention_mask.dim() == 2:
        am = attention_mask.reshape(-1).to(torch.float32)
    else:
        am = torch.ones(npos, dtype=torch.float32, device=hidden.device)
    w16 = _w16_for(weight)
    if w16 is None:
        return None
    if num_items is None:
        den_val = float((tgt != -100).sum().item())
    else:
        den_val = float(num_items)
    if den_val <= 0.0:
        return None

    class _CKHeadFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, tgt_in, am_in, w16_in):
            # The only large temporary: the chunked fp16 logits.  A 256-row
            # piece is built by cuBLAS, streamed by our LSE kernel into LOGF,
            # then freed before the next piece.
            logf = torch.empty(npos, V, dtype=torch.float16, device=x.device)
            lse = torch.empty(npos, dtype=torch.float32, device=x.device)
            amax = torch.empty(npos, dtype=torch.int32, device=x.device)
            nll = torch.empty(npos, dtype=torch.float32, device=x.device)
            ent = torch.empty(npos, dtype=torch.float32, device=x.device)
            wt = w16_in.t()
            for start in range(0, npos, _CH):
                end = min(start + _CH, npos)
                nrows = end - start
                logits = torch.mm(x[start:end], wt)
                _ck_lse_store_kernel[(nrows,)](
                    logits, tgt_in[start:end], lse[start:end],
                    amax[start:end], nll[start:end], ent[start:end],
                    logf[start:end], V, BLOCK=_BLOCK,
                    num_warps=2, num_stages=1,
                )
                del logits
            loss = nll.sum() / den_val
            ctx.save_for_backward(tgt_in, am_in, w16_in, lse, amax, logf, ent)
            ctx.K, ctx.V, ctx.den = K, V, den_val
            step = CKHeadStep(nll.view(B, S).detach())
            hook.step = step
            if hook.trainer is not None:
                hook.store_step(hook.trainer, step)
            return loss

        @staticmethod
        def backward(ctx, grad_out):
            tgt_b, am_b, w16_b, lse_b, amax_b, logf_b, ent_b = ctx.saved_tensors
            dH = torch.empty(npos, ctx.K, dtype=torch.float16, device=tgt_b.device)
            # AMP/Accelerate may scale the scalar loss before calling us.  A
            # custom autograd Function must propagate that upstream gradient;
            # scaling inv_den keeps the fp16 dlogits far below the fp16 range
            # even for GradScaler's 2^16, unlike scaling dH after the fact.
            scale = float(grad_out) if grad_out is not None else 1.0
            inv_den = scale / ctx.den
            for a in range(0, npos, _CH):
                nrows = min(_CH, npos - a)
                dl = torch.empty(nrows, ctx.V, dtype=torch.float16,
                                 device=tgt_b.device)
                _ck_dlogits_kernel[((nrows + _BM - 1) // _BM, ctx.V // _BN)](
                    logf_b[a:a + nrows], tgt_b[a:a + nrows], lse_b[a:a + nrows],
                    dl, nrows, inv_den, ctx.V,
                    BM=_BM, BN=_BN, num_warps=4, num_stages=2,
                )
                dH[a:a + nrows] = torch.mm(dl, w16_b)
                del dl
            am_sum = float(am_b.sum().item())
            ent_mean = float((ent_b * am_b).sum() / max(am_sum, 1.0))
            lab = tgt_b != -100
            total = int(lab.sum().item())
            acc = float(((amax_b == tgt_b) & lab).sum().item()) / total \
                if total > 0 else 0.0
            hook.on_backward_done(hook.trainer, ent_mean, acc)
            return dH, None, None, None

    loss = _CKHeadFn.apply(X, tgt, am, w16)
    if hook.step is None:
        return None
    return loss, hook.step
