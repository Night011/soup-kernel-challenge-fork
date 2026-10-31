"""K3 — our own chunked fused lm_head + CE + entropy + accuracy (Triton).

The reference path materializes a full fp16 logits tensor (4 x 512 x 151936)
plus fp32 copies for log_softmax / cross-entropy / entropy — a few GB of
transient buffers per step. We never materialize full logits.

Per forward step (B*S' rows, vocab V, hidden K):

1. ``_ck_cast_kernel``       — frozen fp32 lm_head weight -> cached fp16 copy
                               (values are bit-identical to the autocast-cast
                               weight the fp16 reference matmul uses).
2. ``_ck_lse_fwd_kernel``    — tiled Triton GEMM X(rows,K) @ W16(K,V): rows are
                               processed in 256-row pieces, vocab in 128-wide
                               tiles; each program keeps a 128-row block and
                               stores pmax/psum/pidx partials.
3. ``_ck_lse_reduce_kernel`` — merges partials into per-row logsumexp and a
                               first-wins argmax.
4. ``_ck_nll_kernel``        — logit_y = sum_k X[r,k] * W16[y_r,k] gathered dot
                               (exact fp16 products, fp32 sums, the same value
                               tensor cores produce) -> nll = logit_y - lse.
5. ``_ck_dlogits_kernel``    — turns a temporary cuBLAS logits piece into its
                               softmax gradient and entropy, without creating
                               a full-batch logits tensor.
Both large GEMMs use cuBLAS on a 256-row piece; only the softmax/metrics work
is our Triton kernel.  A piece is freed before the next one is built.

den is the cross-entropy denominator the Trainer already computes
(num_items_in_batch), so loss value and every gradient scale match the
reference exactly. Per-token losses are exposed through a CKHeadStep so the
rewind recorder can produce row losses without logits.
"""

import torch
import triton
import triton.language as tl

# Геометрия куска: _CH строк — это _CH // _BM блоков по _BM строк;
# словарь режется на NT плиток по _BN столбцов; K — на плитки по _BK.
# Имена короткие, потому что они уходят в constexpr-параметры Triton.
_BM = 64    # GEMM row block
_BN = 64    # vocab tile
_BK = 64    # hidden tile
_CH = 256   # rows per piece (two _BM row blocks)

_W16 = {"key": None, "tensor": None}


@triton.jit
def _ck_cast_kernel(SRC, DST, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK
    o = i + tl.arange(0, BLOCK)
    m = o < n
    tmp = tl.load(SRC + o, mask=m, other=0.0)
    tl.store(DST + o, tmp.to(tl.float16), mask=m)


@triton.jit
def _ck_lse_fwd_kernel(X, W16, STARTS, PMAX, PSUM, PIDX, K, NT,
                       MB: tl.constexpr, BM: tl.constexpr,
                       BN: tl.constexpr, BK: tl.constexpr):
    pid0 = tl.program_id(0)      # piece
    nt = tl.program_id(1)        # vocab tile
    pw = tl.load(STARTS + pid0).to(tl.int32)
    pe = tl.load(STARTS + pid0 + 1).to(tl.int32)
    vw = nt * BN + tl.arange(0, BN)
    for mb in tl.static_range(MB):
        rows = pw + mb * BM + tl.arange(0, BM)
        rmask = rows < pe
        oe = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            x = tl.load(X + rows[:, None] * K + (k0 + oe)[None, :],
                        mask=rmask[:, None], other=0.0)
            w = tl.load(W16 + vw[:, None] * K + (k0 + oe)[None, :])
            acc = tl.dot(x, tl.trans(w), acc)
        lg = acc.to(tl.float16).to(tl.float32)   # same rounding as fp16 logits
        tmax = tl.max(lg, axis=1)
        tsum = tl.sum(tl.exp(lg - tmax[:, None]), axis=1)
        lam = tl.argmax(lg, axis=1).to(tl.int32) + nt * BN
        tmax = tl.where(rmask, tmax, float("-inf"))
        tsum = tl.where(rmask, tsum, 0.0)
        grow = (pid0 * MB + mb) * BM + tl.arange(0, BM)
        tl.store(PMAX + grow * NT + nt, tmax, mask=rmask)
        tl.store(PSUM + grow * NT + nt, tsum, mask=rmask)
        tl.store(PIDX + grow * NT + nt, lam, mask=rmask)


@triton.jit
def _ck_lse_reduce_kernel(PMAX, PSUM, PIDX, LSE, AMAX, NT, n_rows,
                          BM: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < n_rows
    mx = tl.full((BM,), float("-inf"), dtype=tl.float32)
    sm = tl.zeros((BM,), dtype=tl.float32)
    idx = tl.zeros((BM,), dtype=tl.int32)
    for t in range(NT):
        pm = tl.load(PMAX + rows * NT + t, mask=rmask, other=float("-inf"))
        ps = tl.load(PSUM + rows * NT + t, mask=rmask, other=0.0)
        pi = tl.load(PIDX + rows * NT + t, mask=rmask, other=0)
        new_mx = tl.maximum(mx, pm)
        idx = tl.where(pm > mx, pi, idx)
        sm = sm * tl.exp(mx - new_mx) + ps * tl.exp(pm - new_mx)
        mx = new_mx
    lse = mx + tl.log(sm)
    tl.store(LSE + rows, lse, mask=rmask)
    tl.store(AMAX + rows, idx, mask=rmask)


@triton.jit
def _ck_nll_kernel(X, W16, TGT, LSE, NLL, n_rows, K, BM: tl.constexpr,
                   BK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < n_rows
    y = tl.load(TGT + rows, mask=rmask, other=-100)
    ymask = y != -100
    y_safe = tl.where(ymask, y, 0).to(tl.int32)
    oe = tl.arange(0, BK)
    acc = tl.zeros((BM,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rows[:, None] * K + (k0 + oe)[None, :],
                    mask=rmask[:, None], other=0.0).to(tl.float32)
        w = tl.load(W16 + y_safe[:, None] * K + (k0 + oe)[None, :],
                    mask=ymask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(x * w, axis=1)
    nll = tl.where(ymask, tl.load(LSE + rows, mask=rmask, other=0.0) - acc, 0.0)
    tl.store(NLL + rows, nll, mask=rmask)


@triton.jit
def _ck_dlogits_kernel(LOGITS, TGT, AM, LSE, ENT, DL, nrows, inv_den, V,
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
    am = tl.load(AM + rows, mask=rmask, other=0.0)
    contrib = tl.sum(p * (lse[:, None] - lg) * am[:, None], axis=1)
    tl.atomic_add(ENT + rows, contrib, mask=rmask)
    onehot = (vw[None, :] == tgt[:, None]).to(tl.float32)
    d = tl.where(labelled[:, None], (p - onehot) * inv_den, 0.0)
    tl.store(DL + rows[:, None] * V + vw[None, :], d.to(tl.float16),
             mask=rmask[:, None])


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
    NT = V // _BN
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
            lse = torch.empty(npos, dtype=torch.float32, device=x.device)
            amax = torch.empty(npos, dtype=torch.int32, device=x.device)
            nll = torch.empty(npos, dtype=torch.float32, device=x.device)
            wt = w16_in.t()
            # cuBLAS is substantially faster than a handwritten large-vocab
            # GEMM on a T4.  Each temporary is only a 256-row piece, preserving
            # K3's logits-memory advantage.
            for start in range(0, npos, _CH):
                end = min(start + _CH, npos)
                logits = torch.mm(x[start:end], wt)
                logits_f = logits.float()
                piece_lse = torch.logsumexp(logits_f, dim=1)
                target = tgt_in[start:end]
                valid = target != -100
                safe_target = target.clamp(min=0, max=V - 1).to(torch.long)
                target_logits = logits_f.gather(1, safe_target[:, None]).squeeze(1)
                lse[start:end].copy_(piece_lse)
                amax[start:end].copy_(logits.argmax(dim=1))
                nll[start:end].copy_(torch.where(valid, piece_lse - target_logits,
                                                   torch.zeros_like(piece_lse)))
                del target_logits, safe_target, valid, piece_lse, logits_f, logits
            loss = nll.sum() / den_val
            ctx.save_for_backward(x, tgt_in, am_in, w16_in, lse, amax)
            ctx.NT, ctx.den, ctx.K, ctx.V = NT, den_val, K, V
            step = CKHeadStep(nll.view(B, S).detach())
            hook.step = step
            if hook.trainer is not None:
                hook.store_step(hook.trainer, step)
            return loss

        @staticmethod
        def backward(ctx, grad_out):
            x, tgt_b, am_b, w16_b, lse_b, amax_b = ctx.saved_tensors
            dH = torch.empty(npos, K, dtype=torch.float16, device=x.device)
            ent = torch.zeros(npos, dtype=torch.float32, device=x.device)
            inv_den = 1.0 / ctx.den
            wt = w16_b.t()
            for a in range(0, npos, _CH):
                nrows = min(_CH, npos - a)
                logits = torch.mm(x[a:a + nrows], wt)
                dl = torch.empty(nrows, ctx.V, dtype=torch.float16,
                                 device=x.device)
                _ck_dlogits_kernel[((nrows + _BM - 1) // _BM, ctx.NT)](
                    logits, tgt_b[a:a + nrows], am_b[a:a + nrows], lse_b[a:a + nrows],
                    ent[a:a + nrows], dl, nrows, inv_den, ctx.V,
                    BM=_BM, BN=_BN, num_warps=4, num_stages=2,
                )
                dH[a:a + nrows] = torch.mm(dl, w16_b)
                del dl, logits
            # AMP/Accelerate may scale the scalar loss before calling us.  A
            # custom autograd Function must propagate that upstream gradient;
            # otherwise the optimizer later unscales an already-unscaled dH.
            dH.mul_(grad_out.to(dtype=dH.dtype))
            am_sum = float(am_b.sum().item())
            ent_mean = float((ent * am_b).sum() / max(am_sum, 1.0))
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
