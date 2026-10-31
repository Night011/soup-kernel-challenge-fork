"""K3 — compact chunked lm_head + CE (Triton LSE for metrics, PyTorch CE).

The reference path materializes a full fp16 logits tensor (4 x 512 x 151936)
plus fp32 copies for log_softmax / cross-entropy / entropy — a few GB of
transient buffers per step.  We first drop every all-padding row (the K4 trim
already removes all-padding columns) and then work only on the rows needed by
the mask or by the shifted CE target:

* the forward logits are built only for the rows needed by either the
  attention mask (entropy metric) or the shifted CE target (loss).  A
  leading-pad position can have a real shifted target, so the union is used,
  not the attention mask alone;
* our Triton ``_ck_lse_kernel`` produces per-row logsumexp, NLL, entropy and
  argmax for those rows — the metric/token-loss path stays ours;
* the training loss uses stock fp32 cross-entropy; its gradient uses the same
  softmax math in Triton on the labelled rows only;
* the backward recomputes the fp16 logits for the labelled rows and writes the
  fp16 softmax gradient ``(p - onehot) / den`` in 128-row pieces, then
  ``dH = dlogits_piece @ W16`` stays cuBLAS and is scattered back into the
  full hidden gradient (zero at unlabelled/padded rows, exactly like the
  reference's ignored positions).

Why this is correct: the reference's cross-entropy is driven by the shifted
labels, not by the attention mask.  A right-pad row has -100 there and a zero
mask, so its logits, softmax gradient, token loss and entropy weight are all
zero; a leading-pad row can still have a real shifted target (the first real
token), so it is kept for the loss even though its mask weight is zero.  The
union removes only values that the task, metrics or rewind consume as zero.

The large GEMMs stay cuBLAS: a handwritten large-vocab Triton GEMM is much
slower on a T4 and its compile time alone breaks the wall clock.  No full
logits buffer is stored; the backward recomputes one 128-row piece at a time.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# _CH — rows per head piece (a 128-row logits piece is the only large temp).
# _BLOCK — vocab elements per streaming step in the LSE kernel.
_BN = 64
_BK = 64
_CH = 128
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
def _ck_lse_kernel(LOGITS, TGT, LSE, AMAX, NLL, ENT, V,
                   BLOCK: tl.constexpr):
    """Per-row logsumexp + argmax + NLL + entropy.

    Two streaming passes over one row compute a numerically stable online
    logsumexp/argmax followed by entropy. Only rows kept by the mask/target
    union reach it; the caller applies the attention-mask weight when averaging
    entropy.
    """
    r = tl.program_id(0)
    row = LOGITS + r * V

    mx = float("-inf")
    sm = 0.0
    amax = 0
    for v0 in range(0, V, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(row + offs, mask=mask, other=float("-inf")).to(tl.float32)
        bmax = tl.max(x, axis=0)
        bidx = tl.argmax(x, axis=0)
        new_mx = tl.maximum(mx, bmax)
        sm = sm * tl.exp(mx - new_mx) + tl.sum(tl.exp(x - new_mx), axis=0)
        amax = tl.where(bmax > mx, v0 + bidx, amax)
        mx = new_mx

    lse = mx + tl.log(sm)
    tgt = tl.load(TGT + r)
    valid = tgt != -100
    tgt_safe = tl.where(valid, tgt, 0)
    tlog = tl.load(row + tgt_safe).to(tl.float32)
    nll = tl.where(valid, lse - tlog, 0.0)

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
def _ck_ce_lse_kernel(LOGITS, LSE, V, BLOCK: tl.constexpr):
    """Normalise the recomputed logits used by the loss backward pass."""
    row = tl.program_id(0)
    mx = float("-inf")
    sm = 0.0
    for start in range(0, V, BLOCK):
        col = start + tl.arange(0, BLOCK)
        x = tl.load(LOGITS + row * V + col, mask=col < V,
                    other=float("-inf")).to(tl.float32)
        new_mx = tl.maximum(mx, tl.max(x, axis=0))
        sm = sm * tl.exp(mx - new_mx) + tl.sum(tl.exp(x - new_mx), axis=0)
        mx = new_mx
    tl.store(LSE + row, mx + tl.log(sm))


@triton.jit
def _ck_ce_grad_kernel(LOGITS, LSE, TARGET, DLOGITS, V, INV_DEN,
                       BLOCK: tl.constexpr):
    """Write the scaled fp16 CE gradient for a tile of labelled logits."""
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < V
    x = tl.load(LOGITS + row * V + col, mask=mask, other=0.0).to(tl.float32)
    lse = tl.load(LSE + row)
    target = tl.load(TARGET + row)
    grad = (tl.exp(x - lse) - (col == target).to(tl.float32)) * INV_DEN
    tl.store(DLOGITS + row * V + col, grad.to(tl.float16), mask=mask)


@triton.jit
def _ck_select_logits_kernel(LOGITS, ROWS, OUT, V, BLOCK: tl.constexpr):
    """Gather labelled logits and convert them to fp32 for stock CE."""
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source_row = tl.load(ROWS + row)
    values = tl.load(LOGITS + source_row * V + col, mask=col < V, other=0.0)
    tl.store(OUT + row * V + col, values.to(tl.float32), mask=col < V)


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

    # Keep rows needed by either the entropy metric (attention mask) or the
    # shifted CE target.  A leading-pad position can have a real shifted target
    # (the first real token), so the attention mask alone is not enough: the
    # reference CE is driven by shift_labels, not by the mask.
    # These are plain index/copy operations, not part of the autograd graph:
    # the custom Function below owns the only gradient path (dH -> X).
    with torch.no_grad():
        keep = (am != 0) | (tgt != -100)
        idx_keep = keep.nonzero(as_tuple=False).squeeze(1)
        n_keep = idx_keep.numel()
        if n_keep == 0:
            return None
        x_keep = X.index_select(0, idx_keep)
        tgt_keep = tgt.index_select(0, idx_keep)
        am_keep = am.index_select(0, idx_keep)
        lab_keep = tgt_keep != -100
        n_lab = int(lab_keep.sum().item())
        if n_lab == 0:
            return None
        x_lab = x_keep[lab_keep]
        tgt_lab = tgt_keep[lab_keep].to(torch.long)
        idx_lab = idx_keep[lab_keep]

    w16 = _w16_for(weight)
    if w16 is None:
        return None
    if num_items is None:
        den_val = float(n_lab)
    else:
        den_val = float(num_items)
    if den_val <= 0.0:
        return None

    class _CKHeadFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, tgt_in, am_in, w16_in):
            # One 128-row fp16 logits piece at a time; no full logits buffer.
            # The LSE pass runs over every kept row (mask or real shifted
            # target); the CE runs over the labelled rows of the same pieces.
            lse = torch.empty(n_keep, dtype=torch.float32, device=x.device)
            amax = torch.empty(n_keep, dtype=torch.int32, device=x.device)
            nll = torch.empty(n_keep, dtype=torch.float32, device=x.device)
            ent = torch.empty(n_keep, dtype=torch.float32, device=x.device)
            loss_sum = torch.zeros((), dtype=torch.float32, device=x.device)
            wt = w16_in.t()
            for start in range(0, n_keep, _CH):
                end = min(start + _CH, n_keep)
                nrows = end - start
                logits = torch.mm(x_keep[start:end], wt)
                _ck_lse_kernel[(nrows,)](
                    logits, tgt_keep[start:end], lse[start:end],
                    amax[start:end], nll[start:end], ent[start:end],
                    V, BLOCK=_BLOCK, num_warps=2, num_stages=1,
                )
                lab_chunk = lab_keep[start:end]
                lab_rows = lab_chunk.nonzero(as_tuple=False).squeeze(1)
                n_lab_chunk = lab_rows.numel()
                if n_lab_chunk:
                    logits_lab = torch.empty((n_lab_chunk, V), dtype=torch.float32,
                                             device=x.device)
                    _ck_select_logits_kernel[(n_lab_chunk, triton.cdiv(V, 1024))](
                        logits, lab_rows, logits_lab, V, BLOCK=1024, num_warps=4,
                    )
                    tgt_chunk = tgt_keep[start:end].index_select(0, lab_rows).to(torch.long)
                    loss_sum += F.cross_entropy(
                        logits_lab, tgt_chunk, reduction="sum",
                    )
                del logits
            loss = loss_sum / den_val
            token_loss = torch.zeros((B, S), dtype=torch.float32,
                                     device=x.device)
            token_loss.view(-1)[idx_keep] = nll
            ctx.save_for_backward(x_lab, tgt_lab, w16_in, idx_lab)
            ctx.K, ctx.V, ctx.den = K, V, den_val
            ctx.npos, ctx.n_lab = npos, n_lab
            ctx.ent_keep, ctx.amax_keep = ent, amax
            ctx.am_keep = am_keep
            ctx.tgt_keep, ctx.lab_keep = tgt_keep, lab_keep
            step = CKHeadStep(token_loss.detach())
            hook.step = step
            if hook.trainer is not None:
                hook.store_step(hook.trainer, step)
            return loss

        @staticmethod
        def backward(ctx, grad_out):
            x_lab_b, tgt_lab_b, w16_b, idx_lab_b = ctx.saved_tensors
            n_lab_b, npos_b, K_b, V_b = ctx.n_lab, ctx.npos, ctx.K, ctx.V
            dH_lab = torch.empty(n_lab_b, K_b, dtype=torch.float16,
                                 device=x_lab_b.device)
            # AMP/Accelerate may scale the scalar loss before calling us.  A
            # custom autograd Function must propagate that upstream gradient;
            # scaling inv_den keeps the fp16 dlogits far below the fp16 range
            # even for GradScaler's 2^16, unlike scaling dH after the fact.
            scale = float(grad_out) if grad_out is not None else 1.0
            inv_den = scale / ctx.den
            wt_b = w16_b.t()
            for a in range(0, n_lab_b, _CH):
                nrows = min(_CH, n_lab_b - a)
                logits = torch.mm(x_lab_b[a:a + nrows], wt_b)
                lse_b = torch.empty(nrows, dtype=torch.float32, device=x_lab_b.device)
                dlogits = torch.empty_like(logits)
                _ck_ce_lse_kernel[(nrows,)](
                    logits, lse_b, V_b, BLOCK=1024, num_warps=4,
                )
                _ck_ce_grad_kernel[(nrows, triton.cdiv(V_b, 1024))](
                    logits, lse_b, tgt_lab_b[a:a + nrows], dlogits, V_b, inv_den,
                    BLOCK=1024, num_warps=4,
                )
                dH_lab[a:a + nrows] = torch.mm(dlogits, w16_b)
                del logits, lse_b, dlogits
            dH = torch.zeros(npos_b, K_b, dtype=torch.float16,
                             device=x_lab_b.device)
            dH.index_copy_(0, idx_lab_b, dH_lab)
            am_sum = float(ctx.am_keep.sum().item())
            ent_mean = float((ctx.ent_keep * ctx.am_keep).sum().item()) / am_sum \
                if am_sum > 0 else 0.0
            lab = ctx.lab_keep
            total = int(ctx.n_lab)
            acc = float(((ctx.amax_keep == ctx.tgt_keep) & lab).sum().item()) / total \
                if total > 0 else 0.0
            hook.on_backward_done(hook.trainer, ent_mean, acc)
            return dH, None, None, None

    loss = _CKHeadFn.apply(X, tgt, am, w16)
    if hook.step is None:
        return None
    return loss, hook.step
