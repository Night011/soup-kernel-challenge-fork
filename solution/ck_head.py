"""K3 — compact chunked lm_head + CE (Triton loss, metrics and backward).

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
* training loss reuses those fp32 NLL values instead of constructing a second
  fp32 logits/log-softmax buffer; its gradient uses softmax math in Triton;
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
import triton
import triton.language as tl

# _CH — rows per head piece (only one fp16 logits piece is materialized).
# _BLOCK — vocab elements per streaming step in the LSE kernel.
_BN = 64
_BK = 64
_CH = 256
_BLOCK = 256

_W16 = {"key": None, "tensor": None}


def _weight_key(weight):
    return (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype)


def offload_frozen_head(lm):
    """Keep the exact fp32 master on CPU while the frozen head uses its cache.

    Called at the first labelled forward, after Accelerate has placed the
    model. A shared embedding is supported only when its fp16 cache is exact.
    """
    weight = lm.lm_head.weight
    if (getattr(lm, "_ck_head_offload_rejected", False)
            or not weight.is_cuda or weight.dtype != torch.float32
            or weight.requires_grad or not weight.is_contiguous()
            or weight.shape[0] % _BN or weight.shape[1] % _BK):
        return
    embedding = lm.get_input_embeddings()
    shared = embedding.weight is weight or embedding.weight.data_ptr() == weight.data_ptr()
    compute_weight = _w16_for(weight)
    if shared:
        # k-bit preparation commonly upcasts an fp16 embedding to fp32.
        # Only offload a shared master if the cached representation is exact:
        # the lookup must reproduce the original fp32 embedding values.
        if (embedding.weight.requires_grad or embedding.max_norm is not None
                or not torch.equal(compute_weight.float(), weight)):
            lm._ck_head_offload_rejected = True
            return
        if not hasattr(embedding, "_ck_orig_embedding_forward"):
            import types
            embedding._ck_orig_embedding_forward = embedding.forward

            def embedding_forward(module, input):
                if not getattr(lm, "_ck_head_offloaded", False):
                    return module._ck_orig_embedding_forward(input)
                ids = input.contiguous()
                cached = lm._ck_head_compute_weight
                out = torch.empty((*ids.shape, cached.shape[1]), device=ids.device,
                                  dtype=torch.float32)
                _ck_embedding_kernel[(triton.cdiv(out.numel(), 1024),)](
                    ids, cached, out, out.numel(), cached.shape[1],
                    cached.shape[0], BLOCK=1024, num_warps=4,
                )
                return out

            embedding.forward = types.MethodType(embedding_forward, embedding)
    lm._ck_head_compute_weight = compute_weight
    lm._ck_head_gpu_device = weight.device
    weight.data = weight.detach().cpu()
    _W16["key"] = _weight_key(weight)
    lm._ck_head_offloaded = True
    print("[ck] frozen fp32 head offloaded to CPU; fp16 compute copy on GPU", flush=True)


@triton.jit
def _ck_embedding_kernel(IDS, WEIGHT, OUT, N, H: tl.constexpr, V,
                          BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(IDS + i // H, i < N, other=0)
    values = tl.load(WEIGHT + token * H + i % H,
                     (i < N) & (token >= 0) & (token < V), other=0)
    tl.store(OUT + i, values.to(tl.float32), i < N)


def restore_frozen_head(lm):
    """Restore the original parameter, with no rounding, for stock forwards."""
    if not getattr(lm, "_ck_head_offloaded", False):
        return
    weight = lm.lm_head.weight
    old_key = _weight_key(weight)
    weight.data = weight.data.to(lm._ck_head_gpu_device)
    if _W16["key"] == old_key:
        _W16["key"] = _weight_key(weight)
    lm._ck_head_offloaded = False


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
def _ck_ce_grad_kernel(LOGITS, LSE, TARGET, DLOGITS, V, DEN, SCALE,
                       BLOCK: tl.constexpr):
    """Write the scaled fp16 CE gradient for a tile of labelled logits."""
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < V
    x = tl.load(LOGITS + row * V + col, mask=mask, other=0.0).to(tl.float32)
    lse = tl.load(LSE + row)
    target = tl.load(TARGET + row)
    inv_den = tl.load(SCALE) / DEN
    grad = (tl.exp(x - lse) - (col == target).to(tl.float32)) * inv_den
    tl.store(DLOGITS + row * V + col, grad.to(tl.float16), mask=mask)


@triton.jit
def _ck_scatter_grad_kernel(SRC, ROWS, GRAD_OUT, SCALE, DST, N,
                             H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(ROWS + i // H, i < N, other=0)
    value = tl.load(SRC + i, i < N, other=0).to(tl.float32)
    factor = tl.load(GRAD_OUT).to(tl.float32) / tl.load(SCALE)
    tl.store(DST + row * H + i % H, value * factor, i < N)


def _w16_for(weight):
    """Cached fp16 copy of the frozen fp32 lm_head weight (our cast kernel)."""
    key = _weight_key(weight)
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
        (weight.is_cuda or (weight.device.type == "cpu"
                           and _W16["key"] == _weight_key(weight)))
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
        tgt_keep = tgt.index_select(0, idx_keep)
        lab_keep = tgt_keep != -100
        lab_pos = lab_keep.nonzero(as_tuple=False).squeeze(1)
        n_lab = lab_pos.numel()
        if n_lab == 0:
            return None
        # Put labelled rows first inside the head only. This keeps every
        # gradient GEMM compact without per-chunk nonzero synchronizations.
        order = torch.argsort(lab_keep.to(torch.uint8), descending=True, stable=True)
        idx_keep = idx_keep.index_select(0, order)
        x_keep = X.index_select(0, idx_keep)
        tgt_keep = tgt.index_select(0, idx_keep)
        am_keep = am.index_select(0, idx_keep)
        lab_keep = tgt_keep != -100
        idx_lab = idx_keep[:n_lab]

    w16 = _w16_for(weight)
    if w16 is None:
        return None
    if num_items is None:
        den_val = float(n_lab)
    else:
        den_val = float(num_items)
    if den_val <= 0.0:
        return None

    # The scalar CE upstream is known up to AMP scaling. Compute its local
    # derivative while the forward logits are live; carry the current scaler
    # through the GEMM to preserve small fp16 probability gradients. Backward
    # still propagates the actual upstream gradient, including other scales.
    scaler = getattr(getattr(hook.trainer, "accelerator", None), "scaler", None)
    if scaler is not None and scaler.is_enabled():
        amp_scale = getattr(scaler, "_scale", None)
        if amp_scale is None:
            amp_scale = torch.tensor(scaler.get_scale(), device=X.device)
        else:
            amp_scale = amp_scale.detach()
    else:
        amp_scale = torch.ones((), dtype=torch.float32, device=X.device)

    class _CKHeadFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, tgt_in, am_in, w16_in):
            # The same fp32 NLL values feed both loss and rewind. Ignored
            # targets have zero NLL, matching cross_entropy(ignore_index=-100).
            lse = torch.empty(n_keep, dtype=torch.float32, device=x.device)
            amax = torch.empty(n_keep, dtype=torch.int32, device=x.device)
            nll = torch.empty(n_keep, dtype=torch.float32, device=x.device)
            ent = torch.empty(n_keep, dtype=torch.float32, device=x.device)
            dH_lab = torch.empty(n_lab, K, dtype=torch.float16, device=x.device)
            wt = w16_in.t()
            for start in range(0, n_keep, _CH):
                end = min(start + _CH, n_keep)
                nrows = end - start
                # Tensor cores accumulate fp16 products in fp32. Keep that
                # output precision through the loss instead of rounding the
                # logits to fp16 before softmax.
                with torch.autocast("cuda", enabled=False):
                    logits = torch.mm(x_keep[start:end], wt, out_dtype=torch.float32)
                _ck_lse_kernel[(nrows,)](
                    logits, tgt_keep[start:end], lse[start:end],
                    amax[start:end], nll[start:end], ent[start:end],
                    V, BLOCK=_BLOCK, num_warps=2, num_stages=1,
                )
                labelled = max(0, min(end, n_lab) - start)
                if labelled:
                    dlogits = torch.empty((labelled, V), dtype=torch.float16,
                                          device=x.device)
                    _ck_ce_grad_kernel[(labelled, triton.cdiv(V, 1024))](
                        logits, lse[start:end], tgt_keep[start:end], dlogits,
                        V, den_val, amp_scale, BLOCK=1024, num_warps=4,
                    )
                    dH_lab[start:start + labelled] = torch.mm(dlogits, w16_in)
                    del dlogits
                del logits
            loss = nll.sum() / den_val
            token_loss = torch.zeros((B, S), dtype=torch.float32,
                                     device=x.device)
            token_loss.view(-1)[idx_keep] = nll
            ctx.save_for_backward(dH_lab, idx_lab, amp_scale)
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
            dH_lab, idx_lab_b, forward_scale = ctx.saved_tensors
            dH = torch.zeros(ctx.npos, ctx.K, dtype=torch.float16,
                             device=dH_lab.device)
            _ck_scatter_grad_kernel[(triton.cdiv(dH_lab.numel(), 1024),)](
                dH_lab, idx_lab_b, grad_out, forward_scale, dH, dH_lab.numel(),
                H=ctx.K, BLOCK=1024, num_warps=4,
            )
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
