"""Correctness of our kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("shape", [(65, 128), (1536, 1536)])
def test_nf4_decode_matches_bitsandbytes(nested, shape):
    import bitsandbytes.functional as functional
    from solution import ck_nf4

    torch.manual_seed(9)
    weight = torch.randn(*shape, device="cuda", dtype=torch.float16)
    packed, state = functional.quantize_4bit(weight, blocksize=64,
                                            compress_statistics=nested, quant_type="nf4")
    reference_fn = ck_nf4._ORIGINAL_DEQUANT or functional.dequantize_4bit
    reference = reference_fn(packed, state)
    out = ck_nf4.decode_nf4(packed, state)
    assert out is not None
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    transposed = ck_nf4.decode_nf4(packed.t(), state)
    torch.testing.assert_close(transposed, reference.t(), rtol=0, atol=0)


@pytest.mark.parametrize("base_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scale", [2.0, 1.7])
def test_lora_add_matches_pytorch(base_dtype, scale):
    """Preserve both rounding points in PEFT's scale/add and its gradient."""
    from solution.ck_lora import CKLoRAAddFn

    torch.manual_seed(4)
    base = torch.randn(3, 17, 1536, device="cuda", dtype=base_dtype)
    update = torch.randn_like(base, dtype=torch.float16)
    grad = torch.randn_like(base)
    br, ur = base.clone().requires_grad_(), update.clone().requires_grad_()
    ref = br + ur * scale
    ref.backward(grad)
    bo, uo = base.clone().requires_grad_(), update.clone().requires_grad_()
    out = CKLoRAAddFn.apply(bo, uo, scale)
    out.backward(grad)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(bo.grad, br.grad, rtol=0, atol=0)
    torch.testing.assert_close(uo.grad, ur.grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("vocab", [512, 151936])
def test_streamed_head_statistics(dtype, vocab):
    """Online entropy/LSE with a masked vocabulary tail and offset logits."""
    import solution.ck_head as head

    torch.manual_seed(10)
    logits = torch.randn(4, vocab, device="cuda", dtype=dtype) * 2
    logits[2] += 1000
    logits[3].fill_(4)
    target = torch.tensor([0, vocab - 1, -100, 1], device="cuda", dtype=torch.int32)
    lse, nll, entropy = [torch.empty(4, device="cuda") for _ in range(3)]
    amax = torch.empty(4, device="cuda", dtype=torch.int32)
    head._ck_lse_kernel[(4,)](logits, target, lse, amax, nll, entropy,
                              vocab, BLOCK=head._BLOCK, num_warps=4)
    z = logits.float()
    expected_lse = z.logsumexp(-1)
    log_probs = z.log_softmax(-1)
    expected_entropy = -(log_probs.exp() * log_probs).sum(-1)
    expected_nll = -log_probs[torch.arange(4, device="cuda"), target.clamp_min(0).long()]
    expected_nll[target == -100] = 0
    torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=2e-4)
    torch.testing.assert_close(nll, expected_nll, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(entropy, expected_entropy, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(amax, z.argmax(-1).int(), rtol=0, atol=0)


def test_swiglu_matches_pytorch():
    """The starter-derived tiled SwiGLU vs the fp32 PyTorch formula."""
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(4, 512, 8960, device="cuda", dtype=torch.float16,
                       requires_grad=True)
    up = torch.randn(4, 512, 8960, device="cuda", dtype=torch.float16,
                     requires_grad=True)
    grad = torch.randn(4, 512, 8960, device="cuda", dtype=torch.float16)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    dgate, dup = gate.grad.clone(), up.grad.clone()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    torch.testing.assert_close(out.float(), ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(gate.grad.float(), dgate.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(up.grad.float(), dup.float(), rtol=2e-3, atol=2e-3)


def test_shared_frozen_head_offload():
    """Shared fp32 embedding/master survives offload and stock restoration."""
    from solution import ck_head

    torch.manual_seed(6)
    class SharedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(512, 64, device="cuda")
            self.embed.weight.data = self.embed.weight.data.half().float()
            self.embed.weight.requires_grad_(False)
            self.lm_head = torch.nn.Linear(64, 512, bias=False, device="cuda")
            self.lm_head.weight = self.embed.weight

        def get_input_embeddings(self):
            return self.embed

    model = SharedModel()
    ids = torch.tensor([[0, 1, 511], [16, 70, 82]], device="cuda")
    original = model.embed.weight.detach().clone()
    reference = model.embed(ids)
    ck_head.offload_frozen_head(model)
    try:
        assert model.embed.weight.device.type == "cpu"
        assert model.embed.weight is model.lm_head.weight
        out = model.embed(ids)
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        assert out.dtype == torch.float32
    finally:
        ck_head.restore_frozen_head(model)
    torch.testing.assert_close(model.embed.weight, original, rtol=0, atol=0)
    torch.testing.assert_close(model.embed(ids), reference, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(2048, 1536), (3, 5, 1536)])
def test_rmsnorm_matches_reference(dtype, shape):
    """K1: fused RMSNorm forward + backward vs the reference math."""
    import solution.ck_rms as rms

    torch.manual_seed(0)
    hidden = 1536
    eps = 1e-6
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    w = torch.randn(hidden, device="cuda", dtype=dtype)
    dy = torch.randn(*shape, device="cuda", dtype=dtype)

    xr = x.clone().requires_grad_(True)
    xf = xr.to(torch.float32)
    ref = w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) \
        .to(dtype)
    ref.backward(dy.float())
    ref_dx = xr.grad.clone()
    ref_y = ref.detach().clone()

    xo = x.clone().requires_grad_(True)
    out = rms.CKRMSNormFn.apply(xo, w, eps)
    out.backward(dy)
    torch.testing.assert_close(out.float(), ref_y.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(xo.grad.float(), ref_dx.float(),
                               rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("cos_dtype", [torch.float32, torch.float16])
def test_rope_matches_reference(cos_dtype):
    """K2: fused RoPE fwd+bwd vs apply_rotary_pos_emb (fp32 and fp16 cos/sin)."""
    import solution.ck_rope as rope_mod
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    torch.manual_seed(0)
    B, Hq, Hk, S, RD = 2, 14, 2, 64, 128
    q = torch.randn(B, Hq, S, RD, device="cuda", dtype=torch.float16)
    k = torch.randn(B, Hk, S, RD, device="cuda", dtype=torch.float16)
    pos = torch.arange(S, device="cuda").float()
    inv = 1.0 / (10000 ** (torch.arange(0, RD, 2, device="cuda").float() / RD))
    freqs = pos[None, :, None] * inv[None, None, :]
    emb = torch.cat([freqs, freqs], dim=-1)          # (1, S, RD)
    cos = emb.cos().expand(B, -1, -1).to(cos_dtype)
    sin = emb.sin().expand(B, -1, -1).to(cos_dtype)
    dq = torch.randn_like(q)
    dk = torch.randn_like(k)

    qr = q.clone().requires_grad_(True)
    kr = k.clone().requires_grad_(True)
    rq, rk = apply_rotary_pos_emb(qr, kr, cos, sin)
    gq, gk = torch.autograd.grad(
        (rq * dq).float().sum() + (rk * dk).float().sum(), (qr, kr))

    # our path also has to handle the transposed non-contiguous q/k the Qwen2
    # attention really passes (view -> transpose layout).  Qwen2 uses GQA
    # (q has 14 heads, k has 2), so production deliberately takes the stock
    # RoPE fallback; exercise the autograd primitive directly here.
    qt = q.transpose(1, 2).contiguous().transpose(1, 2)
    kt = k.transpose(1, 2).contiguous().transpose(1, 2)
    qo = qt.clone().requires_grad_(True)
    ko = kt.clone().requires_grad_(True)
    oval_q, oval_k = rope_mod.CKRopeFn.apply(qo, ko, cos, sin)
    ((oval_q * dq).float().sum() + (oval_k * dk).float().sum()).backward()
    torch.testing.assert_close(oval_q.float(), rq.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(oval_k.float(), rk.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(qo.grad.float(), gq.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(ko.grad.float(), gk.float(), rtol=2e-2, atol=2e-2)


def test_rmsnorm_fp16_projection_boundary():
    """Direct fp16 output vs fp32 RMSNorm immediately autocast by a projection."""
    from solution.ck_rms import CKRMSNormFn

    torch.manual_seed(8)
    x = torch.randn(3, 17, 1536, device="cuda", dtype=torch.float32)
    w = torch.randn(1536, device="cuda", dtype=torch.float32)
    dy = torch.randn_like(x, dtype=torch.float16)
    xr = x.clone().requires_grad_()
    ref = (w * xr * torch.rsqrt(xr.square().mean(-1, keepdim=True) + 1e-6)).half()
    ref.backward(dy)
    xo = x.clone().requires_grad_()
    out = CKRMSNormFn.apply(xo, w, 1e-6, True)
    out.backward(dy)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(xo.grad, xr.grad, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("amp_scale", [None, 65536.0])
def test_head_loss_matches_reference(amp_scale):
    """K3: compact chunked head + stock fp32 CE vs full fp16 logits + CE."""
    import solution.ck_head as head

    torch.manual_seed(0)
    B, S, K, V = 2, 32, 64, 512   # V % _BN == 0, K % _BK == 0
    labelled_mid = 20
    hidden = torch.randn(B, S, K, device="cuda", dtype=torch.float16)
    weights = torch.randn(V, K, device="cuda", dtype=torch.float32)
    am = torch.ones(B, S, dtype=torch.long, device="cuda")
    # A leading pad makes the preceding row's shifted label valid while its
    # attention-mask value is zero.  Compact K3 must keep that row for CE,
    # even though it excludes it from the entropy average.
    am[0, :3] = 0
    am[:, -4:] = 0
    labels = torch.randint(0, V, (B, S), device="cuda")
    labels[am == 0] = -100
    labels[:, 0] = -100
    lab = labels[:, 1:] != -100
    num_items = int(lab.sum())

    ref_w = weights.to(torch.float16)
    assert hidden.requires_grad is False
    hr = hidden.clone().requires_grad_(True)
    logits = torch.nn.functional.linear(hr, ref_w)
    nll = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, V).float(), labels[:, 1:].reshape(-1),
        reduction="none", ignore_index=-100,
    )
    ref_loss = nll[lab.reshape(-1)].sum() / num_items
    # A non-unit upstream gradient checks that the custom backward propagates
    # loss scaling instead of silently assuming a scale of one.
    upstream = 3.0 if amp_scale is None else amp_scale
    (ref_loss * upstream).backward()
    ref_dh = hr.grad.clone()
    ref_loss_v = float(ref_loss.detach().item())

    ho = hidden.clone().requires_grad_(True)
    hook = head.CKHeadHook()
    if amp_scale is not None:
        from types import SimpleNamespace
        hook.trainer = SimpleNamespace(accelerator=SimpleNamespace(
            scaler=torch.amp.GradScaler("cuda", init_scale=amp_scale)))
    loss, step = head.ck_chunked_lm_loss(
        ho, labels, weights, am, num_items, hook)
    assert loss is not None, "head path must be engageable in this test"
    (loss * upstream).backward()

    tol = 1e-3 if ref_loss_v < 2 else 2e-3
    assert abs(float(loss.detach().item()) - ref_loss_v) <= 3e-3 * max(1, ref_loss_v)
    # An absolute error bound must have the same units with and without AMP.
    # Compare d(loss)/d(hidden), not d(65536*loss)/d(hidden). The 2e-3 bound
    # is stricter than the previous unit-scale bound (2e-2 / 3).
    assert torch.isfinite(ho.grad).all()
    torch.testing.assert_close(ho.grad.float() / upstream, ref_dh.float() / upstream,
                               rtol=2e-2, atol=2e-3)
    assert step is not None
    tl = step.token_loss
    assert tl.shape == (B, S)
    ref_tokens = nll.view(B, S - 1).detach()
    torch.testing.assert_close(tl[:, :-1].float(), ref_tokens.float(),
                               rtol=2e-2, atol=2e-2)


def test_prepare_inputs_trim():
    """K4: only leading/trailing all-pad columns leave, ids keep base positions."""
    import solution.integrate as integ

    B, N = 2, 8
    ids = torch.arange(N, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
    am = torch.zeros(B, N, dtype=torch.long, device="cuda")
    am[0, 1:5] = 1
    am[1, 0:4] = 1
    labels = torch.full((B, N), -100, dtype=torch.long, device="cuda")
    used = am.bool()
    labels[used] = torch.randint(0, 100, (int(used.sum()),), device="cuda")
    inputs = {"input_ids": ids, "attention_mask": am, "labels": labels,
              "position_ids": torch.arange(N, device="cuda").unsqueeze(0).expand(B, -1).contiguous()}
    out = integ._trim_inputs(inputs)
    assert out["input_ids"].shape == (B, 5)
    assert torch.equal(out["position_ids"][0],
                       torch.arange(0, 5, device="cuda"))
    assert torch.equal(out["labels"], labels[:, 0:5])
    assert torch.equal(out["attention_mask"], am[:, 0:5])


def test_smoke_full_qwen2_flow():
    """End-to-end: tiny Qwen2 in the Soup-T4 layout (fp32 body from
    `prepare_model_for_kbit_training`, fp16 autocast compute, fp32 frozen
    lm_head), our patches installed, forward+backward through the chunked head,
    all grads sane. Catches integration breaks before a full `soup train` run."""
    import solution
    from solution import integrate
    from transformers import AutoConfig, Qwen2ForCausalLM

    solution.apply()  # same wiring as harness/_train.py: example SwiGLU + K1..K4

    torch.manual_seed(0)
    cfg = AutoConfig.for_model(
        "qwen2",
        vocab_size=512,  # % 128 == 0
        hidden_size=256,  # % 64 == 0, head_dim = 128
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
    )
    model = Qwen2ForCausalLM(cfg).cuda()
    # the kbit flow: body frozen and upcast to fp32, lm_head fp32 and frozen
    for p in model.parameters():
        p.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)  # stand-in for LoRA
    model.lm_head.weight.requires_grad_(False)

    hook = integrate.engage(model)
    assert hook is not None, "engage must hook the tiny Qwen2 instance"

    try:
        B, S = 2, 12
        am = torch.ones(B, S, dtype=torch.long, device="cuda")
        am[:, -3:] = 0
        am[1, :1] = 0
        ids = torch.randint(0, 512, (B, S), device="cuda")
        labels = torch.randint(0, 512, (B, S), device="cuda")
        labels[am == 0] = -100
        labels[:, 0] = -100

        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=am, labels=labels,
                        use_cache=False)
        assert out.loss is not None
        assert out.logits is None, "chunked head must own the forward"
        loss_v = float(out.loss.detach().item())
        assert 0.0 < loss_v < 100.0 and loss_v == loss_v  # finite, positive-ish

        out.loss.backward()
        emb_grad = model.model.embed_tokens.weight.grad
        assert emb_grad is not None and emb_grad.abs().sum().item() > 0
        assert torch.isfinite(emb_grad).all()
        assert model.lm_head.weight.grad is None  # frozen head
        assert hook.step is not None and hook.step.token_loss.dim() == 2

        # second step (fresh grad graph, the cache path stays sane)
        model.zero_grad(set_to_none=True)
        labels2 = torch.randint(0, 512, (B, S), device="cuda")
        labels2[am == 0] = -100
        labels2[:, 0] = -100
        with torch.autocast("cuda", dtype=torch.float16):
            out2 = model(input_ids=ids, attention_mask=am, labels=labels2,
                         use_cache=False)
        out2.loss.backward()
        assert float(out2.loss.detach().item()) == float(out2.loss.detach().item())
        if not cfg.tie_word_embeddings:
            assert model.lm_head.weight.device.type == "cpu"
        # The fp32 master must be restored for a stock no-grad inference.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            inferred = model(input_ids=ids, attention_mask=am, use_cache=False)
        assert inferred.logits.is_cuda
        assert model.lm_head.weight.is_cuda
    finally:
        integrate.disengage()


def test_head_real_geometry():
    """K3 at the exact challenge geometry (V=151936, K=1536, 4x512 tokens):
    compact rows, stock CE with a Triton backward, and the fp32-to-fp16 head cast. Catches
    head-path breakage in ~10 s instead of a full train roundtrip."""
    import solution.ck_head as head

    torch.manual_seed(0)
    B, S, K, V = 4, 512, 1536, 151936
    # the real flow: peft's kbit preparation upcasts the body (and the last
    # hidden state) to fp32, so the head really gets an fp32 X
    hidden = torch.randn(B, S, K, device="cuda", dtype=torch.float32,
                         requires_grad=True)
    weights = torch.randn(V, K, device="cuda", dtype=torch.float32)
    am = torch.ones(B, S, dtype=torch.long, device="cuda")
    am[:, 300:] = 0                      # long right padding, like the task
    labels = torch.full((B, S), -100, dtype=torch.long, device="cuda")
    # completion-only loss: only the tail of each row is supervised
    for b in range(B):
        lo = 100 + b
        hi = 250 + b
        labels[b, lo:hi] = torch.randint(0, V, (hi - lo,), device="cuda")
    num_items = torch.tensor(int((labels[:, 1:] != -100).sum()),
                             device="cuda", dtype=torch.long)

    hook = head.CKHeadHook()
    loss, step = head.ck_chunked_lm_loss(hidden, labels, weights, am,
                                         num_items, hook)
    assert loss is not None, "real geometry must engage the chunked head"
    loss.backward()
    assert torch.isfinite(loss).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert float(hidden.grad.abs().sum().item()) > 0
    assert step is not None and step.token_loss.shape == (B, S)
    assert torch.isfinite(step.token_loss).all()
