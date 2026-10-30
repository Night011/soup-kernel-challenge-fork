"""Correctness of our kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_swiglu_matches_pytorch():
    """Starter-kit example kernel (not ours, kept as the baseline check)."""
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
    torch.testing.assert_close(gate.grad.float(), dgate, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(up.grad.float(), dup, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("rows", [2048, 17])
def test_rmsnorm_matches_reference(rows):
    """K1: fused RMSNorm forward + backward vs the reference math."""
    import solution.ck_rms as rms

    torch.manual_seed(0)
    hidden = 1536
    eps = 1e-6
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32)
    w = torch.randn(hidden, device="cuda", dtype=torch.float32)
    dy = torch.randn(rows, hidden, device="cuda", dtype=torch.float16)

    xr = x.clone().requires_grad_(True)
    xf = xr.to(torch.float32)
    ref = w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) \
        .to(torch.float16)
    ref.backward(dy.float())
    ref_dx = xr.grad.clone()
    ref_y = ref.detach().clone()

    xo = x.clone().requires_grad_(True)
    out = rms.CKRMSNormFn.apply(xo, w, eps)
    out.backward(dy)
    torch.testing.assert_close(out, ref_y, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(xo.grad, ref_dx, rtol=2e-3, atol=2e-3)


def test_rope_matches_reference():
    """K2: fused RoPE forward + backward vs apply_rotary_pos_emb."""
    import solution.ck_rope as rope_mod
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    torch.manual_seed(0)
    B, Hq, Hk, S, RD = 2, 14, 2, 64, 128
    q = torch.randn(B, Hq, S, RD, device="cuda", dtype=torch.float16)
    k = torch.randn(B, Hk, S, RD, device="cuda", dtype=torch.float16)
    pos = torch.arange(S, device="cuda").float()
    inv = 1.0 / (10000 ** (torch.arange(0, RD, 2, device="cuda").float() / RD))
    freqs = pos[None, :, None] * inv[None, None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0).expand(B, -1, -1).contiguous()
    sin = emb.sin().unsqueeze(0).expand(B, -1, -1).contiguous()
    dq = torch.randn_like(q)
    dk = torch.randn_like(k)

    qr = q.clone().requires_grad_(True)
    kr = k.clone().requires_grad_(True)
    rq, rk = apply_rotary_pos_emb(qr, kr, cos, sin)
    gq, gk = torch.autograd.grad(
        (rq * dq).float().sum() + (rk * dk).float().sum(), (qr, kr))

    qo = q.clone().requires_grad_(True)
    ko = k.clone().requires_grad_(True)
    oval_q, oval_k = rope_mod.CKRopeFn.apply(qo, ko, cos, sin)
    ((oval_q * dq).float().sum() + (oval_k * dk).float().sum()).backward()
    torch.testing.assert_close(oval_q.float(), rq.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(oval_k.float(), rk.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(qo.grad, gq, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(ko.grad, gk, rtol=2e-3, atol=2e-3)


def test_head_loss_matches_reference():
    """K3: chunked fused head+CE vs fp16 logits + F.cross_entropy."""
    import solution.ck_head as head

    torch.manual_seed(0)
    B, S, K, V = 2, 32, 64, 512   # V % 128 == 0
    labelled_mid = 20
    hidden = torch.randn(B, S, K, device="cuda", dtype=torch.float16)
    weights = torch.randn(V, K, device="cuda", dtype=torch.float32)
    am = torch.ones(B, S, dtype=torch.long, device="cuda")
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
    ref_loss.backward()
    ref_dh = hr.grad.clone()
    ref_loss_v = float(ref_loss.detach().item())

    ho = hidden.clone().requires_grad_(True)
    hook = head.CKHeadHook()
    loss, step = head.ck_chunked_lm_loss(
        ho, labels, weights, am, num_items, hook)
    assert loss is not None, "head path must be engageable in this test"
    loss.backward()

    tol = 1e-3 if ref_loss_v < 2 else 2e-3
    assert abs(float(loss.detach().item()) - ref_loss_v) <= 3e-3 * max(1, ref_loss_v)
    torch.testing.assert_close(ho.grad, ref_dh, rtol=2e-2, atol=2e-2)
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
