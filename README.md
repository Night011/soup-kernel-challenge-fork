# Soup Kernel Challenge — our solution

Own Triton kernels for a fixed `soup train` run (Qwen2.5-1.5B, NF4 + LoRA, batch 4,
seq 512, Kaggle T4). Everything is wired from `solution/apply()`; the task settings,
data order, steps, optimizer and Soup's outputs are untouched.

## What we changed

| Kernel (own `@triton.jit`) | File | Replaces |
|---|---|---|
| `_ck_rms_fwd_kernel` / `_ck_rms_bwd_kernel` | `solution/ck_rms.py` | element-wise chain of `Qwen2RMSNorm` (upcast, pow, mean, rsqrt, weight-mul, dtype casts) — one fused pass per row, forward emits fp16 directly (the value the fp16 autocast cast produces), residual stream stays fp32 |
| `_ck_rope_fwd_kernel` / `_ck_rope_bwd_kernel` | `solution/ck_rope.py` | `apply_rotary_pos_emb` (q * cos + rotate_half(q) * sin on q and k, fp32 math, fp16 out) with the inverse-rotary backward |
| `_ck_cast_kernel` | `solution/ck_head.py` | fp32→fp16 cast of the frozen lm_head weight into a cached fp16 copy (values equal to the autocast-cast weight the reference matmul uses) |
| `_ck_lse_fwd_kernel` | `solution/ck_head.py` | the full-logits GEMM `H @ W^T` fused with an online-softmax pass: per 128-row row-block and 128-wide vocab tile we keep running max/sum/argmax partials — full logits are never materialized |
| `_ck_lse_reduce_kernel` | `solution/ck_head.py` | merges per-tile max/sum partials into per-row logsumexp + first-wins argmax |
| `_ck_nll_kernel` | `solution/ck_head.py` | gathers the target's logit as a per-row dot of `H[r]·W[y]` (exact fp16 products, fp32 sums, the value tensor cores produce); per-token nll = logit_y − lse |
| `_ck_dlogits_kernel` | `solution/ck_head.py` | the softmax gradient sweep: recomputes per-tile logits, writes the fp16 dlogits tile `(p − onehot)/den` (zeros at ignored positions), atomic-adds per-token entropy `p·log p` |

`dH = dlogits_piece @ W16` stays a plain cuBLAS matmul done piece by piece so the
piece state is freed before the next piece is built.

Plus two non-kernel hooks:

* **K4, batch edge trim** (`solution/integrate.py:_trim_inputs`) — inside
  `transformers.Trainer._prepare_inputs`: drops only leading/trailing all-padding
  columns of every batch. Real tokens keep their ids (the window of the original
  position ids is passed explicitly), attention masks, labels and therefore
  `num_tokens`, losses, entropy/accuracy and the rewind rows are unchanged.
* **Metric + rewind wiring** (`solution/integrate.py:_ck_sft_compute_loss`) — while
  our hook owns the forward, runs `Trainer.compute_loss` and replicates TRL's
  metric formulas verbatim (entropy over attention-masked rows, `num_tokens`
  from `attention_mask.sum()`, `mean_token_accuracy` from argmax vs shifted
  labels) and fills Soup's `_soup_chunked_step` with per-token losses so
  `rewind.jsonl` stays recordable without full logits.

## Why the head + loss path was the target

The stock run materializes fp16 logits (4 × 512 × 151936), then fp32 copies for
log_softmax, cross-entropy and the entropy metric — several GB of transient
buffers per step. This is both the time spent in element-wise passes over the
whole vocab and the reason `soup train` sits close to the T4's memory ceiling.
Our chunked path holds at most one 256-row piece of logits (≈ 78 MiB fp16) plus
a cached fp16 copy of the head weight, and does both sweeps (logsumexp and
gradient) as Triton GEMMs against the fp16 weight.

## Quality

The loss value, its gradient and every logged metric follow the reference math:

* logits rounding: our GEMM accumulates in fp32 and rounds to fp16 exactly like
  cuBLAS materializing fp16 logits; logsumexp/CE then run in fp32 on those
  values, like `ForCausalLMLoss` does after `logits.float()`;
* the gradient of the mean CE is `(p − onehot) / num_items` with zeros at
  ignored positions, folded into `den = num_items_in_batch` the Trainer already
  computes;
* entropy and accuracy recompute the same per-token quantities as TRL, just in
  the chunked sweep instead of a full-logits tensor.

## Run it

```bash
python harness/run_task.py                 # base, then solution -> runs/result.json
python harness/check_rules.py              # the ten checks against runs/result.json
python -m pytest tests -q                  # Soup's own tests (they must keep passing)
python -m pytest solution -q               # our kernels vs plain PyTorch (needs a GPU)
python harness/run_task.py --profile       # plus a per-kernel CUDA profiler table
```

## Results on Kaggle T4

Filled from `runs/result.json` of the final notebook run (see also SUBMISSION.md):

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| `soup train` wall time, s | ~79–83 (harness README) | TBD | TBD |
| Peak GPU memory, MiB | 14,350 | TBD | TBD |
| Own-kernel share of training time (check 4) | — | TBD (steps 3+, CUDA events) | — |
| Eval loss (public rows) | 2.050 | TBD | — |

Checks: `runs/checks.json`.

## Ideas taken from other work, with links

We wrote all kernels here ourselves; the following ideas are credited:

* chunked fused linear + cross-entropy without materializing logits — the idea
  of Liger-Kernel's `FusedLinearCrossEntropy` (Apache-2.0),
  https://github.com/linkedin/Liger-Kernel, and of CUTLASS-style blockwise
  softmax epilogues;
* online (streaming) max/sum/argmax accumulation — FlashAttention's online
  softmax, https://arxiv.org/abs/2205.14135;
* the fp16 head-weight cache equals the weight tensor autocast hands to cuBLAS —
  PyTorch autocast rules, https://pytorch.org/docs/stable/amp.html;
* TRL's metric formulas (entropy per attention-masked row, `num_tokens`,
  `mean_token_accuracy`) are replicated verbatim from TRL 0.29.1,
  `trl/trainer/sft_trainer.py`;
* the logits-free per-token-loss protocol consumed by Soup's rewind recorder is
  Soup's own `#1693` chunked-step contract (`soup_cli.utils.stream_loss_chunk`,
  `soup_cli.trainer.rewind_hf:row_losses_from_token_losses`) — we reuse the
  contract, not the implementation.

## Repo state

* `solution/` — `ck_rms.py`, `ck_rope.py`, `ck_head.py`, `integrate.py`,
  `__init__.py` (apply wiring), `swiglu.py` (starter-kit example, unchanged),
  `test_kernels.py` (our correctness tests).
* `harness/`, `tests/`, `task/` — untouched.
* `SUBMISSION.md`, Kaggle notebook link — see SUBMISSION.md.

## Limits

* Kernels target exactly the Qwen2 geometry of the task (hidden 1536, V = 151936
  divisible by 128, 28 layers); anything else falls back to stock paths.
* The fp16 head-weight cache assumes the head is frozen (it is: LoRA does not
  target it); the cache key includes the weight version counter and re-casts on
  any change.
* Trim only removes all-padding edge columns; a batch with interior padding
  holes is left untouched.
