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
| `_ck_lse_kernel` | `solution/ck_head.py` | one program per row over the 128-row cuBLAS logits piece: max/argmax, sum(exp), entropy — produces per-row logsumexp, NLL and raw entropy without a full fp32 logits tensor |
| `_ck_ce_lse_kernel` / `_ck_ce_grad_kernel` | `solution/ck_head.py` | computes the normalizer and scaled softmax gradient for recomputed labelled logits, without materializing fp32 logits or probabilities |
| `_ck_select_logits_kernel` | `solution/ck_head.py` | gathers labelled rows of fp16 logits directly into the fp32 tensor consumed by stock cross-entropy |

The large forward/backward logits GEMMs and `dH = dlogits_piece @ W16` stay
cuBLAS matmuls, done piece by piece so only one 128-row piece is alive at a time.
No full logits buffer is stored. A handwritten large-vocab Triton GEMM is much
slower on a T4 and its compile time alone breaks the wall clock.

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
Our chunked path builds one 128-row cuBLAS logits piece (≈ 39 MiB fp16) and owns
the full-vocab softmax/metrics work with Triton. The backward recomputes one
128-row logits piece at a time and feeds it to the dlogits kernel, so peak
memory stays below the stock full-logits path; the large forward/backward logits
GEMMs and `dH = dlogits @ W16` stay cuBLAS. This removes the reference's fp32
logits copies without adding a full logits buffer.

## Quality

The loss value, its gradient and every logged metric follow the reference math:

* logits rounding: cuBLAS produces fp16 logits; the loss uses stock fp32
  cross-entropy and our Triton metric reductions read those fp16 values in
  fp32. The backward recomputes logits and applies the softmax gradient;
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

The following is one measured run of commit `83140ca`, before the current
backward change. Current code must be remeasured; these are not its results.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Harness wall time, s | 85.27 | 73.23 | 1.164× speedup |
| Peak GPU memory, MiB | 14,345 | 8,485 | 1.691× lower |
| Eval loss (public rows) | 2.04969 | 2.04964 | −0.002% |

That run's automatic kernel-share result included the starter-derived SwiGLU,
so it must not be used as evidence for the 5% own-kernel requirement. The
current implementation marks both SwiGLU kernels as examples; measure its own
share and all checks in a fresh run.

Checks: `runs/checks.json`.

## Ideas taken from other work, with links

Our head, normalization and rotary kernels are separate from the starter
example. The tiled SwiGLU is an adaptation of the starter example and does not
count as an own kernel. The following ideas are credited:

* chunked fused linear + cross-entropy without materializing full fp32 logits —
  the idea of Liger-Kernel's `FusedLinearCrossEntropy` (Apache-2.0),
  https://github.com/linkedin/Liger-Kernel, and of CUTLASS-style blockwise
  softmax epilogues;
* streaming max/sum/argmax reductions with a stable logsumexp, in the spirit of
  FlashAttention's online softmax, https://arxiv.org/abs/2205.14135;
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
  `__init__.py` (apply wiring), `swiglu.py` (tiled starter-example adaptation),
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
