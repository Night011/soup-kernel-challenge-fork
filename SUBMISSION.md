# Submission: BAITC

## Links
- Repository, final commit: https://github.com/<you>/<repo>/commit/<sha>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link must
  be in your README):
  https://www.kaggle.com/code/<you>/<notebook>?scriptVersionId=<version>
- Demo video (up to 3 minutes): TBD
- Presentation: TBD

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| TBD | Triton kernels | | | |
| TBD | Integration + metrics | | | |
| TBD | Measurements | | | |
| TBD | Docs / submission | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `_ck_rms_fwd_kernel` / `_ck_rms_bwd_kernel` | `solution/ck_rms.py` | `Qwen2RMSNorm` element-wise chain, forward + backward | TBD % |
| `_ck_rope_fwd_kernel` / `_ck_rope_bwd_kernel` | `solution/ck_rope.py` | `apply_rotary_pos_emb` on q and k, forward + backward | TBD % |
| `_ck_cast_kernel` | `solution/ck_head.py` | fp32→fp16 cast of the frozen lm_head weight (cached fp16 copy) | TBD % |
| `_ck_lse_fwd_kernel` | `solution/ck_head.py` | the full-logits GEMM `H @ W^T` fused with online-softmax (max/sum/argmax partials per vocab tile) | TBD % |
| `_ck_lse_reduce_kernel` | `solution/ck_head.py` | merges tile partials into per-row logsumexp + argmax | TBD % |
| `_ck_nll_kernel` | `solution/ck_head.py` | gathered per-target dot `H[r]·W[y]` → per-token nll (no fp32 copy of logits) | TBD % |
| `_ck_dlogits_kernel` | `solution/ck_head.py` | softmax-gradient sweep + entropy atomics + fp16 dlogits write (second GEMM sweep) | TBD % |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- chunked linear+CE without logits — Liger-Kernel `FusedLinearCrossEntropy` idea
  (Apache-2.0): https://github.com/linkedin/Liger-Kernel
- online softmax accumulation — FlashAttention: https://arxiv.org/abs/2205.14135
- fp16 head-weight cache = the tensor autocast hands to cuBLAS — PyTorch AMP rules.
- TRL 0.29.1 metric formulas replicated verbatim (`entropy`, `num_tokens`,
  `mean_token_accuracy`).
- Soup's own logits-free chunked-step contract (#1693,
  `soup_cli.utils.stream_loss_chunk` / `rewind_hf:row_losses_from_token_losses`)
  to keep `rewind.jsonl` intact — we reuse the contract, not the code.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | TBD | TBD | speedup x |
| Peak GPU memory, MiB | TBD | TBD | base/ours x |
| Eval loss (public rows) | TBD | TBD | % vs base |

Own-kernel share of the training time (check 4, steps 3+, CUDA events): TBD %.
Checks: TBD (of 10).
