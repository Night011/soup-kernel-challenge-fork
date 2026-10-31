# Soup Kernel Challenge — Atlas Solutions

Own Triton kernels and autograd functions for the fixed Soup 0.77.0 task on a Kaggle T4. The harness calls `solution.apply()`. Model, NF4 parameters, data and batch order, 64 steps, LoRA r=16 on q_proj/v_proj with dropout 0.05, optimizer and scheduler keep their starter settings. Only `solution/` and this README are edited.

Platform repository: [BAITC-Hacks/hack-d5eb16fb-atlas-solutions](https://github.com/BAITC-Hacks/hack-d5eb16fb-atlas-solutions).
Development [Kaggle notebook](https://www.kaggle.com/code/hoogle57/notebook00ca772363). The final public saved version and exact `COMMIT` still need to be recorded after optimization; this development link is not a completed submission.

## Measured results

The last complete two-repeat comparison is [6e16e7d](https://github.com/BAITC-Hacks/hack-d5eb16fb-atlas-solutions/commit/6e16e7d), Kaggle directory `runs_full_6e16e7d`:

| Metric | Base Soup | Solution | Ratio |
|---|---:|---:|---:|
| Whole soup train, mean of two runs | 77.11 s | 61.52 s | 1.2534× speedup |
| Peak GPU memory | 14348 MiB | 5040 MiB | 2.8468× base/solution |
| Public eval loss | 2.04965 | 2.04967 | +0.001% |

Solution times: 61.19 and 61.84 s. **All 10 rule checks passed**, including 79 Soup regression tests. Own-kernel shares: 13.316% and 12.975%; the lower value counts. Starter SwiGLU is excluded.

Further development has changed the solution since that full comparison. An exploratory solution-only run of [7a8ba00](https://github.com/BAITC-Hacks/hack-d5eb16fb-atlas-solutions/commit/7a8ba00) gave **61.95 s, 4460 MiB (4.36 GiB), eval 2.04968**, own share **17.72%**, and **50/50 GPU tests**. It uses compact projections, paired NF4 decoding and a 768 MiB cache budget. That single run cannot establish the base-dependent rule checks. Later commits require fresh measurement.

## Implementation

**Head, loss and metrics.** `ck_head.py` processes 256 rows at a time instead of retaining full logits over 151936 vocabulary entries. cuBLAS multiplies fp16 hidden states and head weights with fp32 output. A streaming Triton reduction obtains logsumexp, NLL, entropy and argmax. The same NLL feeds loss and rewind. The local hidden derivative is computed while logits are live, carrying the current AMP scale; backward scatters it with the actual incoming scale. It does not recompute logits in backward.

The head keeps the union of attention-masked rows and rows needed by shifted labels. A padding query can predict a real token, so the attention mask alone is insufficient. Labelled rows are put first inside the head to compact gradient GEMMs, without changing sample order.

The latest collator wrapper derives those indices, padding layout and token count from the same CPU batch before device transfer. They travel with that batch and are removed from model kwargs. A GPU scalar denominator is loaded inside the CE kernel without a host read. Entropy/accuracy are transferred before TRL writes their normal log entry, rather than blocking inside head backward. GPU preparation remains a fallback when CPU metadata is unavailable.

**Frozen head/embedding storage.** The original fp32 master remains intact on CPU; a fp16 compute copy stays on GPU. For tied embeddings this is enabled only if converting the copy back to fp32 exactly reproduces the master. An own lookup kernel emits those original fp32 embedding values. Stock no-grad forwards and disengagement restore the master. External adapter evaluation uses vanilla transformers/PEFT without our code.

**NF4 and paired MLP.** Our decoder fuses NF4 lookup, double-quant scale decoding and offset addition, using the original codebooks. Tests require bitwise equality to bitsandbytes decoding. A separate kernel decodes gate/up directly into a combined matrix: one forward projection GEMM and one input-gradient GEMM replace two of each. The versioned fp16 cache now has a fixed 320 MiB budget, reserving complete gate/up pairs together. Uncached pairs are decoded on demand, not retained in autograd. Packed model weights remain frozen.

**Saved activations.** Forward uses the original fp16 gate/up values. Only backward storage is compressed to symmetric INT8 with a fp32 scale per 1024 values. RMSNorm saves fp16 input plus its original fp32 inverse standard deviation. These are small backward approximations, not bitwise-equivalence claims. Gradient tests and the complete public loss/metric/eval checks bound their observed effect; hidden quality remains to be checked by the jury.

**Padding.** `integrate.py` trims all-padding edge columns and retains original position ids. `ck_padding.py` compacts real token rows in frozen MLPs and base NF4 projections, retaining a few pad rows to align row counts to eight. Attention and LoRA/dropout keep their full batch shapes. A shifted label requiring a padding query disables row compaction for that batch. Checkpoint replay and cached decoding retain full rows. No rows move between batches. Own gather/scatter kernels restore layouts; another fuses MLP scatter with residual addition.

**Other fusion.** RMSNorm has own forward/backward kernels. The latest RoPE rotates Q and K together in one launch per direction, supporting GQA and strided views. LoRA scale/add preserves PEFT's two fp16 rounding points. Native PyTorch dropout generates its original mask; our kernels pack it to bits and combine packing with the following fp16 cast. Tests compare outputs, gradients and the subsequent CUDA RNG stream.

## Own kernels

| File | Own kernels | Work |
|---|---|---|
| `ck_head.py` | `_ck_cast_kernel`, `_ck_embedding_kernel`, `_ck_lse_kernel`, `_ck_ce_grad_kernel`, `_ck_scatter_grad_kernel` | Cast, lookup, loss/metrics, CE derivative and gradient scatter |
| `ck_nf4.py` | `_ck_nf4_kernel`, `_ck_nf4_pair_kernel` | Exact single/paired NF4 decoding |
| `ck_rms.py` | `_ck_rms_fwd_kernel`, `_ck_rms_bwd_kernel` | RMSNorm and dtype boundaries |
| `ck_residual_norm.py` | `_ck_residual_norm_fwd_kernel`, `_ck_residual_norm_bwd_kernel` | Attention residual, compact MLP RMSNorm and joint gradient |
| `ck_rope.py` | `_ck_rope_pair_kernel` | Joint Q/K rotation and backward |
| `ck_lora.py` | `_ck_lora_add_kernel`, `_ck_lora_grad_kernel` | Adapter scale/add and gradient |
| `ck_dropout.py` | `_ck_dropout_pack_kernel`, `_ck_dropout_bwd_kernel` | Native-mask packing/cast and gradient |
| `ck_activation.py` | `_ck_pack_activations_kernel` | Compact backward activation storage |
| `ck_padding.py` | `_ck_mlp_residual_kernel`, `_ck_gather_rows_kernel`, `_ck_scatter_rows_kernel` | Compact row movement and residual addition |

All files above are in `solution/`. `swiglu.py` derives from the starter example: its `example_*` kernels, including compressed-storage and stride adaptations, are excluded from our share. Large GEMMs and SDPA remain PyTorch/cuBLAS work. The unused `_ck_ce_lse_kernel` is not claimed either.

## Reproduce on Kaggle T4

Use the template notebook with Internet and T4 x2, the platform repository URL and an exact commit. Pin Soup 0.77.0, transformers 5.16.1, PEFT 0.20.0, TRL 0.29.1, bitsandbytes 0.50.2 and accelerate 1.14.0, as in the template. The harness uses one GPU and requires idle GPUs at the start.

```bash
python task/prepare_data.py
python -m pytest solution -q
python harness/run_task.py --repeats 2 --out runs_final
python harness/check_rules.py --result runs_final/result.json
python harness/score.py runs_final/result.json
```

Use a new output directory per comparison. During development, `--only solution --out runs_try_<commit>` is faster but cannot establish the base-dependent checks. `--profile` adds a separate profiled run whose wall time is not the solution timing. Imports, cold JIT and adapter saving count in the measured command. The solution does not change the harness allocator, GPU selection or compiler-cache settings, or install packages during measurement.

The latest compiler preparation uses Triton's compile-only warmup API with up to four compiler threads inside the measured process. Compilation overlaps ordinary model loading within the setup function; all threads are joined before that function returns and before training. The harness still starts from its own empty cache, and the full setup time remains in the wall-clock measurement. Warmup executes no GPU kernels and is not part of the claimed GPU kernel time; ordinary training launches supply that work.

## Limits

Optimized paths target frozen NF4 Qwen2 under fp16 autocast on T4. Unsupported layouts/models/PEFT modes use guarded fallbacks. No built-in Soup optimization switch is changed. CPU masters require host RAM and an initial transfer, whose time counts. The fixed cache budget trades memory for repeated decoding. INT8 saved activations and fp16 saved norm inputs are approximate; public results do not guarantee hidden-seed quality or a specific hidden-run time.

## Sources of ideas

New kernels are handwritten; no third-party kernel libraries or kernel implementations are imported/copied:

- Chunked linear/CE and its forward-computed derivative: [Liger fused linear cross-entropy](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py).
- Stable online softmax: [FlashAttention paper](https://arxiv.org/abs/2205.14135).
- Paired gate/up projections: [NVIDIA TensorRT-LLM post-load fusion](https://nvidia.github.io/TensorRT-LLM/features/auto_deploy/transforms/post_load_fusion.html).
- Removing padding work: [PyTorch transformer building blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html).
- FP16 matrix alignment: [NVIDIA matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).
- Compact dropout state: [Triton low-memory dropout](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html). Our path packs the actual native mask and preserves its RNG.
- NF4 semantics: installed bitsandbytes `QuantState`; dtype boundaries: [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html).

TRL 0.29.1 metric formulas and Soup's chunked-step/rewind contract are preserved in the integration. Starter SwiGLU attribution remains explicit in its source. See `LICENSE` and `NOTICE` for Apache-2.0 notices; task data/model retain their original licenses.
