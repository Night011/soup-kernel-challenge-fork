"""Compile normal training specializations concurrently inside the timed run.

No kernels are executed here: Triton's warmup API compiles, and its runtime
handles are initialized without calling the launcher. All worker
threads are joined before this function returns. The harness's fresh cache
directory, GPU visibility and external whole-command timer are unchanged.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import triton


def prepare_kernels(lm):
    if getattr(lm, "_ck_kernels_prepared", False):
        return
    layers = getattr(lm.model, "layers", ())
    if not layers:
        return
    gate = getattr(getattr(layers[0].mlp, "gate_proj", None), "weight", None)
    if getattr(getattr(gate, "quant_state", None), "quant_type", None) != "nf4":
        return
    q_proj = layers[0].self_attn.q_proj
    adapter = getattr(q_proj, "active_adapters", ["default"])[0]
    scale = float(getattr(q_proj, "scaling", {}).get(adapter, 2.0))
    lm._ck_kernels_prepared = prepare_config(lm.config, scale)


def prepare_model_name(name, scale):
    from transformers import AutoConfig
    return prepare_config(AutoConfig.from_pretrained(name), scale)


def prepare_config(cfg, scale=2.0):
    h, width, vocab = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
    if h % 512 or vocab % 64:
        return
    from . import ck_rms as rms, ck_rope as rope, ck_head as head
    from . import ck_nf4 as nf4, ck_lora as lora, ck_dropout as dropout
    from . import ck_padding as padding, ck_residual_norm as residual, ck_activation as activation
    from . import ck_attention as attention
    from . import swiglu

    f, half, i32, i64, u8, i8, boolean = (torch.float32, torch.float16, torch.int32,
                                         torch.int64, torch.uint8, torch.int8, torch.bool)
    groups = {}

    def add(kernel, args, **kwargs):
        kwargs.setdefault("num_warps", 4)
        groups.setdefault(id(kernel), [kernel, []])[1].append((args, kwargs))

    eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
    block = triton.next_power_of_2(h)
    # Runtime scalar values select their usual divisibility classes; dtype
    # arguments become MockTensor objects in JITFunction.warmup (no allocation).
    for output in (f, half):
        add(rms._ck_rms_fwd_kernel, (f, f, output, half, f, eps),
            H=h, BLOCK=block, SAVE_FP16=True, num_stages=2)
        add(rms._ck_rms_bwd_kernel, (half, output, f, f, f),
            H=h, BLOCK=block, num_stages=2)
    add(head._ck_cast_kernel, (f, half, vocab * h), BLOCK=8192, num_warps=8)
    add(head._ck_embedding_kernel, (i64, half, f, 1024 * h, h, vocab), BLOCK=1024)
    add(head._ck_lse_kernel, (f, i32, f, i32, f, f, vocab), BLOCK=4096, num_stages=1)
    add(head._ck_ce_grad_kernel, (f, f, i32, half, vocab, f, f),
        BLOCK=1024, DEN_TENSOR=True)
    add(head._ck_scatter_grad_kernel, (half, i64, f, f, half, 1024 * h), H=h, BLOCK=1024)
    add(nf4._ck_nf4_kernel, (u8, u8, f, f, f, f, half, h * width),
        NESTED=True, BLOCK=1024, enable_fp_fusion=False)
    add(nf4._ck_nf4_pair_kernel, (u8, u8, u8, u8, f, f, f, f, f, f, f, f, half, h * width),
        NESTED=True, BLOCK=1024, enable_fp_fusion=False)
    add(lora._ck_lora_add_kernel, (f, half, f, 1024 * h),
        SCALE=scale, BLOCK=1024, enable_fp_fusion=False)
    add(lora._ck_lora_grad_kernel, (f, half, 1024 * h), SCALE=scale, BLOCK=1024)
    add(dropout._ck_dropout_pack_kernel, (f, boolean, half, u8, 1024 * h), BLOCK=1024)
    add(dropout._ck_dropout_bwd_kernel, (half, u8, f, 1024 * h, 1.0 / 0.95), BLOCK=1024)
    add(activation._ck_pack_activations_kernel, (half, half, i8, i8, f, f, 1024 * width),
        COLS=width, STRIDE=2 * width, BLOCK=activation.PACK_BLOCK)
    for rows in (1024, 1032):
        add(swiglu.example_swiglu_fwd_kernel, (half, half, half, rows, width, 2 * width),
            BM=swiglu._SW_BM, BN=swiglu._SW_BN, num_warps=8, num_stages=2)
        add(swiglu.example_swiglu_bwd_kernel,
            (half, i8, i8, f, f, half, half, rows, width, width, 2 * width),
            BM=swiglu._SW_BM, BN=swiglu._SW_BN, COMPRESSED=True,
            PACK=activation.PACK_BLOCK, num_warps=8, num_stages=2)
    add(padding._ck_mlp_residual_kernel, (f, half, i32, f, 1024 * h), H=h, BLOCK=1024)
    for source in (f, half):
        add(padding._ck_gather_rows_kernel, (source, i64, half, 1024 * h), H=h, BLOCK=1024)
    kv = (getattr(cfg, "head_dim", None) or h // cfg.num_attention_heads) * cfg.num_key_value_heads
    for columns in sorted({h, kv}):
        add(padding._ck_scatter_rows_kernel, (half, i32, half, 1024 * columns),
            H=columns, BLOCK=1024)
    add(residual._ck_residual_norm_fwd_kernel, (f, half, f, i32, f, half, half, f, eps),
        H=h, BLOCK=block)
    add(residual._ck_residual_norm_bwd_kernel, (f, half, f, i32, half, f, f, half),
        H=h, BLOCK=block)
    hq, hk = cfg.num_attention_heads, cfg.num_key_value_heads
    dim = getattr(cfg, "head_dim", None) or h // hq
    for backward in (False, True):
        for sequence in (512, 511):
            qb, qh, qs = (sequence * h, sequence * dim, dim) if backward else (sequence * h, dim, h)
            kb, kh, ks = (sequence * kv, sequence * dim, dim) if backward else (sequence * kv, dim, kv)
            add(rope._ck_rope_pair_kernel,
                (f, f, f, f, f, f, sequence, hq, hk, qb, qh, qs, kb, kh, ks, 0, dim, 0, dim),
                RD=dim, BR=rope._ROPE_ROWS, BACKWARD=backward)
    for sequence in (512, 511):
        add(attention._ck_attention_pack_kernel,
            (f, f, f, i64, half, half, half, 1024 * hq * dim, sequence,
             sequence * hq * dim, sequence * dim, dim,
             sequence * hk * dim, sequence * dim, dim,
             sequence * hk * dim, dim, hk * dim),
            HQ=hq, HK=hk, D=dim, BLOCK=1024)
        add(attention._ck_attention_unpack_grad_kernel,
            (half, half, half, i32, f, f, f, sequence),
            HQ=hq, HK=hk, D=dim, BR=8)

    started = time.perf_counter()
    # Initialize the normal CUDA compilation target once, on the main thread.
    from triton.runtime import driver
    driver.active.get_current_target()
    device = torch.cuda.current_device()

    def compile_group(group):
        kernel, specs = group
        torch.cuda.set_device(device)
        failures = []
        for args, kwargs in specs:
            try:
                compiled = kernel.warmup(*args, grid=(1, 1), **kwargs)
                # Triton normally defers launcher creation and CUDA module
                # loading until the first real invocation. Prepare those
                # handles in the same timed setup, without launching work.
                initialize = getattr(compiled, "_init_handles", None)
                if initialize is not None:
                    initialize()
            except Exception as exc:
                # Unsupported warmup signatures retain normal on-demand JIT.
                failures.append((kernel.__name__, type(exc).__name__))
        return failures

    workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ck_compile") as pool:
        failures = [failure for result in pool.map(compile_group, groups.values()) for failure in result]
    count = sum(len(group[1]) for group in groups.values())
    print(f"[ck] in-process parallel compile: {count - len(failures)}/{count} variants, "
          f"{workers} joined threads, {time.perf_counter() - started:.2f} s", flush=True)
    if failures:
        print(f"[ck] on-demand JIT fallback for: {failures}", flush=True)
    return True
