"""Compile normal training specializations concurrently inside the timed run.

No kernels are executed here: Triton's warmup API compiles, and its runtime
handles are initialized without calling the launcher. All worker
threads are joined before this function returns. The harness's fresh cache
directory, GPU visibility and external whole-command timer are unchanged.
"""

import os
import time
import atexit
from threading import RLock
from concurrent.futures import ThreadPoolExecutor

import torch
import triton

FIXED_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FIXED_SCALE = 2.0
_PREPARED_MODELS = set()
_EARLY = None
_EARLY_LOCK = RLock()


def prepared_for_model(name, scale):
    """Reuse only preparation of this exact model and LoRA scale in this process."""
    return (str(name), float(scale)) in _PREPARED_MODELS


def early_for_model(name, scale):
    with _EARLY_LOCK:
        return _EARLY is not None and _EARLY[2] == (str(name), float(scale))


def start_early(name=FIXED_MODEL, scale=FIXED_SCALE):
    """Begin compile-only preparation; the setup boundary must close it."""
    global _EARLY
    key = (str(name), float(scale))
    with _EARLY_LOCK:
        if prepared_for_model(*key):
            return False
        if _EARLY is not None:
            return _EARLY[2] == key
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ck_early")
        try:
            future = pool.submit(prepare_model_name, *key)
        except BaseException:
            pool.shutdown(wait=True)
            raise
        _EARLY = (pool, future, key, time.perf_counter())
        return True


def close_early(name=None, scale=None):
    """Join every preparation worker, even on failed/unsupported setup.

    A result is reusable only for the requested exact model/LoRA scale.
    Holding the lock through shutdown prevents a second preparation from
    starting before the current workers have actually stopped.
    """
    global _EARLY
    expected = None if name is None else (str(name), float(scale))
    with _EARLY_LOCK:
        if _EARLY is None:
            return expected is not None and prepared_for_model(*expected)
        pool, future, key, started = _EARLY
        prepared = False
        try:
            prepared = bool(future.result())
        except Exception as exc:
            print(f"[ck] early compiler fallback: {type(exc).__name__}", flush=True)
        finally:
            try:
                pool.shutdown(wait=True)
            finally:
                _EARLY = None
        print(f"[ck] early compiler workers joined: "
              f"{time.perf_counter() - started:.2f} s elapsed", flush=True)
        return prepared and expected == key and prepared_for_model(*expected)


# Some GPU unit tests call apply() without constructing a Soup model. Normal
# training closes at setup; process exit is the cleanup fallback for those tests.
atexit.register(close_early)


def prepare_kernels(lm):
    close_early()
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
    prepared = prepare_config(AutoConfig.from_pretrained(name), scale)
    if prepared:
        _PREPARED_MODELS.add((str(name), float(scale)))
    return prepared


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
    add(head._ck_head_exact_kernel, (f, f, i32, vocab * h), BLOCK=8192, num_warps=8)
    add(head._ck_embedding_kernel, (i64, half, f, 1024 * h, h, vocab), BLOCK=1024)
    add(head._ck_lse_kernel, (f, i32, f, i32, f, f, vocab), BLOCK=4096, num_stages=1)
    add(head._ck_ce_grad_kernel, (f, f, i32, half, vocab, f, f),
        BLOCK=1024, DEN_TENSOR=True)
    add(head._ck_scatter_grad_kernel, (half, i64, f, f, half, 1024 * h), H=h, BLOCK=1024)
    add(nf4._ck_nf4_kernel, (u8, u8, f, f, f, f, half, h * width),
        NESTED=True, BLOCK=1024, enable_fp_fusion=False)
    add(nf4._ck_nf4_pair_kernel, (u8, u8, u8, u8, f, f, f, f, f, f, f, f, half, h * width),
        NESTED=True, BLOCK=1024, enable_fp_fusion=False)
    for compact in (False, True):
        add(lora._ck_lora_add_kernel, (f, half, f, 1024 * h, i32 if compact else f, h),
            COMPACT=compact, SCALE=scale, BLOCK=1024, enable_fp_fusion=False)
        add(lora._ck_lora_grad_kernel, (f, half, 1024 * h, i64 if compact else f, h),
            COMPACT=compact, SCALE=scale, BLOCK=1024)
        add(dropout._ck_dropout_pack_kernel, (f, boolean, half, u8, 1024 * h, i32 if compact else u8, h),
            COMPACT=compact, BLOCK=1024)
        add(dropout._ck_dropout_bwd_kernel, (half, u8, f, 1024 * h, 1.0 / 0.95, i32 if compact else u8, h),
            COMPACT=compact, BLOCK=1024)
    for rows in (1024, 1032, 1023):
        add(swiglu.example_swiglu_fwd_kernel, (half, half, half, rows, width, 2 * width),
            BM=swiglu._SW_BM, BN=swiglu._SW_BN, num_warps=8, num_stages=2)
        add(swiglu.example_swiglu_bwd_kernel,
            (half, half, half, half, half, half, half, rows, width, 2 * width, 2 * width),
            BM=swiglu._SW_BM, BN=swiglu._SW_BN, COMPRESSED=False,
            PACK=activation.PACK_BLOCK, num_warps=8, num_stages=2)
    add(padding._ck_mlp_residual_kernel, (f, half, i32, f, 1024 * h), H=h, BLOCK=1024)
    for source in (f, half):
        # gather_rows allocates with hidden.new_empty, retaining its dtype.
        add(padding._ck_gather_rows_kernel, (source, i64, source, 1024 * h), H=h, BLOCK=1024)
    kv = (getattr(cfg, "head_dim", None) or h // cfg.num_attention_heads) * cfg.num_key_value_heads
    for columns in sorted({h, kv}):
        add(padding._ck_scatter_rows_kernel, (half, i32, half, 1024 * columns),
            H=columns, BLOCK=1024)
    add(padding._ck_scatter_rows_kernel, (f, i32, f, 1024 * h), H=h, BLOCK=1024)
    for attention_type in (half, f):
        add(residual._ck_residual_norm_fwd_kernel,
            (f, attention_type, f, i32, f, half, half, f, eps), H=h, BLOCK=block)
        add(residual._ck_residual_norm_bwd_kernel,
            (f, half, f, i32, half, f, f, attention_type), H=h, BLOCK=block)
    hq, hk = cfg.num_attention_heads, cfg.num_key_value_heads
    dim = getattr(cfg, "head_dim", None) or h // hq
    for backward in (False, True):
        for sequence in (512, 511):
            qb, qh, qs = (sequence * h, sequence * dim, dim) if backward else (sequence * h, dim, h)
            kb, kh, ks = (sequence * kv, sequence * dim, dim) if backward else (sequence * kv, dim, kv)
            for table_batch in (0, sequence * dim):
                add(rope._ck_rope_pair_kernel,
                    (f, f, f, f, f, f, sequence, hq, hk, qb, qh, qs, kb, kh, ks,
                     table_batch, dim, table_batch, dim),
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
