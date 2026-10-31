"""Our solution: own Triton kernels, wired into the fixed `soup train` run.

apply() is called once by the harness before Soup builds model/trainer:

* `patch_qwen2_mlp`      — the starter-kit example (not counted as our kernel).
* `patch_qwen2_rmsnorm`  — K1: fused RMSNorm fwd+bwd (`solution/ck_rms.py`).
* `patch_qwen2_rope`     — K2: fused rotary embedding fwd+bwd (`ck_rope.py`).
* `integrate.apply_all`  — K3: chunked fused lm_head+CE without full logits
  (`ck_head.py`), K4 batch edge trim, TRL metric/rewind wiring, and the hook
  that swaps the causal-LM forward at Soup's documented setup point.
"""

import os


def _setup_triton_cache():
    """Keep Triton's compiler cache on Kaggle's persistent volume.

    The harness exports TRITON_CACHE_DIR for every measured run (a fresh
    directory per run), so this is a no-op there and never reuses a prebuilt
    cache.  In an interactive notebook the cache survives a restart.
    """
    if os.environ.get("TRITON_CACHE_DIR"):
        return
    root = "/kaggle/working"
    if not os.path.isdir(root):
        return
    path = os.path.join(root, ".triton_cache")
    try:
        os.makedirs(path, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = path
        print(f"[ck] triton cache: {path}", flush=True)
    except OSError:
        pass


_setup_triton_cache()

from .swiglu import patch_qwen2_mlp  # noqa: E402
from .ck_rms import patch_qwen2_rmsnorm  # noqa: E402
from .ck_rope import patch_qwen2_rope  # noqa: E402
from . import integrate  # noqa: E402


def apply():
    """Ordered wiring; never raises — a failed patch prints and falls back to stock."""
    import traceback

    print("[ck] apply: begin", flush=True)
    for label, fn in (
        ("swiglu", patch_qwen2_mlp),
        ("rmsnorm", patch_qwen2_rmsnorm),
        ("rope", patch_qwen2_rope),
        ("integrate", integrate.apply_all),
    ):
        try:
            fn()
            print(f"[ck] apply: {label} ok", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"[ck] apply: {label} FAILED (stock path stays in place)",
                  flush=True)
    print("[ck] apply: done", flush=True)
