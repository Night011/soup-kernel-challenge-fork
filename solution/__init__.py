"""Our solution: own Triton kernels, wired into the fixed `soup train` run.

apply() is called once by the harness before Soup builds model/trainer:

* `patch_qwen2_mlp`      — the starter-kit example (not counted as our kernel).
* `patch_qwen2_rmsnorm`  — K1: fused RMSNorm fwd+bwd (`solution/ck_rms.py`).
* `patch_qwen2_rope`     — K2: fused rotary embedding fwd+bwd (`ck_rope.py`).
* `integrate.apply_all`  — K3: chunked fused lm_head+CE without full logits
  (`ck_head.py`), K4 batch edge trim, TRL metric/rewind wiring, and the hook
  that swaps the causal-LM forward at Soup's documented setup point.
"""

from .swiglu import patch_qwen2_mlp
from .ck_rms import patch_qwen2_rmsnorm
from .ck_rope import patch_qwen2_rope
from . import integrate


def apply():
    patch_qwen2_mlp()  # the example: works out of the box, does not count
    patch_qwen2_rmsnorm()
    patch_qwen2_rope()
    integrate.apply_all()
