"""Fuse PEFT's scaled adapter addition, preserving its dtype rounding.

The quantized base projection and both LoRA GEMMs keep their stock paths.
Only ``base + update * scaling`` and its backward are fused. In particular,
the product is rounded to update.dtype before the addition, as in PyTorch.
"""

import torch
import triton
import triton.language as tl
from .ck_dropout import compact_lora_dropout


@triton.jit
def _ck_lora_add_kernel(BASE, UPDATE, OUT, N, INVERSE, H,
                        COMPACT: tl.constexpr, SCALE: tl.constexpr,
                        BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = tl.load(BASE + i, i < N, other=0).to(tl.float32)
    if COMPACT:
        row = tl.load(INVERSE + i // H, i < N, other=-1)
        update = tl.load(UPDATE + tl.maximum(row, 0) * H + i % H,
                         (i < N) & (row >= 0), other=0)
    else:
        update = tl.load(UPDATE + i, i < N, other=0)
    scaled = (update.to(tl.float32) * SCALE).to(UPDATE.dtype.element_ty)
    tl.store(OUT + i, base + scaled.to(tl.float32), i < N)


@triton.jit
def _ck_lora_grad_kernel(GRAD, DUPDATE, N, ROWS, H,
                         COMPACT: tl.constexpr, SCALE: tl.constexpr,
                         BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Addition first casts its gradient to update.dtype; the multiply's
    # backward then scales and rounds it once more.
    if COMPACT:
        row = tl.load(ROWS + i // H, i < N, other=0)
        grad = tl.load(GRAD + row * H + i % H, i < N, other=0).to(DUPDATE.dtype.element_ty)
    else:
        grad = tl.load(GRAD + i, i < N, other=0).to(DUPDATE.dtype.element_ty)
    value = grad.to(tl.float32) * SCALE
    tl.store(DUPDATE + i, value, i < N)


class CKLoRAAddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, update, scale, layout=None):
        out = torch.empty_like(base)
        ctx.scale, ctx.update_dtype = scale, update.dtype
        ctx.layout, ctx.update_shape = layout, update.shape
        inverse = layout[2] if layout is not None else base
        _ck_lora_add_kernel[(triton.cdiv(base.numel(), 1024),)](
            base, update, out, base.numel(), inverse, base.shape[-1],
            COMPACT=layout is not None, SCALE=scale, BLOCK=1024,
            num_warps=4, enable_fp_fusion=False,
        )
        return out

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        du = torch.empty(ctx.update_shape, device=grad.device, dtype=ctx.update_dtype)
        rows = ctx.layout[0] if ctx.layout is not None else grad
        _ck_lora_grad_kernel[(triton.cdiv(du.numel(), 1024),)](
            grad, du, du.numel(), rows, grad.shape[-1],
            COMPACT=ctx.layout is not None, SCALE=ctx.scale, BLOCK=1024, num_warps=4,
        )
        return (grad, du, None, None)[:len(ctx.needs_input_grad)]


def patch_lora_add():
    """Specialize the vanilla, single-adapter autocast path in PEFT 0.20.

    Other modes retain PEFT's forward. Our addition is out of place, so it
    also replaces the defensive clone of the base projection in that path.
    """
    from peft.tuners.lora.bnb import Linear4bit

    original = Linear4bit.forward
    if getattr(original, "_ck_is_ours", False):
        return

    def forward(self, x, *args, **kwargs):
        adapters = self.active_adapters
        if (not x.is_cuda or not x.is_contiguous()
                or not torch.is_autocast_enabled()
                or self.disable_adapters or self.merged
                or len(adapters) != 1 or args or kwargs):
            return original(self, x, *args, **kwargs)
        adapter = adapters[0]
        if adapter not in self.lora_A or adapter in self.lora_variant:
            return original(self, x, *args, **kwargs)
        self._check_forward_args(x)
        from .ck_padding import projection_layout
        layout = projection_layout(x)
        base = self.base_layer(x)
        update = self.lora_B[adapter](
            self.lora_A[adapter](compact_lora_dropout(self.lora_dropout[adapter], x, layout))
        )
        scale = self.scaling[adapter]
        if (base.is_contiguous() and update.is_contiguous()
                and (base.shape == update.shape or layout is not None)
                and base.dtype in (torch.float16, torch.float32)
                and update.dtype == torch.float16):
            return CKLoRAAddFn.apply(base, update, scale, layout)
        if layout is not None:
            update = update.new_zeros((base.numel() // base.shape[-1], base.shape[-1])).index_copy(0, layout[0], update).view_as(base)
        return base.clone() + update * scale

    forward._ck_is_ours = True
    Linear4bit.forward = forward
