"""Batch-local removal of unused MLP rows; attention and RNG shapes stay full.

Only frozen, token-wise MLPs are compacted. Padding queries needed by shifted
labels disable this optimization: a fully masked SDPA query can see unmasked
keys, so keeping only its last-layer MLP would not preserve its history.
"""

from contextlib import contextmanager

import torch
import triton
import triton.language as tl

_LAYOUT = None


def mlp_layout(attention_mask, labels, allow_cpu=False):
    if (not isinstance(attention_mask, torch.Tensor)
            or not isinstance(labels, torch.Tensor)
            or attention_mask.ndim != 2 or labels.shape != attention_mask.shape
            or (not attention_mask.is_cuda and not allow_cpu)
            or labels.device != attention_mask.device):
        return None
    active = attention_mask.bool()
    if bool(((labels[:, 1:] != -100) & ~active[:, :-1]).any().item()):
        return None
    rows = active.flatten().nonzero().flatten()
    total = active.numel()
    # Avoid launches when there is little padding; align FP16 GEMM rows to 8.
    if rows.numel() == 0 or rows.numel() * 20 >= total * 19:
        return None
    extra = (-rows.numel()) % 8
    if extra:
        padding = (~active).flatten().nonzero().flatten()[:extra]
        rows = torch.cat((rows, padding)).sort().values
    inverse = torch.full((total,), -1, device=rows.device, dtype=torch.int32)
    inverse[rows] = torch.arange(rows.numel(), device=rows.device, dtype=torch.int32)
    return rows, tuple(active.shape), inverse


@contextmanager
def compact_mlp_rows(attention_mask, labels, prepared=None):
    global _LAYOUT
    previous = _LAYOUT
    if (prepared is not None and isinstance(attention_mask, torch.Tensor)
            and tuple(attention_mask.shape) == tuple(prepared.get("shape", ()))):
        _LAYOUT = prepared.get("mlp_layout")
    else:
        _LAYOUT = mlp_layout(attention_mask, labels)
    try:
        yield
    finally:
        _LAYOUT = previous


def gather_norm_rows(norm, hidden):
    layout = _LAYOUT
    if (layout is None or not getattr(norm, "_ck_compact_mlp", False)
            or hidden.ndim != 3 or tuple(hidden.shape[:2]) != layout[1]):
        return hidden
    return hidden.reshape(-1, hidden.shape[-1]).index_select(0, layout[0])


def scatter_mlp_rows(mlp, hidden):
    layout = _LAYOUT
    if (layout is None or not getattr(mlp, "_ck_compact_mlp", False)
            or hidden.ndim != 2 or hidden.shape[0] != layout[0].numel()):
        return hidden
    if getattr(mlp, "_ck_fused_residual", False):
        return hidden
    rows, (batch, sequence) = layout[:2]
    full = hidden.new_zeros((batch * sequence, hidden.shape[-1]))
    return full.index_copy_(0, rows, hidden).view(batch, sequence, -1)


@triton.jit
def _ck_mlp_residual_kernel(RESIDUAL, PACKED, INVERSE, OUT, N,
                            H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed_row = tl.load(INVERSE + i // H, i < N, other=-1)
    value = tl.load(PACKED + tl.maximum(packed_row, 0) * H + i % H,
                    (i < N) & (packed_row >= 0), other=0).to(tl.float32)
    residual = tl.load(RESIDUAL + i, i < N, other=0).to(tl.float32)
    tl.store(OUT + i, residual + value, i < N)


@triton.jit
def _ck_gather_rows_kernel(GRAD, ROWS, OUT, N,
                            H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(ROWS + i // H, i < N, other=0)
    value = tl.load(GRAD + row * H + i % H, i < N, other=0)
    tl.store(OUT + i, value, i < N)


@triton.jit
def _ck_scatter_rows_kernel(PACKED, INVERSE, OUT, N,
                             H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(INVERSE + i // H, i < N, other=-1)
    value = tl.load(PACKED + tl.maximum(row, 0) * H + i % H,
                    (i < N) & (row >= 0), other=0)
    tl.store(OUT + i, value, i < N)


def projection_layout(hidden):
    if (_LAYOUT is not None and hidden.ndim == 3 and hidden.is_contiguous()
            and tuple(hidden.shape[:2]) == _LAYOUT[1]):
        return _LAYOUT
    return None


def gather_rows(hidden, layout):
    hidden = hidden.contiguous()
    rows = layout[0]
    out = hidden.new_empty((rows.numel(), hidden.shape[-1]))
    _ck_gather_rows_kernel[(triton.cdiv(out.numel(), 1024),)](
        hidden, rows, out, out.numel(), H=out.shape[-1], BLOCK=1024, num_warps=4,
    )
    return out


def scatter_rows(hidden, layout):
    hidden = hidden.contiguous()
    out = hidden.new_empty((*layout[1], hidden.shape[-1]))
    _ck_scatter_rows_kernel[(triton.cdiv(out.numel(), 1024),)](
        hidden, layout[2], out, out.numel(), H=out.shape[-1], BLOCK=1024, num_warps=4,
    )
    return out


class CompactMLPResidual(torch.autograd.Function):
    """Scatter plus residual addition in one pass, without a padded MLP buffer."""

    @staticmethod
    def forward(ctx, residual, packed, rows, inverse):
        residual, packed = residual.contiguous(), packed.contiguous()
        out = torch.empty_like(residual)
        hidden = residual.shape[-1]
        _ck_mlp_residual_kernel[(triton.cdiv(out.numel(), 1024),)](
            residual, packed, inverse, out, out.numel(), H=hidden, BLOCK=1024,
            num_warps=4,
        )
        ctx.save_for_backward(rows)
        ctx.packed_shape, ctx.packed_dtype = packed.shape, packed.dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        rows, = ctx.saved_tensors
        grad = grad.contiguous()
        packed_grad = torch.empty(ctx.packed_shape, device=grad.device,
                                   dtype=ctx.packed_dtype)
        _ck_gather_rows_kernel[(triton.cdiv(packed_grad.numel(), 1024),)](
            grad, rows, packed_grad, packed_grad.numel(), H=grad.shape[-1],
            BLOCK=1024, num_warps=4,
        )
        return grad, packed_grad, None, None


def _compact_decoder_forward(self, hidden_states, attention_mask=None,
                              position_ids=None, past_key_values=None,
                              use_cache=False, position_embeddings=None, **kwargs):
    layout = _LAYOUT
    if (layout is None or hidden_states.ndim != 3
            or tuple(hidden_states.shape[:2]) != layout[1]):
        return self._ck_decoder_original(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            position_embeddings=position_embeddings, **kwargs)
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask,
        position_ids=position_ids, past_key_values=past_key_values,
        use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
    from .ck_residual_norm import residual_compact_norm
    fused = residual_compact_norm(residual, hidden_states, self.post_attention_layernorm, layout)
    if fused is None:
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
    else:
        residual, hidden_states = fused
    hidden_states = self.mlp(hidden_states)
    return CompactMLPResidual.apply(residual, hidden_states, layout[0], layout[2])


def install_fused_residual(layer):
    import inspect
    import types

    original = layer.forward
    # Older transformers return a tuple and have a different layer contract.
    if inspect.signature(original).return_annotation not in (torch.Tensor, "torch.Tensor"):
        return False
    layer._ck_decoder_original = original
    layer.forward = types.MethodType(_compact_decoder_forward, layer)
    layer.mlp._ck_fused_residual = True
    return True


def remove_fused_residual(layer):
    original = getattr(layer, "_ck_decoder_original", None)
    if original is not None:
        layer.forward = original
        del layer._ck_decoder_original
    layer.mlp._ck_fused_residual = False
