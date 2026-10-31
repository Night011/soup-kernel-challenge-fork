"""Fused NF4 decoding, including the double-quantized block scales.

The representation and lookup tables come from bitsandbytes' QuantState.
This replaces its two dequantization kernels and the scale-offset addition;
the original quantized weights, codebooks, matmuls and gradients are retained.
"""

import torch
import triton
import triton.language as tl

_ORIGINAL_DEQUANT = None
_ORIGINAL_MATMUL_FORWARD = None
_ORIGINAL_MATMUL_BACKWARD = None
_CACHEABLE = set()
_DECODED = {}
_PAIRED = {}


def clear_weight_cache():
    _CACHEABLE.clear()
    _DECODED.clear()
    _PAIRED.clear()


def paired_mlp_weights(gate_weight, up_weight, states=None):
    """Decode gate/up directly into a combined matrix, cached only if budgeted."""
    if states is None:
        states = [getattr(w, "quant_state", None) for w in (gate_weight, up_weight)]
    if not can_pair_weights(gate_weight, up_weight, states):
        return None
    gp, up = gate_weight.data_ptr(), up_weight.data_ptr()
    cacheable = gp in _CACHEABLE and up in _CACHEABLE
    gs, us = states
    version = (_cache_key(gate_weight, gs), _cache_key(up_weight, us))
    cached = _PAIRED.get((gp, up)) if cacheable else None
    if cached is not None and cached[0] == version:
        return cached[1]
    n = gs.shape[0] * gs.shape[1]
    combined = torch.empty((2 * gs.shape[0], gs.shape[1]),
                            device=gate_weight.device, dtype=torch.float16)
    _ck_nf4_pair_kernel[(triton.cdiv(n, 1024), 2)](
        gate_weight, up_weight, gs.absmax, us.absmax, gs.code, us.code,
        gs.state2.code if gs.nested else gs.absmax,
        us.state2.code if us.nested else us.absmax,
        gs.state2.absmax if gs.nested else gs.absmax,
        us.state2.absmax if us.nested else us.absmax,
        gs.offset if gs.nested else gs.absmax,
        us.offset if us.nested else us.absmax,
        combined, n, NESTED=gs.nested, BLOCK=1024, num_warps=4,
        enable_fp_fusion=False,
    )
    if cacheable:
        _DECODED.pop(gp, None)
        _DECODED.pop(up, None)
        _PAIRED[(gp, up)] = (version, combined)
    return combined


def can_pair_weights(gate, up, states=None):
    if states is None:
        states = [getattr(w, "quant_state", None) for w in (gate, up)]
    for weight, state in zip((gate, up), states):
        if (state is None or not weight.is_cuda or weight.dtype != torch.uint8
                or not weight.is_contiguous() or state.dtype != torch.float16
                or state.quant_type != "nf4" or state.blocksize != 64
                or len(state.shape) != 2 or not state.absmax.is_cuda
                or not state.code.is_cuda or state.code.numel() != 16):
            return False
        if state.nested and (state.state2.nested or state.state2.blocksize != 256
                             or state.state2.dtype != torch.float32
                             or not state.state2.absmax.is_cuda
                             or not state.state2.code.is_cuda or not state.offset.is_cuda):
            return False
    return (states[0].nested == states[1].nested
            and tuple(states[0].shape) == tuple(states[1].shape)
            and gate.device == up.device)


def register_mlp_cache(lm, limit_bytes=0):
    """Bound the cache and prioritize pairs which also fuse MLP matmuls."""
    clear_weight_cache()
    reserved = 0
    layers = list(getattr(lm.model, "layers", []))
    groups = [[getattr(getattr(layer, "mlp", None), name, None)
               for name in ("gate_proj", "up_proj")] for layer in layers]
    groups += [[getattr(getattr(layer, "mlp", None), "down_proj", None)] for layer in layers]
    groups += [[getattr(getattr(layer, "self_attn", None), name, None)]
               for layer in layers for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
    for group in groups:
        weights = [getattr(projection, "weight", None) for projection in group]
        states = [getattr(weight, "quant_state", None) for weight in weights]
        if any(state is None or weight.requires_grad or not weight.is_cuda
               or state.dtype != torch.float16 or state.quant_type != "nf4"
               for weight, state in zip(weights, states)):
            continue
        size = sum(state.shape[0] * state.shape[1] * 2 for state in states)
        if reserved + size <= limit_bytes:
            _CACHEABLE.update(weight.data_ptr() for weight in weights)
            reserved += size
    print(f"[ck] exact frozen decode cache: {len(_CACHEABLE)} weights, "
          f"{reserved / 2**20:.1f} MiB limit", flush=True)


def _cache_key(packed, state):
    # The task freezes these weights. Versions still invalidate a cache if a
    # caller explicitly updates either the packed values or their scales.
    tensors = [packed, state.absmax, state.code]
    if state.nested:
        tensors += [state.state2.absmax, state.state2.code, state.offset]
    return tuple((t.data_ptr(), t._version) for t in tensors)


@triton.jit
def _ck_nf4_kernel(PACKED, ABSMAX, CODE4, CODE8, SCALE2, OFFSET, OUT, N,
                    NESTED: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(PACKED + i // 2, i < N, other=0).to(tl.int32)
    code = tl.where(i % 2 == 0, packed >> 4, packed & 15)
    unit = tl.load(CODE4 + code)
    block = i // 64
    if NESTED:
        scale_code = tl.load(ABSMAX + block, i < N, other=0).to(tl.int32)
        unit_scale = tl.load(CODE8 + scale_code)
        second_scale = tl.load(SCALE2 + block // 256, i < N, other=0)
        scale = unit_scale * second_scale
        scale = scale + tl.load(OFFSET)
    else:
        scale = tl.load(ABSMAX + block, i < N, other=0)
    tl.store(OUT + i, unit * scale, i < N)


@triton.jit
def _ck_nf4_pair_kernel(GATE, UP, AG, AU, C4G, C4U, C8G, C8U,
                         S2G, S2U, OG, OU, OUT, N,
                         NESTED: tl.constexpr, BLOCK: tl.constexpr):
    part = tl.program_id(1)
    if part == 0:
        packed_ptr, absmax_ptr, code4_ptr = GATE, AG, C4G
        code8_ptr, scale2_ptr, offset_ptr = C8G, S2G, OG
    else:
        packed_ptr, absmax_ptr, code4_ptr = UP, AU, C4U
        code8_ptr, scale2_ptr, offset_ptr = C8U, S2U, OU
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(packed_ptr + i // 2, i < N, other=0).to(tl.int32)
    code = tl.where(i % 2 == 0, packed >> 4, packed & 15)
    unit = tl.load(code4_ptr + code)
    block = i // 64
    if NESTED:
        scale_code = tl.load(absmax_ptr + block, i < N, other=0).to(tl.int32)
        unit_scale = tl.load(code8_ptr + scale_code)
        second_scale = tl.load(scale2_ptr + block // 256, i < N, other=0)
        scale = unit_scale * second_scale
        scale = scale + tl.load(offset_ptr)
    else:
        scale = tl.load(absmax_ptr + block, i < N, other=0)
    tl.store(OUT + part * N + i, unit * scale, i < N)


def decode_nf4(packed, state, out=None):
    """Return the decoded matrix or None when the stock path is required."""
    if (state is None or not packed.is_cuda or packed.dtype != torch.uint8
            or not packed.is_contiguous() or state.quant_type != "nf4"
            or state.dtype != torch.float16 or state.blocksize != 64
            or len(state.shape) != 2 or not state.absmax.is_cuda
            or not isinstance(state.code, torch.Tensor)
            or not state.code.is_cuda or state.code.numel() != 16):
        return None
    key = None
    if out is None and packed.data_ptr() in _CACHEABLE:
        key = _cache_key(packed, state)
        cached = _DECODED.get(packed.data_ptr())
        if cached is not None and cached[0] == key:
            return cached[1].t() if packed.shape[0] == 1 else cached[1]
    if state.nested:
        s2 = state.state2
        if (s2 is None or s2.nested or s2.blocksize != 256
                or s2.dtype != torch.float32 or not s2.absmax.is_cuda
                or not s2.code.is_cuda or not isinstance(state.offset, torch.Tensor)
                or not state.offset.is_cuda):
            return None
        code8, scale2, offset = s2.code, s2.absmax, state.offset
    else:
        code8 = scale2 = offset = state.absmax
    if out is None:
        out = torch.empty(state.shape, dtype=state.dtype, device=packed.device)
    elif (not out.is_contiguous() or out.dtype != state.dtype
          or tuple(out.shape) != tuple(state.shape) or out.device != packed.device):
        return None
    _ck_nf4_kernel[(triton.cdiv(out.numel(), 1024),)](
        packed, state.absmax, state.code, code8, scale2, offset, out, out.numel(),
        NESTED=state.nested, BLOCK=1024, num_warps=4, enable_fp_fusion=False,
    )
    if key is not None:
        _DECODED[packed.data_ptr()] = (key, out)
    return out.t() if packed.shape[0] == 1 else out


def patch_nf4_decode():
    import bitsandbytes.functional as functional
    patch_nf4_matmul()
    global _ORIGINAL_DEQUANT
    if getattr(functional.dequantize_4bit, "_ck_is_ours", False):
        return
    original = functional.dequantize_4bit
    _ORIGINAL_DEQUANT = original

    def dequantize(A, quant_state=None, absmax=None, out=None,
                   blocksize=None, quant_type="fp4"):
        result = decode_nf4(A, quant_state, out=out)
        if result is not None:
            return result
        return original(A, quant_state=quant_state, absmax=absmax, out=out,
                        blocksize=blocksize, quant_type=quant_type)

    dequantize._ck_is_ours = True
    functional.dequantize_4bit = dequantize


def patch_nf4_matmul():
    """Use our fused decode for the frozen fp16 forward as well as backward."""
    from bitsandbytes.autograd._functions import MatMul4Bit
    import torch.nn.functional as functional
    global _ORIGINAL_MATMUL_FORWARD, _ORIGINAL_MATMUL_BACKWARD

    original = MatMul4Bit.forward
    if getattr(original, "_ck_is_ours", False):
        return
    _ORIGINAL_MATMUL_FORWARD = original
    original_backward = MatMul4Bit.backward
    _ORIGINAL_MATMUL_BACKWARD = original_backward

    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        if (not A.is_cuda or A.dtype != torch.float16 or A.numel() == 0 or B.numel() <= 1
                or not B.is_contiguous() or quant_state is None
                or (bias is not None and (bias.requires_grad or bias.dtype != A.dtype))):
            return original(ctx, A, B, out, bias, quant_state)
        packed = B.view(-1, 1)
        decoded = decode_nf4(packed, quant_state)
        if decoded is None:
            return original(ctx, A, B, out, bias, quant_state)
        from .ck_padding import projection_layout, gather_rows, scatter_rows
        layout = projection_layout(A) if not B.requires_grad else None
        if layout is None:
            output = functional.linear(A, decoded, bias)
        else:
            output = scatter_rows(functional.linear(gather_rows(A, layout), decoded, bias), layout)
            ctx._ck_projection_layout = layout
        if out is not None:
            out.copy_(output)
            output = out
        # Preserve the state expected by bitsandbytes' unchanged backward.
        ctx.is_empty = False
        ctx.state = quant_state
        ctx.dtype_A, ctx.dtype_B = A.dtype, B.dtype
        ctx.dtype_bias = None if bias is None else bias.dtype
        ctx.tensors = (None, packed) if any(ctx.needs_input_grad[:2]) else (None, None)
        return output

    def backward(ctx, grad_output):
        layout = getattr(ctx, "_ck_projection_layout", None)
        if layout is None:
            return original_backward(ctx, grad_output)
        from .ck_padding import gather_rows, scatter_rows
        grad_A = None
        if ctx.needs_input_grad[0]:
            packed = ctx.tensors[1]
            decoded = decode_nf4(packed, ctx.state)
            grad = gather_rows(grad_output.to(ctx.dtype_A), layout)
            grad_A = scatter_rows(torch.mm(grad, decoded), layout)
        return grad_A, None, None, None, None

    forward._ck_is_ours = True
    MatMul4Bit.forward = staticmethod(forward)
    MatMul4Bit.backward = staticmethod(backward)
