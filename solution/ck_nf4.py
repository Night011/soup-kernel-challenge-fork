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
_CACHEABLE = set()
_DECODED = {}
_PAIRED = {}


def clear_weight_cache():
    _CACHEABLE.clear()
    _DECODED.clear()
    _PAIRED.clear()


def paired_mlp_weights(gate_weight, up_weight):
    """Concatenate exact gate/up matrices once, replacing their separate cache."""
    gp, up = gate_weight.data_ptr(), up_weight.data_ptr()
    if gp not in _CACHEABLE or up not in _CACHEABLE:
        return None
    gs, us = gate_weight.quant_state, up_weight.quant_state
    version = (_cache_key(gate_weight, gs), _cache_key(up_weight, us))
    cached = _PAIRED.get((gp, up))
    if cached is not None and cached[0] == version:
        return cached[1]
    g = decode_nf4(gate_weight, gs)
    u = decode_nf4(up_weight, us)
    if g is None or u is None:
        return None
    combined = torch.cat((g, u), dim=0)
    _DECODED.pop(gp, None)
    _DECODED.pop(up, None)
    _PAIRED[(gp, up)] = (version, combined)
    return combined


def register_mlp_cache(lm, limit_bytes=2688 * 2**20):
    """Bound exact frozen projection caches to 2.625 GiB of GPU RAM."""
    clear_weight_cache()
    reserved = 0
    for layer in getattr(lm.model, "layers", []):
        projections = [getattr(getattr(layer, "mlp", None), name, None)
                       for name in ("gate_proj", "up_proj", "down_proj")]
        projections += [getattr(getattr(layer, "self_attn", None), name, None)
                        for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
        for projection in projections:
            weight = getattr(projection, "weight", None)
            state = getattr(weight, "quant_state", None)
            if (state is None or weight.requires_grad or not weight.is_cuda
                    or state.dtype != torch.float16 or state.quant_type != "nf4"):
                continue
            size = state.shape[0] * state.shape[1] * 2
            if reserved + size <= limit_bytes:
                _CACHEABLE.add(weight.data_ptr())
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
    global _ORIGINAL_MATMUL_FORWARD

    original = MatMul4Bit.forward
    if getattr(original, "_ck_is_ours", False):
        return
    _ORIGINAL_MATMUL_FORWARD = original

    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        if (not A.is_cuda or A.dtype != torch.float16 or A.numel() == 0 or B.numel() <= 1
                or not B.is_contiguous() or quant_state is None
                or (bias is not None and (bias.requires_grad or bias.dtype != A.dtype))):
            return original(ctx, A, B, out, bias, quant_state)
        packed = B.view(-1, 1)
        decoded = decode_nf4(packed, quant_state)
        if decoded is None:
            return original(ctx, A, B, out, bias, quant_state)
        output = functional.linear(A, decoded, bias)
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

    forward._ck_is_ours = True
    MatMul4Bit.forward = staticmethod(forward)
