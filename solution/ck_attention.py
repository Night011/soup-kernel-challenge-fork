"""Compact only padding in attention, using PyTorch's existing SDPA backend.

Our kernels pack Q/K/V (including GQA expansion and the fp16 boundary) and
restore gradients. Attention itself is aten's existing CUTLASS implementation,
not claimed as our kernel. Projections and LoRA dropout retain their full
shapes and order. No external attention package is used.
"""

import types

import torch
import triton
import triton.language as tl


def attention_layout_cpu(mask, labels):
    if mask.is_cuda or labels.is_cuda or mask.ndim != 2 or mask.shape != labels.shape:
        return None
    active = mask.bool()
    if bool(((labels[:, 1:] != -100) & ~active[:, :-1]).any()):
        return None
    lengths = active.sum(1).to(torch.int32)
    if bool((lengths == 0).any()) or bool(active.all()):
        return None
    rows = active.flatten().nonzero().flatten()
    inverse = torch.full((active.numel(),), -1, dtype=torch.int32)
    inverse[rows] = torch.arange(rows.numel(), dtype=torch.int32)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32)))
    return {"rows": rows, "inverse": inverse, "offsets": offsets,
            "shape": tuple(mask.shape), "max_length": int(lengths.max())}


@triton.jit
def _ck_attention_pack_kernel(Q, K, V, ROWS, PQ, PK, PV, N, S,
                               QB, QH, QS, KB, KH, KS, VB, VH, VS,
                               HQ: tl.constexpr, HK: tl.constexpr,
                               D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed_row, head, dimension = i // (HQ * D), (i // D) % HQ, i % D
    full_row = tl.load(ROWS + packed_row, i < N, other=0)
    batch, sequence = full_row // S, full_row % S
    kv_head = head // (HQ // HK)
    q = tl.load(Q + batch * QB + head * QH + sequence * QS + dimension, i < N, other=0)
    k = tl.load(K + batch * KB + kv_head * KH + sequence * KS + dimension, i < N, other=0)
    v = tl.load(V + batch * VB + kv_head * VH + sequence * VS + dimension, i < N, other=0)
    tl.store(PQ + i, q, i < N)
    tl.store(PK + i, k, i < N)
    tl.store(PV + i, v, i < N)


@triton.jit
def _ck_attention_unpack_grad_kernel(GQ, GK, GV, INVERSE, DQ, DK, DV, S,
                                      HQ: tl.constexpr, HK: tl.constexpr,
                                      D: tl.constexpr, BR: tl.constexpr):
    combined = tl.program_id(0)
    batch, head = combined // (HQ + HK), combined % (HQ + HK)
    sequence = tl.program_id(1) * BR + tl.arange(0, BR)
    dimension = tl.arange(0, D)
    packed_row = tl.load(INVERSE + batch * S + sequence, sequence < S, other=-1)
    mask = (sequence[:, None] < S) & (packed_row[:, None] >= 0)
    if head < HQ:
        source = (tl.maximum(packed_row[:, None], 0) * HQ + head) * D + dimension[None, :]
        value = tl.load(GQ + source, mask, other=0).to(tl.float32)
        destination = ((batch * HQ + head) * S + sequence[:, None]) * D + dimension[None, :]
        tl.store(DQ + destination, value, sequence[:, None] < S)
    else:
        head -= HQ
        gk = tl.full((BR, D), 0, tl.float32)
        gv = tl.full((BR, D), 0, tl.float32)
        for group in range(HQ // HK):
            source = (tl.maximum(packed_row[:, None], 0) * HQ + head * (HQ // HK) + group) * D + dimension[None, :]
            gk += tl.load(GK + source, mask, other=0).to(tl.float32)
            gv += tl.load(GV + source, mask, other=0).to(tl.float32)
        destination = ((batch * HK + head) * S + sequence[:, None]) * D + dimension[None, :]
        tl.store(DK + destination, gk, sequence[:, None] < S)
        tl.store(DV + destination, gv, sequence[:, None] < S)


class _PackQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, rows, inverse):
        b, hq, s, d = q.shape
        hk = k.shape[1]
        shape = (1, rows.numel(), hq, d)
        pq = torch.empty(shape, device=q.device, dtype=torch.float16)
        pk, pv = torch.empty_like(pq), torch.empty_like(pq)
        _ck_attention_pack_kernel[(triton.cdiv(pq.numel(), 1024),)](
            q, k, v, rows, pq, pk, pv, pq.numel(), s,
            *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
            HQ=hq, HK=hk, D=d, BLOCK=1024, num_warps=4)
        ctx.save_for_backward(inverse)
        ctx.shapes, ctx.dtypes = (q.shape, k.shape, v.shape), (q.dtype, k.dtype, v.dtype)
        return pq, pk, pv

    @staticmethod
    def backward(ctx, gq, gk, gv):
        inverse, = ctx.saved_tensors
        gradients = [torch.empty(shape, device=gq.device, dtype=dtype)
                     for shape, dtype in zip(ctx.shapes, ctx.dtypes)]
        b, hq, s, d = ctx.shapes[0]
        hk = ctx.shapes[1][1]
        _ck_attention_unpack_grad_kernel[(b * (hq + hk), triton.cdiv(s, 8))](
            gq.contiguous(), gk.contiguous(), gv.contiguous(), inverse, *gradients,
            s, HQ=hq, HK=hk, D=d, BR=8, num_warps=4)
        return *gradients, None, None


class _NativeVarlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, offsets, maximum, scale):
        with torch.autocast("cuda", enabled=False):
            result = torch.ops.aten._efficient_attention_forward(
                q, k, v, None, offsets, offsets, maximum, maximum,
                0.0, 1, True, scale=scale)
        out, lse, seed, offset, max_q, max_k = result
        ctx.save_for_backward(q, k, v, offsets, out, lse, seed, offset)
        ctx.maximums, ctx.scale = (max_q, max_k), scale
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k, v, offsets, out, lse, seed, offset = ctx.saved_tensors
        with torch.autocast("cuda", enabled=False):
            dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
                grad.contiguous(), q, k, v, None, out, offsets, offsets,
                *ctx.maximums, lse, 0.0, seed, offset, 1, False, scale=ctx.scale)
        return dq, dk, dv, None, None, None


class _RestoreOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, layout):
        from .ck_padding import scatter_rows
        ctx.layout, ctx.shape = layout, packed.shape
        return scatter_rows(packed.reshape(packed.shape[1], -1),
                            (layout["rows"], layout["shape"], layout["inverse"]))

    @staticmethod
    def backward(ctx, grad):
        from .ck_padding import gather_rows
        return gather_rows(grad, (ctx.layout["rows"],)).view(ctx.shape), None


def packed_attention(q, k, v, layout, scale):
    q, k, v = _PackQKV.apply(q, k, v, layout["rows"], layout["inverse"])
    packed = _NativeVarlen.apply(q, k, v, layout["offsets"], layout["max_length"], scale)
    return _RestoreOutput.apply(packed, layout)


def _attention_forward(self, hidden_states, position_embeddings, attention_mask,
                        past_key_values=None, **kwargs):
    from .ck_padding import current_attention_layout
    layout = current_attention_layout()
    if (layout is None or tuple(hidden_states.shape[:2]) != layout["shape"]
            or past_key_values is not None or not self.training
            or self.attention_dropout != 0.0
            or self.config._attn_implementation != "sdpa"
            or kwargs.get("output_attentions", False)
            or (self.sliding_window is not None and self.sliding_window < hidden_states.shape[1])):
        return self._ck_attention_original(hidden_states, position_embeddings, attention_mask,
                                            past_key_values=past_key_values, **kwargs)
    from transformers.models.qwen2 import modeling_qwen2 as qm
    shape = (*hidden_states.shape[:-1], -1, self.head_dim)
    q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
    k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
    v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    q, k = qm.apply_rotary_pos_emb(q, k, *position_embeddings)
    out = packed_attention(q, k, v, layout, self.scaling)
    return self.o_proj(out), None


def install_attention(module):
    if getattr(module, "head_dim", 0) not in (64, 128):
        return False
    module._ck_attention_original = module.forward
    module.forward = types.MethodType(_attention_forward, module)
    return True


def remove_attention(module):
    if hasattr(module, "_ck_attention_original"):
        module.forward = module._ck_attention_original
        del module._ck_attention_original
