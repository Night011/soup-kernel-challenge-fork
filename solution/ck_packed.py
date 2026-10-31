"""Keep valid tokens packed through the transformer, within one original batch.

The LoRA dropout boundary alone expands to the original shape to preserve
native PyTorch RNG. Attention uses the original per-sample sequence offsets.
"""

from contextlib import contextmanager

import torch

_LORA_LAYOUT = None


def current_lora_layout():
    return _LORA_LAYOUT


@contextmanager
def packed_lora_layout(layout):
    global _LORA_LAYOUT
    old = _LORA_LAYOUT
    _LORA_LAYOUT = layout
    try:
        yield
    finally:
        _LORA_LAYOUT = old


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, full, layout):
        from .ck_padding import gather_rows
        ctx.layout = layout
        return gather_rows(full, layout)

    @staticmethod
    def backward(ctx, grad):
        from .ck_padding import scatter_rows
        return scatter_rows(grad, ctx.layout), None


class _Expand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, layout):
        from .ck_padding import scatter_rows
        ctx.layout = layout
        return scatter_rows(packed, layout)

    @staticmethod
    def backward(ctx, grad):
        from .ck_padding import gather_rows
        return gather_rows(grad, ctx.layout), None


def expand_for_dropout(x, layout):
    return _Expand.apply(x, layout)


def packed_transformer(model, input_ids, attention_mask, position_ids,
                       inputs_embeds, metadata, past_key_values=None,
                       output_attentions=False, output_hidden_states=False):
    layout = metadata.get("attention_layout") if metadata is not None else None
    if (layout is None or not model.training or not torch.is_grad_enabled()
            or past_key_values is not None or output_attentions or output_hidden_states
            or getattr(model, "gradient_checkpointing", False)
            or model.config._attn_implementation != "sdpa"
            or attention_mask is None or tuple(attention_mask.shape) != layout["shape"]
            or any(layer.self_attn.attention_dropout != 0.0
                   or (layer.self_attn.sliding_window is not None
                       and layer.self_attn.sliding_window < attention_mask.shape[1])
                   for layer in model.layers)):
        return None
    from transformers.models.qwen2 import modeling_qwen2 as qm
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from .ck_attention import _PackQKV, _NativeVarlen

    full = model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
    batch, sequence, hidden = full.shape
    rows, inverse = layout["rows"], layout["inverse"]
    row_layout = (rows, (batch, sequence), inverse)
    if position_ids is None:
        position_ids = torch.arange(sequence, device=full.device).unsqueeze(0)
    cos, sin = model.rotary_emb(full, position_ids)
    cos = cos.expand(batch, -1, -1).reshape(-1, cos.shape[-1]).index_select(0, rows).unsqueeze(0)
    sin = sin.expand(batch, -1, -1).reshape(-1, sin.shape[-1]).index_select(0, rows).unsqueeze(0)
    x = _Gather.apply(full, row_layout)
    count = rows.numel()
    identity = layout["packed_rows"]
    packed_inverse = layout["packed_inverse"]
    with packed_lora_layout(row_layout):
        for layer in model.layers:
            residual = x
            normalized = layer.input_layernorm(x)
            attn = layer.self_attn
            dim = attn.head_dim
            q = attn.q_proj(normalized).view(1, count, -1, dim).transpose(1, 2)
            k = attn.k_proj(normalized).view(1, count, -1, dim).transpose(1, 2)
            v = attn.v_proj(normalized).view(1, count, -1, dim).transpose(1, 2)
            q, k = qm.apply_rotary_pos_emb(q, k, cos, sin)
            q, k, v = _PackQKV.apply(q, k, v, identity, packed_inverse)
            out = _NativeVarlen.apply(q, k, v, layout["offsets"],
                                      layout["max_length"], attn.scaling)
            x = residual + attn.o_proj(out.view(count, hidden))
            residual = x
            x = residual + layer.mlp(layer.post_attention_layernorm(x))
        x = model.norm(x)
    return BaseModelOutputWithPast(last_hidden_state=_Expand.apply(x, row_layout))
