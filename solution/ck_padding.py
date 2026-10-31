"""Batch-local removal of unused MLP rows; attention and RNG shapes stay full.

Only frozen, token-wise MLPs are compacted. Padding queries needed by shifted
labels disable this optimization: a fully masked SDPA query can see unmasked
keys, so keeping only its last-layer MLP would not preserve its history.
"""

from contextlib import contextmanager

import torch

_LAYOUT = None


def mlp_layout(attention_mask, labels):
    if (not isinstance(attention_mask, torch.Tensor)
            or not isinstance(labels, torch.Tensor)
            or attention_mask.ndim != 2 or labels.shape != attention_mask.shape
            or not attention_mask.is_cuda or labels.device != attention_mask.device):
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
    return rows, tuple(active.shape)


@contextmanager
def compact_mlp_rows(attention_mask, labels):
    global _LAYOUT
    previous = _LAYOUT
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
    rows, (batch, sequence) = layout
    full = hidden.new_zeros((batch * sequence, hidden.shape[-1]))
    return full.index_copy_(0, rows, hidden).view(batch, sequence, -1)
