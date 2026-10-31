"""Blockwise int8 storage for the frozen MLP's saved backward activations.

Only the saved gate/up tensors are compressed. The forward result is computed
from the original inputs. Each 256-element block gets its own fp32 scale.
"""

import torch
import triton
import triton.language as tl

PACK_BLOCK = 256


@triton.jit
def _ck_pack_activations_kernel(GATE, UP, QGATE, QUP, SGATE, SUP, N, COLS, STRIDE,
                                 BLOCK: tl.constexpr):
    block = tl.program_id(0)
    i = block * BLOCK + tl.arange(0, BLOCK)
    source = (i // COLS) * STRIDE + i % COLS
    gate = tl.load(GATE + source, i < N, other=0.0).to(tl.float32)
    up = tl.load(UP + source, i < N, other=0.0).to(tl.float32)
    sg = tl.maximum(tl.max(tl.abs(gate), axis=0) / 127.0, 1.0e-20)
    su = tl.maximum(tl.max(tl.abs(up), axis=0) / 127.0, 1.0e-20)
    qg = tl.minimum(tl.floor(tl.abs(gate) / sg + 0.5), 127.0)
    qu = tl.minimum(tl.floor(tl.abs(up) / su + 0.5), 127.0)
    qg *= tl.where(gate < 0, -1.0, 1.0)
    qu *= tl.where(up < 0, -1.0, 1.0)
    tl.store(QGATE + i, qg.to(tl.int8), i < N)
    tl.store(QUP + i, qu.to(tl.int8), i < N)
    tl.store(SGATE + block, sg)
    tl.store(SUP + block, su)


def pack_activations(gate, up):
    qgate = torch.empty(gate.shape, dtype=torch.int8, device=gate.device)
    qup = torch.empty(up.shape, dtype=torch.int8, device=up.device)
    blocks = triton.cdiv(gate.numel(), PACK_BLOCK)
    scales_gate = torch.empty(blocks, dtype=torch.float32, device=gate.device)
    scales_up = torch.empty_like(scales_gate)
    _ck_pack_activations_kernel[(blocks,)](
        gate, up, qgate, qup, scales_gate, scales_up, gate.numel(),
        gate.shape[-1], gate.reshape(-1, gate.shape[-1]).stride(0),
        BLOCK=PACK_BLOCK, num_warps=4,
    )
    return qgate, qup, scales_gate, scales_up
