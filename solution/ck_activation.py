"""Blockwise int8 storage for the frozen MLP's saved backward activations.

Only the saved gate/up tensors are compressed. The forward result is computed
from the original inputs. Each 256-element block gets its own fp32 scale.
"""

import torch
import triton
import triton.language as tl

PACK_BLOCK = 256


@triton.jit
def _ck_pack_activations_kernel(GATE, UP, QGATE, QUP, SGATE, SUP, N,
                                 BLOCK: tl.constexpr):
    block = tl.program_id(0)
    i = block * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(GATE + i, i < N, other=0.0).to(tl.float32)
    up = tl.load(UP + i, i < N, other=0.0).to(tl.float32)
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
    qgate = torch.empty_like(gate, dtype=torch.int8)
    qup = torch.empty_like(up, dtype=torch.int8)
    blocks = triton.cdiv(gate.numel(), PACK_BLOCK)
    scales_gate = torch.empty(blocks, dtype=torch.float32, device=gate.device)
    scales_up = torch.empty_like(scales_gate)
    _ck_pack_activations_kernel[(blocks,)](
        gate, up, qgate, qup, scales_gate, scales_up, gate.numel(),
        BLOCK=PACK_BLOCK, num_warps=4,
    )
    return qgate, qup, scales_gate, scales_up
