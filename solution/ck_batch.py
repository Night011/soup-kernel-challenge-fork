"""Prepare small batch-derived indices on CPU before accelerator transfer."""

import torch


def prepare_metadata(batch):
    from .integrate import _trim_inputs_inner
    from .ck_padding import mlp_layout

    if not isinstance(batch, dict):
        return batch
    am, labels = batch.get("attention_mask"), batch.get("labels")
    if (not isinstance(am, torch.Tensor) or not isinstance(labels, torch.Tensor)
            or am.is_cuda or labels.is_cuda or am.ndim != 2 or am.shape != labels.shape):
        return batch
    batch = _trim_inputs_inner(batch, allow_cpu=True)
    am, labels = batch["attention_mask"], batch["labels"]
    targets = torch.full_like(labels, -100)
    targets[:, :-1] = labels[:, 1:]
    labelled = targets.flatten() != -100
    labelled_rows = labelled.nonzero().flatten()
    metric_only = (am.bool().flatten() & ~labelled).nonzero().flatten()
    result = dict(batch)
    result["_ck_batch_meta"] = {
        "shape": tuple(am.shape),
        "num_tokens": int(am.sum().item()),
        "head_indices": torch.cat((labelled_rows, metric_only)),
        "num_labelled": labelled_rows.numel(),
        "mlp_layout": mlp_layout(am, labels, allow_cpu=True),
    }
    if result["_ck_batch_meta"]["mlp_layout"] is not None:
        from .ck_attention import attention_layout_cpu
        result["_ck_batch_meta"]["attention_layout"] = attention_layout_cpu(am, labels)
    return result


def patch_collator_metadata():
    try:
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    except (ImportError, AttributeError):
        return
    original = DataCollatorForLanguageModeling.torch_call
    if getattr(original, "_ck_is_ours", False):
        return

    def collate(self, examples):
        from . import integrate
        batch = original(self, examples)
        return prepare_metadata(batch) if integrate._ACTIVE is not None else batch

    collate._ck_is_ours = True
    DataCollatorForLanguageModeling.torch_call = collate
