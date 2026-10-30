"""Integration of our Triton kernels into the Soup training loop.

Hooks, all guarded so anything but the exact Qwen2 + CUDA + fp16 SFT flow of
the challenge task behaves exactly like stock code:

* `patch_trainer_prepare_inputs` — K4: drop all-padding columns at the edges of
  each batch (leading prompt-pad and trailing completion-pad). Real tokens,
  per-column position ids, labels and metrics keep their values; only pad
  columns leave the GPU work. The window of original position ids is passed
  explicitly so RoPE sees the same ids as the stock run.
* `patch_trl_compute_loss` — while our head-loss hook owns the forward, run the
  plain Trainer.compute_loss (labels/num_items path) and replicate TRL metric
  bookkeeping verbatim (entropy, num_tokens, mean_token_accuracy) plus Soup's
  rewind token-loss contract (#1693).
* `engage(model)` / `disengage()` — swap the causal-LM instance forward:
  full transformer body, then our chunked Triton head+loss instead of
  lm_head + full logits.
* `wrap_setup_transformers` — installs the hook where Soup builds its final
  PEFT model (SFTTrainerWrapper._setup_transformers, documented hook point).
"""

import types

import torch

_ACTIVE = None  # CKHeadHook while a chunked run is in flight
_TRL_LOSS = None


# ---------------------------------------------------------------------------
# K4 — batch edge trim (batch-internal padding removal, explicitly allowed)
# ---------------------------------------------------------------------------

def _trim_inputs(inputs):
    try:
        return _trim_inputs_inner(inputs)
    except Exception:
        import traceback
        traceback.print_exc()
        return inputs


def _trim_inputs_inner(inputs):
    if not isinstance(inputs, dict) or len(inputs) == 0:
        return inputs
    am = inputs.get("attention_mask")
    if am is None or am.dim() != 2 or not am.is_cuda:
        return inputs
    ids = inputs.get("input_ids")
    labels = inputs.get("labels")
    if ids is None or labels is None:
        return inputs
    if ids.shape != am.shape or labels.shape != am.shape:
        return inputs
    B, N = am.shape
    if N < 2:
        return inputs
    used = am.any(dim=0)
    if used.all():
        return inputs
    used8 = used.to(torch.uint8)
    left = int(used8.argmax().item())              # first non-empty column
    right = N - int(used8.flip(0).argmax().item())  # after last non-empty
    if left >= right:
        return inputs
    if not bool(used[left:right].all().item()):
        return inputs  # interior gaps: do not touch anything
    out = dict(inputs)
    out["input_ids"] = ids[:, left:right]
    out["attention_mask"] = am[:, left:right]
    out["labels"] = labels[:, left:right]
    pos = inputs.get("position_ids")
    if pos is None:
        pos = torch.arange(N, device=ids.device).unsqueeze(0).expand(B, -1)
    if pos.dim() == 2 and pos.shape == am.shape:
        out["position_ids"] = pos[:, left:right]
    else:
        out["position_ids"] = (
            torch.arange(left, right, device=ids.device)
            .unsqueeze(0).expand(B, -1)
        )
    return out


def patch_trainer_prepare_inputs():
    import transformers

    if getattr(transformers.Trainer._prepare_inputs, "_ck_is_ours", False):
        return
    orig = transformers.Trainer._prepare_inputs

    def prepared(self, inputs):
        out = orig(self, inputs)
        try:
            return _trim_inputs(out)
        except Exception:
            return out

    prepared._ck_is_ours = True
    transformers.Trainer._prepare_inputs = prepared
    print("[ck] patch prepare_inputs ok", flush=True)
    return prepared


# ---------------------------------------------------------------------------
# Causal-LM instance forward with our chunked Triton head + loss
# ---------------------------------------------------------------------------

def _causal_lm_of(model):
    lm = model
    getter = getattr(lm, "get_base_model", None)
    if getter is not None:
        try:
            lm = getter()
        except Exception:
            return None
    if lm is not None and hasattr(lm, "base_model") and hasattr(lm.base_model, "model"):
        lm = lm.base_model.model
    if hasattr(lm, "lm_head") and hasattr(lm, "model"):
        return lm
    return None


def _log_stock_metrics(trainer, logits, attention_mask, labels):
    """Full-logits metric replication (rare fallback path when disengaged)."""
    try:
        from trl.trainer.utils import entropy_from_logits

        if trainer is None or logits is None:
            return
        mode = "train" if getattr(trainer.model, "training", False) else "eval"
        metrics = trainer._metrics[mode]
        with torch.no_grad():
            per_token = entropy_from_logits(logits)
            mask = (
                attention_mask.to(torch.float32) if attention_mask is not None
                else torch.ones(per_token.shape, device=per_token.device)
            )
            ent = (per_token * mask).sum() / mask.sum()
            metrics["entropy"].append(float(ent.detach().item()))
            shift_labels = labels[:, 1:]
            predictions = logits[:, :-1].argmax(dim=-1)
            smask = shift_labels != -100
            correct = (predictions == shift_labels) & smask
            n = int(smask.sum().item())
            if n > 0:
                acc = float(correct.sum().item()) / n
            else:
                acc = 0.0
            metrics["mean_token_accuracy"].append(acc)
    except Exception:
        pass


def _ck_causal_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    logits_to_keep=0,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=True,
    num_items_in_batch=None,
    **kwargs,
):
    from . import ck_head
    from transformers.modeling_outputs import CausalLMOutputWithPast

    if labels is None or not torch.is_grad_enabled():
        return self._ck_orig_forward(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
            logits_to_keep=logits_to_keep, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
        )
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=False if use_cache is None else use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    hook = _ACTIVE
    try:
        loss, step = ck_head.ck_chunked_lm_loss(
            hidden, labels, self.lm_head.weight, attention_mask,
            num_items_in_batch, hook,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        print("[ck] chunked head failed: falling back to stock logits path",
              flush=True)
        loss, step = None, None
    logits = None
    if loss is None:  # our guards said no: stock logits path, stock metrics
        logits = self.lm_head(hidden)
        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.vocab_size,
            num_items_in_batch=num_items_in_batch,
        )
        _log_stock_metrics(hook.trainer, logits, attention_mask, labels)
        step = None
    out = CausalLMOutputWithPast(
        loss=loss,
        # the stock fallback keeps real logits: Soup's rewind recorder prefers
        # them over the chunked step, exactly like an unpatched run
        logits=logits,
        past_key_values=getattr(outputs, "past_key_values", None),
        hidden_states=None,
        attentions=None,
    )
    hook.step = step
    return out


def engage(model):
    """Install the instance forward + hook for the challenge flow, else None."""
    global _ACTIVE
    disengage()
    print("[ck] engage: begin", flush=True)
    if not torch.cuda.is_available():
        print("[ck] engage: skip (no cuda)", flush=True)
        return None
    try:
        from . import ck_head

        lm = _causal_lm_of(model)
        if lm is None:
            print("[ck] engage: skip (no causal lm)", flush=True)
            return None
        cfg = lm.config
        if getattr(cfg, "model_type", "") != "qwen2":
            print(f"[ck] engage: skip (model_type={getattr(cfg, 'model_type', None)})"
                  , flush=True)
            return None
        w = lm.lm_head.weight
        if not (w.is_cuda and w.dtype == torch.float32 and not w.requires_grad):
            print(f"[ck] engage: skip (head {w.dtype}, cuda={w.is_cuda}, "
                  f"requires_grad={w.requires_grad})", flush=True)
            return None
        if cfg.vocab_size % 128 != 0 or cfg.hidden_size % 64 != 0:
            print("[ck] engage: skip (V%128 or K%64)", flush=True)
            return None
        hook = ck_head.CKHeadHook()
        hook.causal_lm = lm
        hook.orig_forward = lm.forward
        lm.forward = types.MethodType(_ck_causal_forward, lm)
        _ACTIVE = hook
        print("[ck] engage: OK", flush=True)
        return hook
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def disengage():
    global _ACTIVE
    hook = _ACTIVE
    _ACTIVE = None
    if hook is None:
        return
    try:
        if hook.causal_lm is not None and hook.orig_forward is not None:
            hook.causal_lm.forward = hook.orig_forward
    except Exception:
        pass


# ---------------------------------------------------------------------------
# TRL compute_loss replacement while engaged
# ---------------------------------------------------------------------------

def _ck_sft_compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
    from transformers import Trainer

    hook = _ACTIVE
    mode = "train" if self.model.training else "eval"
    engaged = bool(
        hook is not None
        and hook.causal_lm is not None
        and mode == "train"
        and inputs.get("labels") is not None
        and "shift_labels" not in inputs
    )
    if not engaged:
        return _TRL_LOSS(self, model, inputs, return_outputs=return_outputs,
                         num_items_in_batch=num_items_in_batch)
    hook.trainer = self
    inputs.pop("_prediction_loss_only", None)
    inputs["use_cache"] = False
    # same call Soup's own ChunkedLossTrainer makes (#1693)
    loss, outputs = Trainer.compute_loss(
        self, model, inputs, return_outputs=True,
        num_items_in_batch=num_items_in_batch,
    )
    # publish this micro-batch's per-token losses for Soup's rewind recorder;
    # entropy/accuracy are appended by CKHeadHook.on_backward_done (they are
    # produced by the gradient sweep, i.e. after this call)
    step = hook.take_step()
    step_attr = "_soup_chunked_step"
    try:
        from soup_cli.utils.stream_loss_chunk import STEP_ATTR
        step_attr = STEP_ATTR
    except Exception:
        pass
    setattr(self, step_attr, step)
    if "attention_mask" in inputs:
        num_tokens_in_batch = self.accelerator.gather_for_metrics(
            inputs["attention_mask"].sum()
        ).sum().item()
    else:
        local = torch.tensor(
            inputs["position_ids"].size(1),
            device=inputs["position_ids"].device,
        )
        num_tokens_in_batch = self.accelerator.gather_for_metrics(local).sum().item()
    self._total_train_tokens += num_tokens_in_batch
    self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
    if return_outputs:
        return (loss, outputs)
    return loss


def patch_trl_compute_loss():
    import trl

    if getattr(trl.SFTTrainer.compute_loss, "_ck_is_ours", False):
        return
    global _TRL_LOSS
    _TRL_LOSS = trl.SFTTrainer.compute_loss
    patched = _ck_sft_compute_loss
    patched._ck_is_ours = True
    trl.SFTTrainer.compute_loss = patched
    print("[ck] patch trl compute_loss ok", flush=True)


# ---------------------------------------------------------------------------
# Soup hook point
# ---------------------------------------------------------------------------

def wrap_setup_transformers():
    import soup_cli.trainer.sft as sft_mod

    cls = sft_mod.SFTTrainerWrapper
    if getattr(cls._setup_transformers, "_ck_is_ours", False):
        return
    orig = cls._setup_transformers

    def hooked(self, cfg, tcfg):
        out = orig(self, cfg, tcfg)
        try:
            engage(self.model)
        except Exception:
            import traceback
            traceback.print_exc()
        return out

    hooked._ck_is_ours = True
    cls._setup_transformers = hooked
    print("[ck] patch setup_transformers ok", flush=True)


def apply_all():
    """Ordered wiring; each patch is idempotent and guarded."""
    patch_trainer_prepare_inputs()
    patch_trl_compute_loss()
    wrap_setup_transformers()
