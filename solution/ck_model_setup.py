"""Stage the frozen tied head's ordinary PEFT fp16-to-fp32 cast through CPU.

The final parameter, values, dtype and CUDA device are the same. Original
PEFT preparation still owns freezing, cache clearing and checkpoint hooks.
"""

import functools

import torch


class _HeadRestoreFailure(RuntimeError):
    pass


def _upcast_tied_head(model):
    """Return whether a shared CUDA fp16 Parameter was staged successfully."""
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    weight = getattr(head, "weight", None)
    if (weight is None or getattr(embedding, "weight", None) is not weight
            or not isinstance(weight, torch.nn.Parameter)
            or not weight.is_cuda or weight.dtype != torch.float16
            or not weight.is_contiguous()):
        return False
    device = weight.device
    host = weight.detach().cpu()
    try:
        # No local reference retains the former CUDA half storage. Upcast on
        # CPU before copying the fp32 result, so both GPU sizes never coexist.
        weight.data = host
        # The old CUDA half storage is now unused. Release its cached pages
        # before allocating the larger master; allocator settings stay intact.
        torch.cuda.empty_cache()
        weight.data = host.float().to(device)
    except Exception:
        try:
            weight.data = host.to(device)
        except Exception as restore_error:
            raise _HeadRestoreFailure("tied head CUDA storage could not be restored") from restore_error
        raise
    return True


def _matches_challenge(model):
    cfg = getattr(model, "config", None)
    return bool(
        getattr(model, "is_loaded_in_4bit", False)
        and getattr(cfg, "model_type", None) == "qwen2"
        and getattr(cfg, "_name_or_path", None) == "Qwen/Qwen2.5-1.5B-Instruct"
        and getattr(cfg, "hidden_size", None) == 1536
        and getattr(cfg, "vocab_size", None) == 151936
    )


def _wrap_prepare(original):
    @functools.wraps(original)
    def prepare(model, *args, **kwargs):
        if _matches_challenge(model):
            try:
                if _upcast_tied_head(model):
                    print("[ck] tied head upcast staged through CPU before unchanged PEFT preparation", flush=True)
            except _HeadRestoreFailure:
                # A failed rollback must remain a visible training failure.
                raise
            except Exception as exc:
                print(f"[ck] tied head upcast fallback after restoring fp16 CUDA storage: "
                      f"{type(exc).__name__}", flush=True)
        return original(model, *args, **kwargs)

    prepare._ck_is_ours = True
    return prepare


def patch_kbit_head_upcast():
    import peft

    original = peft.prepare_model_for_kbit_training
    if not getattr(original, "_ck_is_ours", False):
        peft.prepare_model_for_kbit_training = _wrap_prepare(original)
