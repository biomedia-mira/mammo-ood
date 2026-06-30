from __future__ import annotations

from typing import Any, Dict

import torch


def torch_load_compat(checkpoint_path: str) -> Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def extract_prefixed_state_dict(checkpoint_path: str, prefix: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch_load_compat(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefixed_state_dict = {
        key.replace(prefix, "", 1): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not prefixed_state_dict:
        raise ValueError(f"Could not find {prefix}* weights in checkpoint: {checkpoint_path}")
    return prefixed_state_dict


def load_best_or_current_prefixed_state_dict(checkpoint_callback, prefix: str, current_state_dict):
    if checkpoint_callback.best_model_path:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        return extract_prefixed_state_dict(checkpoint_callback.best_model_path, prefix)

    print(
        f"Warning: no best checkpoint found for prefix {prefix}, falling back to current in-memory weights."
    )
    return current_state_dict


def load_last_or_current_prefixed_state_dict(checkpoint_callback, prefix: str, current_state_dict):
    last_model_path = getattr(checkpoint_callback, "last_model_path", None)
    if last_model_path:
        print(f"Last checkpoint: {last_model_path}")
        return extract_prefixed_state_dict(last_model_path, prefix)

    print(
        f"Warning: no last checkpoint found for prefix {prefix}, falling back to current in-memory weights."
    )
    return current_state_dict
