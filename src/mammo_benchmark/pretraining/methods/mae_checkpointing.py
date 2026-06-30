from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from common.checkpointing import torch_load_compat


def _filter_keys(keys: Iterable[str], allowed_prefixes: Iterable[str]) -> list[str]:
    prefixes = tuple(allowed_prefixes)
    return [key for key in keys if not key.startswith(prefixes)]


def load_state_dict_with_report(
    module: nn.Module,
    state_dict: Dict[str, Any],
    *,
    context: str,
    strict: bool,
    allowed_missing_prefixes: Tuple[str, ...] = (),
    allowed_unexpected_prefixes: Tuple[str, ...] = (),
    fail_on_remaining: bool,
) -> tuple[list[str], list[str]]:
    if strict:
        module.load_state_dict(state_dict, strict=True)
        print(f"[Checkpoint] {context}: strict=True load succeeded")
        return [], []

    incompatible = module.load_state_dict(state_dict, strict=False)
    raw_missing = list(incompatible.missing_keys)
    raw_unexpected = list(incompatible.unexpected_keys)
    remaining_missing = _filter_keys(raw_missing, allowed_missing_prefixes)
    remaining_unexpected = _filter_keys(raw_unexpected, allowed_unexpected_prefixes)

    if raw_missing or raw_unexpected:
        print(
            f"[Checkpoint] {context}: strict=False "
            f"missing={len(raw_missing)} unexpected={len(raw_unexpected)}"
        )
        if raw_missing:
            print(f"[Checkpoint] {context} missing_keys (first 10): {raw_missing[:10]}")
        if raw_unexpected:
            print(f"[Checkpoint] {context} unexpected_keys (first 10): {raw_unexpected[:10]}")
    else:
        print(f"[Checkpoint] {context}: loaded cleanly")

    if fail_on_remaining and (remaining_missing or remaining_unexpected):
        raise RuntimeError(
            f"{context} checkpoint load left unsupported mismatches. "
            f"missing={remaining_missing[:10]} unexpected={remaining_unexpected[:10]}"
        )

    return raw_missing, raw_unexpected


def _extract_mae_state_dict(checkpoint: Dict[str, Any], checkpoint_format: str) -> Dict[str, torch.Tensor]:
    if checkpoint_format == "official":
        state_dict = checkpoint.get("model", checkpoint)
    elif checkpoint_format == "mammo_model":
        state_dict = checkpoint.get("model", checkpoint)
        if "state_dict" in checkpoint:
            prefixed = {
                key.replace("model.", "", 1): value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("model.")
            }
            if prefixed:
                state_dict = prefixed
    else:
        raise ValueError(f"Unsupported MAE checkpoint format: {checkpoint_format}")

    state_dict = dict(state_dict)
    state_dict.pop("pos_embed", None)
    state_dict.pop("decoder_pos_embed", None)
    return state_dict


def load_mae_ssl_checkpoint(model: nn.Module, checkpoint_path: str, checkpoint_format: str) -> None:
    checkpoint = torch_load_compat(checkpoint_path)
    state_dict = _extract_mae_state_dict(checkpoint, checkpoint_format)
    load_state_dict_with_report(
        model,
        state_dict,
        context=f"MAE init from {checkpoint_path}",
        strict=False,
        allowed_missing_prefixes=(
            "pos_embed",
            "mask_token",
            "decoder_pos_embed",
            "decoder_embed",
            "decoder_blocks",
            "decoder_norm",
            "decoder_pred",
        ),
        allowed_unexpected_prefixes=("pos_embed", "decoder_pos_embed"),
        fail_on_remaining=True,
    )
