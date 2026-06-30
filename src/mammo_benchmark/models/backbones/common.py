import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class BackboneBundle:
    backbone: nn.Module
    token_dim: int
    layer_ids: Tuple[int, ...]
    supports_cls_token: bool
    native_feature_dim: int


@dataclass
class LayerFeatures:
    patch_tokens: torch.Tensor
    cls_token: Optional[torch.Tensor] = None


def torch_load_compat(checkpoint_path: str):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def load_state_dict_with_report(
    module: nn.Module,
    state_dict: Dict[str, Any],
    *,
    context: str,
    strict: bool,
) -> None:
    if strict:
        module.load_state_dict(state_dict, strict=True)
        print(f"[Backbone] {context}: strict=True load succeeded")
        return

    incompatible = module.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        print(f"[Backbone] {context}: strict=False missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[Backbone] {context} missing_keys (first 10): {missing[:10]}")
        if unexpected:
            print(f"[Backbone] {context} unexpected_keys (first 10): {unexpected[:10]}")


def flatten_nested_block_keys(state_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Convert nested block keys (blocks.GROUP.LAYER.*) to flat (blocks.LAYER.*)."""
    out = {}
    for k, v in state_dict.items():
        key = k[len(prefix):] if prefix and k.startswith(prefix) else k
        m = re.match(r"blocks\.(\d+)\.(\d+)\.(.*)", key)
        if m:
            out[f"blocks.{m.group(2)}.{m.group(3)}"] = v
        else:
            out[key] = v
    return out
