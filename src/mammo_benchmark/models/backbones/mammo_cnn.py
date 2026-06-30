from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .common import LayerFeatures, torch_load_compat
from .mammo_clip import EfficientNet


MAMMO_CLIP_CNN_ARCHES = {
    "tf_efficientnetv2-detect": ("efficientnet-b2", 1408),
    "tf_efficientnet_b5_ns-detect": ("efficientnet-b5", 2048),
}


class EfficientNetCNNBackbone(nn.Module):
    supports_cls_token = False

    def __init__(self, model: EfficientNet, token_dim: int):
        super().__init__()
        self.model = model
        self.token_dim = int(token_dim)
        self.native_feature_dim = self.token_dim

    def forward_native_feature(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.model.extract_features(x)
        if feature_map.ndim != 4:
            raise RuntimeError(f"Expected CNN feature map with shape [B, C, H, W], got {tuple(feature_map.shape)}")
        return feature_map.mean(dim=[2, 3])

    def forward_intermediate_features(self, x: torch.Tensor, layer_ids: Tuple[int, ...]) -> list[LayerFeatures]:
        if tuple(layer_ids) != (-1,):
            raise ValueError(f"EfficientNet CNN supports only layer_ids=(-1,), got {layer_ids}")
        feature_map = self.model.extract_features(x)
        if feature_map.ndim != 4:
            raise RuntimeError(f"Expected CNN feature map with shape [B, C, H, W], got {tuple(feature_map.shape)}")
        patch_tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        return [LayerFeatures(patch_tokens=patch_tokens)]


def _load_image_encoder_state(checkpoint: Dict[str, Any], checkpoint_path: str) -> Dict[str, torch.Tensor]:
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise KeyError(f"Mammo-CLIP checkpoint missing dict key 'model': {checkpoint_path}")

    state_dict = {
        key.removeprefix("image_encoder."): value
        for key, value in model_state.items()
        if key.startswith("image_encoder.")
    }
    if not state_dict:
        raise KeyError(f"Mammo-CLIP checkpoint has no image_encoder.* weights: {checkpoint_path}")
    return state_dict


def _load_prefixed_image_encoder_state(checkpoint: Dict[str, Any], checkpoint_path: str) -> Dict[str, torch.Tensor]:
    state_dict = {
        key.removeprefix("module.image_encoder."): value
        for key, value in checkpoint.items()
        if key.startswith("module.image_encoder.")
    }
    if not state_dict:
        raise KeyError(f"Checkpoint has no module.image_encoder.* weights: {checkpoint_path}")
    return state_dict


def _image_encoder_config(checkpoint: Dict[str, Any], checkpoint_path: str) -> Dict[str, Any]:
    try:
        config = checkpoint["config"]["model"]["image_encoder"]
    except KeyError as error:
        raise KeyError(f"Mammo-CLIP checkpoint missing config.model.image_encoder: {checkpoint_path}") from error
    if not isinstance(config, dict):
        raise TypeError(f"Expected config.model.image_encoder to be a dict in {checkpoint_path}, got {type(config)}")
    return config


def build_mammo_clip_cnn_backbone(checkpoint_path: str, input_size: Tuple[int, int]) -> EfficientNetCNNBackbone:
    checkpoint_path_obj = Path(checkpoint_path)
    checkpoint = torch_load_compat(str(checkpoint_path_obj))
    encoder_config = _image_encoder_config(checkpoint, str(checkpoint_path_obj))
    encoder_name = str(encoder_config.get("name", ""))
    if encoder_name not in MAMMO_CLIP_CNN_ARCHES:
        raise ValueError(
            f"Unsupported Mammo-CLIP CNN image encoder '{encoder_name}' in {checkpoint_path_obj}. "
            f"Supported: {sorted(MAMMO_CLIP_CNN_ARCHES)}"
        )

    arch_name, token_dim = MAMMO_CLIP_CNN_ARCHES[encoder_name]
    model = EfficientNet.from_name(arch_name, num_classes=1, image_size=input_size)
    image_encoder_state = _load_image_encoder_state(checkpoint, str(checkpoint_path_obj))
    model.load_state_dict(image_encoder_state, strict=True)
    print(
        f"[Backbone] Mammo-CLIP CNN strict load succeeded "
        f"arch={arch_name} token_dim={token_dim} weights={len(image_encoder_state)}"
    )
    return EfficientNetCNNBackbone(model=model, token_dim=token_dim)


def build_versamammo_effnet_backbone(checkpoint_path: str, input_size: Tuple[int, int]) -> EfficientNetCNNBackbone:
    checkpoint_path_obj = Path(checkpoint_path)
    checkpoint = torch_load_compat(str(checkpoint_path_obj))
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected VersaMammo EfficientNet checkpoint to be a dict, got {type(checkpoint)}")

    arch_name = "efficientnet-b5"
    token_dim = 2048
    model = EfficientNet.from_name(arch_name, num_classes=1, image_size=input_size)
    image_encoder_state = _load_prefixed_image_encoder_state(checkpoint, str(checkpoint_path_obj))
    model.load_state_dict(image_encoder_state, strict=True)
    print(
        f"[Backbone] VersaMammo EfficientNet strict load succeeded "
        f"arch={arch_name} token_dim={token_dim} weights={len(image_encoder_state)}"
    )
    return EfficientNetCNNBackbone(model=model, token_dim=token_dim)
