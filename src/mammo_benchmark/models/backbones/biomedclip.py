import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from .common import LayerFeatures, load_state_dict_with_report, torch_load_compat


class BioMedClipVisionBackbone(nn.Module):
    """Wrapper for the BioMedCLIP OpenCLIP/TIMM ViT image tower."""

    def __init__(self, model: nn.Module, image_mean: Tuple[float, ...], image_std: Tuple[float, ...]):
        super().__init__()
        self.model = model
        self.token_dim = int(model.num_features)
        self.supports_cls_token = getattr(model, "cls_token", None) is not None
        self.input_size = tuple(int(value) for value in model.patch_embed.img_size)
        self.num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 1 if self.supports_cls_token else 0))
        self.native_feature_dim = self.token_dim
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1), persistent=False)

    def forward_native_feature(self, x: torch.Tensor) -> torch.Tensor:
        if not self.supports_cls_token:
            raise RuntimeError("BioMedCLIP backbone does not expose CLS token")
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(f"BioMedCLIP expects input size {self.input_size}, got {tuple(x.shape[-2:])}")

        x = (x.clamp(0.0, 1.0) - self.image_mean) / self.image_std
        tokens = self.model.patch_embed(x)
        tokens = self.model._pos_embed(tokens)
        tokens = self.model.patch_drop(tokens)
        tokens = self.model.norm_pre(tokens)
        for block in self.model.blocks:
            tokens = block(tokens)
        tokens = self.model.norm(tokens)
        return tokens[:, 0, :].contiguous()

    def forward_intermediate_features(self, x: torch.Tensor, layer_ids: Tuple[int, ...]) -> list[LayerFeatures]:
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(f"BioMedCLIP expects input size {self.input_size}, got {tuple(x.shape[-2:])}")

        x = (x.clamp(0.0, 1.0) - self.image_mean) / self.image_std
        tokens = self.model.patch_embed(x)
        tokens = self.model._pos_embed(tokens)
        tokens = self.model.patch_drop(tokens)
        tokens = self.model.norm_pre(tokens)

        outputs = []
        for layer_idx, block in enumerate(self.model.blocks):
            tokens = block(tokens)
            if layer_idx in layer_ids:
                cls_token = tokens[:, 0, :].contiguous() if self.supports_cls_token else None
                outputs.append(LayerFeatures(patch_tokens=tokens[:, self.num_prefix_tokens:, :].contiguous(), cls_token=cls_token))
        return outputs


def build_biomedclip_visual_model(checkpoint_dir: str, input_size: Tuple[int, int] | None = None) -> tuple[nn.Module, Tuple[float, ...], Tuple[float, ...]]:
    from open_clip.timm_model import TimmModel

    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "open_clip_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    model_cfg = config["model_cfg"]
    vision_cfg = model_cfg["vision_cfg"]
    preprocess_cfg = config["preprocess_cfg"]

    model = TimmModel(
        vision_cfg["timm_model_name"],
        embed_dim=model_cfg["embed_dim"],
        image_size=vision_cfg["image_size"],
        pool=vision_cfg.get("timm_pool", "avg"),
        proj=vision_cfg.get("timm_proj", "linear"),
        proj_bias=vision_cfg.get("timm_proj_bias", False),
        pretrained=vision_cfg.get("timm_model_pretrained", False),
    )
    checkpoint = torch_load_compat(str(checkpoint_dir / "open_clip_pytorch_model.bin"))
    state_dict = {
        key.removeprefix("visual."): value
        for key, value in checkpoint.items()
        if key.startswith("visual.")
    }
    load_state_dict_with_report(model, state_dict, context="BioMedCLIP visual tower", strict=True)
    visual_model = model.trunk
    if input_size is not None:
        input_size = tuple(int(value) for value in input_size)
        if input_size != tuple(int(value) for value in visual_model.patch_embed.img_size):
            visual_model.set_input_size(img_size=input_size)
            print(f"[Backbone] BioMedCLIP resized ViT positional embedding to input_size={input_size}")
    return visual_model, tuple(preprocess_cfg["mean"]), tuple(preprocess_cfg["std"])
