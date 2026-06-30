import json
import math
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerFeatures, load_state_dict_with_report, torch_load_compat


class UniMedClipVisionBackbone(nn.Module):
    """Wrapper for the UniMedCLIP OpenCLIP ViT image tower."""

    def __init__(self, model: nn.Module, image_mean: Tuple[float, ...], image_std: Tuple[float, ...]):
        super().__init__()
        self.model = model
        self.token_dim = int(model.conv1.out_channels)
        self.supports_cls_token = True
        self.input_size = tuple(int(value) for value in model.image_size)
        self.native_feature_dim = self.token_dim
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1), persistent=False)

        # Native LP uses the CLIP visual CLS after ln_post. proj stays frozen
        # because it maps into the contrastive text-image space.
        if isinstance(self.model.proj, nn.Parameter):
            self.model.proj.requires_grad = False

    def forward_native_feature(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(f"UniMedCLIP expects input size {self.input_size}, got {tuple(x.shape[-2:])}")

        x = (x.clamp(0.0, 1.0) - self.image_mean) / self.image_std
        tokens = self.model.conv1(x)
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1).permute(0, 2, 1)
        cls_token = self.model.class_embedding.to(tokens.dtype) + torch.zeros(
            tokens.shape[0], 1, tokens.shape[-1], dtype=tokens.dtype, device=tokens.device,
        )
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.model.positional_embedding.to(tokens.dtype)
        tokens = self.model.ln_pre(tokens)
        tokens = tokens.permute(1, 0, 2)
        for block in self.model.transformer.resblocks:
            tokens = block(tokens)
        tokens = tokens.permute(1, 0, 2)
        tokens = self.model.ln_post(tokens)
        return tokens[:, 0, :].contiguous()

    def forward_intermediate_features(self, x: torch.Tensor, layer_ids: Tuple[int, ...]) -> list[LayerFeatures]:
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(f"UniMedCLIP expects input size {self.input_size}, got {tuple(x.shape[-2:])}")

        x = (x.clamp(0.0, 1.0) - self.image_mean) / self.image_std
        tokens = self.model.conv1(x)
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1).permute(0, 2, 1)
        cls_token = self.model.class_embedding.to(tokens.dtype) + torch.zeros(
            tokens.shape[0], 1, tokens.shape[-1], dtype=tokens.dtype, device=tokens.device,
        )
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.model.positional_embedding.to(tokens.dtype)
        tokens = self.model.ln_pre(tokens)
        tokens = tokens.permute(1, 0, 2)

        outputs = []
        for layer_idx, block in enumerate(self.model.transformer.resblocks):
            tokens = block(tokens)
            if layer_idx in layer_ids:
                batch_tokens = tokens.permute(1, 0, 2)
                outputs.append(LayerFeatures(
                    patch_tokens=batch_tokens[:, 1:, :].contiguous(),
                    cls_token=batch_tokens[:, 0, :].contiguous(),
                ))
        return outputs


def resolve_unimed_src() -> Path:
    default_src = Path(__file__).resolve().parents[4] / "third_party" / "UniMed-CLIP" / "src"
    return Path(os.environ.get("UNIMED_CLIP_SRC", default_src))


def import_unimed_open_clip():
    unimed_src = resolve_unimed_src()
    if not unimed_src.exists():
        raise FileNotFoundError(
            f"UniMedCLIP source repo not found at {unimed_src}. "
            "Set UNIMED_CLIP_SRC to the UniMed-CLIP/src directory."
        )

    existing = sys.modules.get("open_clip")
    existing_file = str(getattr(existing, "__file__", "")) if existing is not None else ""
    if existing is not None and str(unimed_src) not in existing_file:
        raise RuntimeError(
            "UniMedCLIP requires the custom open_clip fork from the UniMed-CLIP repo. "
            "Run it in a fresh process before importing the pip open_clip package."
        )

    if str(unimed_src) not in sys.path:
        sys.path.insert(0, str(unimed_src))

    from open_clip.model import QuickGELU, VisualTransformer
    from open_clip.transform import get_mean_std

    return QuickGELU, VisualTransformer, get_mean_std


def resize_unimedclip_pos_embed(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
    interpolation: str = "bicubic",
) -> None:
    old_pos_embed = state_dict.get("positional_embedding")
    if old_pos_embed is None:
        return

    extra_tokens = 1
    grid_size = tuple(int(value) for value in model.grid_size)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    pos_emb_tok = old_pos_embed[:extra_tokens]
    pos_emb_img = old_pos_embed[extra_tokens:]
    old_grid_len = int(pos_emb_img.shape[0])
    old_grid_size = int(math.sqrt(old_grid_len))
    if old_grid_size * old_grid_size != old_grid_len:
        raise ValueError(f"Cannot resize UniMedCLIP positional embedding with non-square old grid length={old_grid_len}")

    orig_dtype = pos_emb_img.dtype
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img.float(),
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    ).to(dtype=orig_dtype)
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(grid_size[0] * grid_size[1], -1)
    state_dict["positional_embedding"] = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    print(f"[Backbone] UniMedCLIP resized positional embedding from {(old_grid_size, old_grid_size)} to {grid_size}")


def build_unimedclip_visual_model(
    checkpoint_path: str,
    model_name: str = "ViT-L-14-336-quickgelu",
    input_size: Tuple[int, int] | None = None,
) -> tuple[nn.Module, Tuple[float, ...], Tuple[float, ...]]:
    QuickGELU, VisualTransformer, get_mean_std = import_unimed_open_clip()

    config_path = resolve_unimed_src() / "open_clip" / "model_configs" / f"{model_name}.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    vision_cfg = config["vision_cfg"]
    width = int(vision_cfg["width"])
    head_width = int(vision_cfg.get("head_width", 64))
    patch_size = int(vision_cfg["patch_size"])
    image_size = tuple(int(value) for value in input_size) if input_size is not None else int(vision_cfg["image_size"])
    image_size_tuple = image_size if isinstance(image_size, tuple) else (image_size, image_size)
    if any(size % patch_size != 0 for size in image_size_tuple):
        raise ValueError(f"UniMedCLIP input size must be divisible by patch_size={patch_size}, got {image_size_tuple}")

    model = VisualTransformer(
        image_size=image_size,
        patch_size=patch_size,
        width=width,
        layers=int(vision_cfg["layers"]),
        heads=width // head_width,
        mlp_ratio=float(vision_cfg.get("mlp_ratio", 4.0)),
        output_dim=int(config["embed_dim"]),
        act_layer=QuickGELU if config.get("quick_gelu", False) else nn.GELU,
    )

    checkpoint = torch_load_compat(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]

    visual_state_dict = {}
    visual_prefixes = ("conv1.", "class_embedding", "positional_embedding", "ln_pre.", "transformer.", "ln_post.", "proj")
    for key, value in state_dict.items():
        normalized_key = key.removeprefix("module.")
        if normalized_key.startswith("visual."):
            visual_state_dict[normalized_key.removeprefix("visual.")] = value
        elif normalized_key.startswith(visual_prefixes):
            visual_state_dict[normalized_key] = value

    if not visual_state_dict:
        raise ValueError(f"No UniMedCLIP visual weights found in {checkpoint_path}")

    resize_unimedclip_pos_embed(visual_state_dict, model)
    load_state_dict_with_report(model, visual_state_dict, context="UniMedCLIP visual tower", strict=True)
    image_mean, image_std = get_mean_std()
    return model, tuple(image_mean), tuple(image_std)
