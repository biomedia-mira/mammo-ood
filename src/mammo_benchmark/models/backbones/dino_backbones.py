import sys
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn


PACKAGE_DIR = Path(__file__).resolve().parents[2]
PRETRAINING_DIR = PACKAGE_DIR / "pretraining"
if str(PRETRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAINING_DIR))


def build_dinov2_model(variant: str, img_size: Union[int, Tuple[int, int]] = 518) -> nn.Module:
    """Build a local DINOv2 model compatible with official/checkpoint state dicts."""
    from architectures.dinov2.models.vision_transformer import vit_base as dinov2_vit_base

    builders = {
        "dinov2_vitb14": (dinov2_vit_base, 0),
        "dinov2_vitb14_reg": (dinov2_vit_base, 4),
    }
    if variant not in builders:
        raise ValueError(f"Unsupported DINOv2 variant={variant}")

    builder, num_register_tokens = builders[variant]
    model = builder(
        img_size=img_size,
        patch_size=14,
        init_values=1.0e-5,
        ffn_layer="mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        num_register_tokens=num_register_tokens,
        interpolate_offset=0.1,
        interpolate_antialias=False,
    )
    model.init_weights()
    return model


def build_dinov3_model(variant: str) -> nn.Module:
    """Build a DINOv3 model from the local architecture code."""
    from architectures.dinov3.models.vision_transformer import vit_base as dinov3_vit_base
    from architectures.dinov3.models.vision_transformer import vit_large as dinov3_vit_large

    builders = {"vitb": dinov3_vit_base, "vitl": dinov3_vit_large}
    builder = builders[variant]
    model = builder(patch_size=16, n_storage_tokens=4, layerscale_init=1e-5, mask_k_bias=True)
    model.init_weights()
    return model


def build_xray_dino_model(img_size: int = 512) -> nn.Module:
    """Build a DINOv2-arch ViT-L with patch_size=16 for X-ray DINO."""
    from architectures.dinov2.models.vision_transformer import DinoVisionTransformer

    model = DinoVisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        ffn_layer="mlp",
        init_values=1e-5,
        num_register_tokens=0,
    )
    model.init_weights()
    return model
