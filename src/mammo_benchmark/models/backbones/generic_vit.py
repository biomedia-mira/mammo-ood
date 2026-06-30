from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerFeatures


class DinoBackbone(nn.Module):
    """Wrapper for models with get_intermediate_layers() (DINOv2, DINOv3, MAMA)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.token_dim = int(model.embed_dim)
        self.supports_cls_token = True
        self.native_feature_dim = self.token_dim

    def forward_native_feature(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model.get_intermediate_layers(
            x,
            n=1,
            reshape=False,
            return_class_token=True,
            norm=True,
        )
        _, cls_token = outputs[-1]
        return cls_token.contiguous()

    def forward_intermediate_features(self, x: torch.Tensor, layer_ids: Tuple[int, ...]) -> list[LayerFeatures]:
        outputs = self.model.get_intermediate_layers(
            x,
            n=layer_ids,
            reshape=False,
            return_class_token=True,
            norm=False,
        )
        return [LayerFeatures(patch_tokens=patch_tokens.contiguous(), cls_token=cls_token.contiguous()) for patch_tokens, cls_token in outputs]


class MAEBackbone(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.token_dim = int(model.embed_dim)
        self.supports_cls_token = True
        self.native_feature_dim = self.token_dim

    def forward_native_feature(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.model.patch_embed(x)
        patch_pos_embed = self.model.pos_embed[:, 1:, :]
        if patch_pos_embed.shape[1] != tokens.shape[1]:
            grid_h = self.model.patch_embed.img_size[0] // self.model.patch_embed.patch_size[0]
            grid_w = self.model.patch_embed.img_size[1] // self.model.patch_embed.patch_size[1]
            patch_pos_embed = patch_pos_embed.reshape(1, grid_h, grid_w, -1).permute(0, 3, 1, 2)
            new_h = x.shape[2] // self.model.patch_embed.patch_size[0]
            new_w = x.shape[3] // self.model.patch_embed.patch_size[1]
            patch_pos_embed = F.interpolate(
                patch_pos_embed, size=(new_h, new_w), mode="bicubic", align_corners=False,
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, self.token_dim)

        tokens = tokens + patch_pos_embed
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        tokens = torch.cat((cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        for block in self.model.blocks:
            tokens = block(tokens)
        tokens = self.model.norm(tokens)
        return tokens[:, 0, :].contiguous()

    def forward_intermediate_features(self, x: torch.Tensor, layer_ids: Tuple[int, ...]) -> list[LayerFeatures]:
        tokens = self.model.patch_embed(x)
        patch_pos_embed = self.model.pos_embed[:, 1:, :]

        # Interpolate pos_embed if input size differs from what the model expects
        if patch_pos_embed.shape[1] != tokens.shape[1]:
            grid_h = self.model.patch_embed.img_size[0] // self.model.patch_embed.patch_size[0]
            grid_w = self.model.patch_embed.img_size[1] // self.model.patch_embed.patch_size[1]
            patch_pos_embed = patch_pos_embed.reshape(1, grid_h, grid_w, -1).permute(0, 3, 1, 2)
            new_h = x.shape[2] // self.model.patch_embed.patch_size[0]
            new_w = x.shape[3] // self.model.patch_embed.patch_size[1]
            patch_pos_embed = F.interpolate(
                patch_pos_embed, size=(new_h, new_w), mode="bicubic", align_corners=False,
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, self.token_dim)

        tokens = tokens + patch_pos_embed
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        tokens = torch.cat((cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)

        outputs = []
        for layer_idx, block in enumerate(self.model.blocks):
            tokens = block(tokens)
            if layer_idx in layer_ids:
                outputs.append(LayerFeatures(patch_tokens=tokens[:, 1:, :].contiguous(), cls_token=tokens[:, 0, :].contiguous()))
        return outputs
