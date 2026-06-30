from typing import Tuple

import torch
import torch.nn as nn

from .common import LayerFeatures


class MedSiglipVisionBackbone(nn.Module):
    """Wrapper for the MedSigLIP SigLIP vision tower from Hugging Face."""

    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int] | None = None,
        interpolate_pos_encoding: bool = False,
    ):
        super().__init__()
        self.model = model
        self.token_dim = int(model.config.hidden_size)
        self.supports_cls_token = False
        self.native_feature_dim = self.token_dim
        native_input_size = (int(model.config.image_size), int(model.config.image_size))
        self.input_size = tuple(int(value) for value in input_size) if input_size is not None else native_input_size
        self.interpolate_pos_encoding = bool(interpolate_pos_encoding)
        patch_size = int(model.config.patch_size)
        if self.input_size != native_input_size and not self.interpolate_pos_encoding:
            raise ValueError(
                f"MedSigLIP non-native input size {self.input_size} requires positional interpolation; "
                "use a *_interp model key."
            )
        if self.interpolate_pos_encoding and any(size % patch_size != 0 for size in self.input_size):
            raise ValueError(f"MedSigLIP interpolated input size must be divisible by patch_size={patch_size}, got {self.input_size}")

    def forward_native_feature(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(f"MedSigLIP expects input size {self.input_size}, got {tuple(x.shape[-2:])}")

        x = x.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        hidden_states = self.model.embeddings(x, interpolate_pos_encoding=self.interpolate_pos_encoding)
        encoder_outputs = self.model.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=False,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.model.post_layernorm(last_hidden_state)
        pooled = self.model.head(last_hidden_state)
        return pooled

    def forward_intermediate_features(self, x: torch.Tensor, layer_ids: Tuple[int, ...]) -> list[LayerFeatures]:
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(f"MedSigLIP expects input size {self.input_size}, got {tuple(x.shape[-2:])}")

        x = x.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        hidden_states = self.model.embeddings(x, interpolate_pos_encoding=self.interpolate_pos_encoding)
        encoder_outputs = self.model.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=True,
        )
        all_hidden_states = encoder_outputs.hidden_states

        outputs = []
        for layer_idx in layer_ids:
            hidden_state_idx = layer_idx + 1
            if hidden_state_idx >= len(all_hidden_states):
                raise ValueError(f"MedSigLIP layer_id={layer_idx} is out of range for {len(all_hidden_states) - 1} layers")
            outputs.append(LayerFeatures(patch_tokens=all_hidden_states[hidden_state_idx].contiguous()))
        return outputs
