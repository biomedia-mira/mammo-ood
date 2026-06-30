from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule, Trainer


@dataclass
class _ReferenceSample:
    index: int
    raw_image: torch.Tensor
    model_input: torch.Tensor


def _normalize_image(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().float()
    image = image - image.min()
    image = image / image.clamp_min(0).max().clamp_min(1e-6)
    return image


def _ensure_rgb(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
    if image.shape[0] == 1:
        return image.repeat(3, 1, 1)
    if image.shape[0] >= 3:
        return image[:3]
    return image.repeat(3 // image.shape[0] + 1, 1, 1)[:3]


def _colorize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap = _normalize_image(heatmap.unsqueeze(0)).squeeze(0)
    red = heatmap
    green = (1.0 - (2.0 * heatmap - 1.0).abs()).clamp(0.0, 1.0)
    blue = 1.0 - heatmap
    return torch.stack([red, green, blue], dim=0)


def _select_reference_indices(dataset_length: int, num_reference_images: int) -> list[int]:
    if dataset_length <= 0 or num_reference_images <= 0:
        return []
    if num_reference_images >= dataset_length:
        return list(range(dataset_length))
    if num_reference_images == 1:
        return [dataset_length // 2]

    last_index = dataset_length - 1
    step = last_index / float(num_reference_images - 1)
    indices = {int(round(i * step)) for i in range(num_reference_images)}
    return sorted(indices)


class TokenHeatmapCallback(Callback):
    def __init__(
        self,
        *,
        enabled: bool = True,
        every_n_epochs: int = 4,
        num_reference_images: int = 3,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.every_n_epochs = max(int(every_n_epochs), 1)
        self.num_reference_images = max(int(num_reference_images), 1)
        self._references: list[_ReferenceSample] = []

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled or not trainer.is_global_zero:
            return

        datamodule = trainer.datamodule
        dataset = getattr(datamodule, "dataset_val", None)
        if dataset is None:
            return

        reference_indices = _select_reference_indices(len(dataset), self.num_reference_images)
        references: list[_ReferenceSample] = []
        for index in reference_indices:
            raw_image = dataset.load_image_tensor(index).cpu()
            transformed, _ = dataset[index]
            model_input = transformed["global_crops"][0].detach().cpu()
            references.append(
                _ReferenceSample(
                    index=index,
                    raw_image=raw_image,
                    model_input=model_input,
                )
            )
        self._references = references

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled or not trainer.is_global_zero or not self._references:
            return

        epoch_number = trainer.current_epoch + 1
        if epoch_number % self.every_n_epochs != 0:
            return

        logger = trainer.logger
        experiment = getattr(logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_image"):
            return

        device = pl_module.device
        backbone = pl_module.ssl_model.student.backbone
        was_training = backbone.training
        backbone.eval()

        try:
            try:
                inputs = torch.stack([sample.model_input for sample in self._references], dim=0).to(device, non_blocking=True)
                with torch.no_grad():
                    outputs = backbone.forward_features(inputs)
                cls_tokens = outputs["x_norm_clstoken"]
                patch_tokens = outputs["x_norm_patchtokens"]

                patch_size = int(pl_module.cfg.student.patch_size)
                _, _, height, width = inputs.shape
                grid_h = height // patch_size
                grid_w = width // patch_size
                expected_patches = grid_h * grid_w

                if patch_tokens.shape[1] != expected_patches:
                    pl_module.print(
                        "[TokenHeatmap] Skipping heatmap logging because patch grid "
                        f"({patch_tokens.shape[1]}) does not match expected {expected_patches} "
                        f"for input {height}x{width} and patch_size={patch_size}."
                    )
                    return

                similarity = F.cosine_similarity(patch_tokens, cls_tokens.unsqueeze(1), dim=-1)
                similarity_maps = similarity.reshape(similarity.shape[0], grid_h, grid_w).detach().float().cpu()

                for sample, sim_map in zip(self._references, similarity_maps):
                    raw_rgb = _ensure_rgb(_normalize_image(sample.raw_image)).cpu()
                    heat_rgb = _colorize_heatmap(sim_map).cpu()
                    heat_rgb = F.interpolate(
                        heat_rgb.unsqueeze(0),
                        size=raw_rgb.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    overlay = (0.6 * raw_rgb + 0.4 * heat_rgb).clamp(0.0, 1.0)
                    panel = torch.cat([raw_rgb, heat_rgb, overlay], dim=-1).cpu()
                    experiment.add_image(
                        f"token_heatmap/ref_{sample.index}",
                        panel,
                        global_step=trainer.global_step,
                        dataformats="CHW",
                    )
            except Exception as exc:  # pragma: no cover - diagnostics must never kill training
                pl_module.print(f"[TokenHeatmap] Skipping heatmap logging due to error: {exc}")
        finally:
            backbone.train(was_training)
