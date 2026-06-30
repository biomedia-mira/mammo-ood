from __future__ import annotations

from typing import Iterator

import torch
import torchvision


class NoEmptyCrop:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xt = self.base_transform(x)
        if x.min() == x.max():
            return xt
        while xt.min() == xt.max():
            xt = self.base_transform(x)
        return xt


class TissueAwareCrop:
    """Enhanced crop that requires a minimum fraction of tissue (non-zero) pixels.

    Since mmap preprocessing already zeroes out background, tissue pixels
    are simply those > 0.  This replaces NoEmptyCrop when tissue_aware_crop
    is enabled, ensuring crops land on meaningful breast tissue rather than
    mostly-empty background regions.
    """

    def __init__(self, base_transform, min_tissue_fraction: float = 0.3, max_retries: int = 20):
        self.base_transform = base_transform
        self.min_tissue_fraction = min_tissue_fraction
        self.max_retries = max_retries

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # If the source image is entirely uniform, no point retrying.
        if x.min() == x.max():
            return self.base_transform(x)

        best_crop = None
        best_fraction = -1.0
        for _ in range(self.max_retries):
            crop = self.base_transform(x)
            fraction = (crop > 0).float().mean().item()
            if fraction >= self.min_tissue_fraction:
                return crop
            if fraction > best_fraction:
                best_fraction = fraction
                best_crop = crop
        return best_crop


class PairedLoader:
    """Zip an SSL loader with a probe loader and yield a single nested batch.

    The probe iterator is restarted if needed, but in practice both loaders
    should have matching lengths because they are built from the same split.
    """

    def __init__(self, ssl_loader, probe_loader):
        self.ssl_loader = ssl_loader
        self.probe_loader = probe_loader

    def __iter__(self) -> Iterator[dict[str, object]]:
        probe_iter = iter(self.probe_loader)
        for ssl_batch in self.ssl_loader:
            try:
                probe_batch = next(probe_iter)
            except StopIteration:
                probe_iter = iter(self.probe_loader)
                probe_batch = next(probe_iter)
            yield {'ssl': ssl_batch, 'probe': probe_batch}

    def __len__(self) -> int:
        return min(len(self.ssl_loader), len(self.probe_loader))


def _to_display_crops(crops: torch.Tensor, max_images: int, mean, std) -> torch.Tensor:
    images = crops[:max_images].detach().float().cpu()
    mean = torch.tensor(mean, dtype=images.dtype).view(1, -1, 1, 1)
    std = torch.tensor(std, dtype=images.dtype).view(1, -1, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def log_dino_crops(
    module,
    ssl_batch,
    batch_idx: int,
    *,
    every_n_epochs: int = 10,
    max_images: int = 8,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
) -> None:
    if batch_idx != 0 or module.current_epoch % every_n_epochs != 0:
        return
    if not module.trainer.is_global_zero or module.logger is None:
        return
    experiment = getattr(module.logger, "experiment", None)
    if not hasattr(experiment, "add_image"):
        return

    for key, tag in (
        ("collated_global_crops", "dino/global_crops"),
        ("collated_local_crops", "dino/local_crops"),
    ):
        crops = ssl_batch.get(key) if isinstance(ssl_batch, dict) else None
        if not isinstance(crops, torch.Tensor) or crops.numel() == 0:
            continue
        images = _to_display_crops(crops, max_images=max_images, mean=mean, std=std)
        grid = torchvision.utils.make_grid(images, nrow=4, padding=2)
        experiment.add_image(tag, grid, global_step=module.global_step)

    # Log iBOT masks overlaid on global crops.
    masks = ssl_batch.get("collated_masks") if isinstance(ssl_batch, dict) else None
    crops = ssl_batch.get("collated_global_crops") if isinstance(ssl_batch, dict) else None
    if not isinstance(masks, torch.Tensor) or masks.numel() == 0:
        return
    if not isinstance(crops, torch.Tensor) or crops.numel() == 0:
        return
    _, _, H, W = crops.shape
    N = masks.shape[1]
    # Infer patch grid from crop spatial dims and known patch sizes.
    hp = wp = None
    for ps in (16, 14, 8):
        hp_try, wp_try = H // ps, W // ps
        if hp_try * wp_try == N:
            hp, wp = hp_try, wp_try
            break
    if hp is None:
        return
    crop_imgs = _to_display_crops(crops, max_images=max_images, mean=mean, std=std)
    mask_2d = masks[:max_images].detach().float().cpu().reshape(-1, 1, hp, wp)
    mask_up = torch.nn.functional.interpolate(mask_2d, size=(H, W), mode='nearest')
    # Overlay: dim unmasked patches, add red tint on masked patches
    # so masks are visible even on black background.
    overlay = crop_imgs * (1.0 - 0.5 * mask_up)
    overlay[:, 0:1] = overlay[:, 0:1] + 0.4 * mask_up  # red channel boost
    overlay = overlay.clamp(0.0, 1.0)
    grid = torchvision.utils.make_grid(overlay, nrow=4, padding=2)
    experiment.add_image("dino/ibot_masks", grid, global_step=module.global_step)
