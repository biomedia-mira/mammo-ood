from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from architectures.dinov3.data.transforms import GaussianBlur
from methods.dino_common import NoEmptyCrop, TissueAwareCrop

class MammoDataAugmentationDINOv3:
    def __init__(
        self,
        *,
        global_crops_scale: Sequence[float],
        local_crops_scale: Sequence[float],
        local_crops_number: int,
        global_crops_size: Sequence[int],
        local_crops_size: Sequence[int],
        input_channels: int,
        horizontal_flips: bool,
        share_color_jitter: bool,
        random_color_jitter: float,
        color_jitter_prob: float,
        mean: Sequence[float],
        std: Sequence[float],
        crop_ratio: Sequence[float] | None = None,
        online_probe_enabled: bool = False,
        tissue_aware_crop: bool = False,
        tissue_aware_mask: bool = False,
        patch_size: int = 16,
    ) -> None:
        self.global_crops_scale = tuple(global_crops_scale)
        self.local_crops_scale = tuple(local_crops_scale)
        self.local_crops_number = int(local_crops_number)
        self.global_crops_size = tuple(int(v) for v in global_crops_size)
        self.local_crops_size = tuple(int(v) for v in local_crops_size)
        self.input_channels = int(input_channels)
        self.horizontal_flips = bool(horizontal_flips)
        self.share_color_jitter = bool(share_color_jitter)
        self.random_color_jitter = float(random_color_jitter)
        self.color_jitter_prob = float(color_jitter_prob)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.tissue_aware_crop = bool(tissue_aware_crop)
        self.tissue_aware_mask = bool(tissue_aware_mask)
        self.patch_size = int(patch_size)

        # Aspect ratio: auto-compute from global_crops_size (W/H) if not given.
        # Mammograms are portrait (H > W); constraining ratio prevents landscape distortion.
        if crop_ratio is None:
            h, w = self.global_crops_size[0], self.global_crops_size[1]
            base_ratio = w / h
            self.crop_ratio = (base_ratio * 0.95, base_ratio * 1.05)
        else:
            self.crop_ratio = tuple(float(v) for v in crop_ratio)

        crop_wrapper = TissueAwareCrop if self.tissue_aware_crop else NoEmptyCrop

        self.geometric_augmentation_global = crop_wrapper(v2.Compose([
            v2.RandomResizedCrop(
                self.global_crops_size,
                scale=self.global_crops_scale,
                ratio=self.crop_ratio,
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5 if self.horizontal_flips else 0.0),
        ]))

        self.geometric_augmentation_local = crop_wrapper(v2.Compose([
            v2.RandomResizedCrop(
                self.local_crops_size,
                scale=self.local_crops_scale,
                ratio=self.crop_ratio,
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5 if self.horizontal_flips else 0.0),
        ]))

        # DINOv3 in this repo uses grayscale-repeat 3-channel inputs, so
        # keep MAE-style light brightness/contrast jitter only.
        if self.random_color_jitter > 0 and self.color_jitter_prob > 0:
            color_jitter = v2.RandomApply([
                v2.ColorJitter(
                    brightness=self.random_color_jitter,
                    contrast=self.random_color_jitter,
                )
            ], p=self.color_jitter_prob)
        else:
            color_jitter = torch.nn.Identity()

        self.shared_color_transform = color_jitter
        self.global_transfo1 = v2.Compose([
            color_jitter if not self.share_color_jitter else torch.nn.Identity(),
            GaussianBlur(p=1.0),
            v2.Normalize(mean=self.mean, std=self.std),
        ])
        self.global_transfo2 = v2.Compose([
            color_jitter if not self.share_color_jitter else torch.nn.Identity(),
            GaussianBlur(p=0.1),
            v2.Normalize(mean=self.mean, std=self.std),
        ])
        self.local_transfo = v2.Compose([
            color_jitter if not self.share_color_jitter else torch.nn.Identity(),
            GaussianBlur(p=0.5),
            v2.Normalize(mean=self.mean, std=self.std),
        ])

    def _patch_tissue_mask(self, crop: torch.Tensor) -> torch.Tensor:
        """Compute a binary patch-level tissue mask from a crop (before normalize).

        Returns a bool tensor of shape (H_patches, W_patches) where True means
        the patch contains at least one non-zero pixel (i.e. tissue).
        Reuses the same logic as MAE's ``informative_patch_mask``.
        """
        return (
            F.max_pool2d(crop[:1], kernel_size=self.patch_size, stride=self.patch_size) > 0
        ).squeeze(0)  # (H_patches, W_patches)

    def __call__(self, image: torch.Tensor) -> dict[str, Any]:
        if self.share_color_jitter:
            image = self.shared_color_transform(image)

        im1_base = self.geometric_augmentation_global(image)
        im2_base = self.geometric_augmentation_global(image)

        if torch.rand(1).item() > 0.5:
            global_crop_1 = self.global_transfo1(im1_base)
            global_crop_2 = self.global_transfo2(im2_base)
        else:
            global_crop_1 = self.global_transfo2(im1_base)
            global_crop_2 = self.global_transfo1(im2_base)

        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image))
            for _ in range(self.local_crops_number)
        ]

        out: dict[str, Any] = {
            'global_crops': [global_crop_1, global_crop_2],
            'global_crops_teacher': [global_crop_1, global_crop_2],
            'local_crops': local_crops,
            'offsets': (),
        }

        if self.tissue_aware_mask:
            # Compute tissue masks from crops BEFORE photometric transforms.
            # im1_base / im2_base are post-geometric, pre-normalize — background
            # pixels are still 0, so ``> 0`` correctly identifies tissue.
            out['tissue_masks'] = [
                self._patch_tissue_mask(im1_base),
                self._patch_tissue_mask(im2_base),
            ]

        return out


class MammoValAugmentationDINOv3:
    """Deterministic validation transform: resize + center crop + normalize."""

    def __init__(
        self,
        *,
        global_crops_size: Sequence[int],
        local_crops_size: Sequence[int],
        local_crops_number: int,
        input_channels: int,
        mean: Sequence[float],
        std: Sequence[float],
        tissue_aware_mask: bool = False,
        patch_size: int = 16,
    ) -> None:
        self.global_crops_size = tuple(int(v) for v in global_crops_size)
        self.local_crops_size = tuple(int(v) for v in local_crops_size)
        self.local_crops_number = int(local_crops_number)
        self.tissue_aware_mask = bool(tissue_aware_mask)
        self.patch_size = int(patch_size)

        self.global_base_transform = v2.Compose([
            v2.Resize(self.global_crops_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(self.global_crops_size),
        ])
        self.local_base_transform = v2.Compose([
            v2.Resize(self.local_crops_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(self.local_crops_size),
        ])
        self.normalize = v2.Normalize(mean=mean, std=std)

    def _patch_tissue_mask(self, crop: torch.Tensor) -> torch.Tensor:
        return (
            F.max_pool2d(crop[:1], kernel_size=self.patch_size, stride=self.patch_size) > 0
        ).squeeze(0)

    def __call__(self, image: torch.Tensor) -> dict[str, Any]:
        global_base = self.global_base_transform(image)
        local_base = self.local_base_transform(image)
        global_crop = self.normalize(global_base)
        local_crop = self.normalize(local_base)
        out: dict[str, Any] = {
            'global_crops': [global_crop, global_crop],
            'global_crops_teacher': [global_crop, global_crop],
            'local_crops': [local_crop] * self.local_crops_number,
            'offsets': (),
        }
        if self.tissue_aware_mask:
            tissue_mask = self._patch_tissue_mask(global_base)
            out['tissue_masks'] = [tissue_mask, tissue_mask]
        return out
