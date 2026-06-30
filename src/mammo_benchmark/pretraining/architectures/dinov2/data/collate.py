# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


def _apply_tissue_constraint(mask, tissue_mask):
    """Restrict an iBOT block mask to tissue patches only.

    After the standard block-mask generator produces a 2-D bool mask, this
    function zeroes out masked positions that fall on background (non-tissue)
    patches and then fills the shortfall by randomly selecting from the
    remaining *tissue* patches that are not yet masked.

    Args:
        mask: bool tensor (H_patches, W_patches) from MaskingGenerator.
        tissue_mask: bool tensor (H_patches, W_patches) where True = tissue.

    Returns:
        Adjusted bool tensor of the same shape.
    """
    target_count = int(mask.sum().item())
    # Keep only masked positions that overlap with tissue.
    mask = mask & tissue_mask
    current_count = int(mask.sum().item())

    if current_count < target_count:
        # Fill shortfall from un-masked tissue patches.
        available = tissue_mask & (~mask)
        available_flat = available.flatten()
        available_indices = available_flat.nonzero(as_tuple=False).flatten()
        need = min(target_count - current_count, len(available_indices))
        if need > 0:
            chosen = available_indices[torch.randperm(len(available_indices))[:need]]
            mask_flat = mask.flatten()
            mask_flat[chosen] = True
            mask = mask_flat.reshape(mask.shape)

    return mask


def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    tissue_aware_mask=False,
):
    images = [sample["image"] for sample in samples_list]

    n_global_crops = len(images[0]["global_crops"])
    n_local_crops = len(images[0]["local_crops"])

    collated_global_crops = torch.stack([s["global_crops"][i] for i in range(n_global_crops) for s in images])

    collated_local_crops = torch.stack([s["local_crops"][i] for i in range(n_local_crops) for s in images])

    # Collate per-crop tissue masks when tissue-aware masking is active.
    collated_tissue_masks = None
    if tissue_aware_mask and "tissue_masks" in images[0]:
        collated_tissue_masks = torch.stack(
            [s["tissue_masks"][i] for i in range(n_global_crops) for s in images]
        )  # (B_total, H_patches, W_patches)

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    # Apply tissue constraint: restrict masks to tissue patches only.
    if collated_tissue_masks is not None:
        for idx in range(len(masks_list)):
            tissue_mask_i = collated_tissue_masks[idx]  # (H_patches, W_patches)
            masks_list[idx] = _apply_tissue_constraint(masks_list[idx], tissue_mask_i)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
