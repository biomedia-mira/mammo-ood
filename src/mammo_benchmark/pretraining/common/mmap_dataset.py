from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseMammoMMapDataset(Dataset):
    def __init__(
        self,
        data,
        *,
        image_normalization: float,
        mmap_path: str,
        repeat_channels: int = 3,
    ) -> None:
        self.image_normalization = image_normalization
        self.repeat_channels = repeat_channels

        self.study_ids = data.study_id.to_numpy()
        self.image_ids = data.image_id.to_numpy()
        self.global_indices = data.global_index.to_numpy()
        self.data_mmap = np.load(mmap_path, mmap_mode="r")

    def __len__(self):
        return self.study_ids.shape[0]

    def load_image_tensor(self, index: int) -> torch.Tensor:
        global_idx = self.global_indices[index]
        image_tensor = torch.tensor(self.data_mmap[global_idx], dtype=torch.float32)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor / self.image_normalization
        if self.repeat_channels > 1 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(self.repeat_channels, 1, 1)
        return image_tensor
