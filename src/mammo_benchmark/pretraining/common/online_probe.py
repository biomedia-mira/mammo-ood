from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from common.mmap_dataset import BaseMammoMMapDataset


def extract_density_labels(data) -> np.ndarray:
    label_col = "tissueden_index" if "tissueden_index" in data.columns else "tissueden"
    labels = data[label_col].to_numpy(dtype=np.int64)
    if label_col == "tissueden":
        labels = labels - 1
    return labels


def build_probe_eval_transform(image_size, mean, std):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])


class MammoDensityProbeDataset(BaseMammoMMapDataset):
    def __init__(
        self,
        data,
        *,
        image_size,
        mmap_path,
        repeat_channels: int,
        mean,
        std,
        image_normalization: float = 65535.0,
    ) -> None:
        self.labels = extract_density_labels(data)
        self.transform = build_probe_eval_transform(image_size, mean, std)
        super().__init__(
            data,
            image_normalization=image_normalization,
            mmap_path=mmap_path,
            repeat_channels=repeat_channels,
        )

    def __getitem__(self, index):
        image_tensor = self.load_image_tensor(index)
        pixel_values = self.transform(image_tensor)
        return {
            'x': pixel_values,
            'y': torch.tensor(self.labels[index], dtype=torch.long),
            'study_id': self.study_ids[index],
            'image_id': self.image_ids[index],
        }


@dataclass
class ProbeMetrics:
    acc: float
    balacc: float
    auroc: float | None


class OnlineProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = int(num_classes)
        self.epoch_storage = {
            'train': {'probas': [], 'targets': []},
            'val': {'probas': [], 'targets': []},
        }

    def step(self, features: torch.Tensor, targets: torch.Tensor):
        logits = self.head(features.detach())
        loss = self.criterion(logits, targets)
        probas = torch.softmax(logits, dim=1)
        return loss, probas

    def reset(self, split: str) -> None:
        self.epoch_storage[split]['probas'] = []
        self.epoch_storage[split]['targets'] = []

    def update(self, split: str, probas: torch.Tensor, targets: torch.Tensor) -> None:
        self.epoch_storage[split]['probas'].append(probas.detach().cpu())
        self.epoch_storage[split]['targets'].append(targets.detach().cpu())

    def compute_metrics(self, split: str) -> ProbeMetrics | None:
        probas_list = self.epoch_storage[split]['probas']
        targets_list = self.epoch_storage[split]['targets']
        if not probas_list or not targets_list:
            return None

        probas = torch.cat(probas_list).numpy()
        targets = torch.cat(targets_list).numpy()
        preds = np.argmax(probas, axis=1)

        auroc = None
        try:
            if self.num_classes == 2:
                auroc = float(roc_auc_score(targets, probas[:, 1]))
            else:
                auroc = float(roc_auc_score(targets, probas, average='macro', multi_class='ovr'))
        except ValueError:
            auroc = None

        return ProbeMetrics(
            acc=float(accuracy_score(targets, preds)),
            balacc=float(balanced_accuracy_score(targets, preds)),
            auroc=auroc,
        )
