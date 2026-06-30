from typing import Dict

import torch
import torch.nn as nn

HEAD_TYPE = "native_linear"


class NativeLinearHead(nn.Module):
    # BatchNorm(affine=False) calibrates feature magnitudes so all backbones share the same head learning rate (MAE paper Appendix A.1).
    input_mode = "native_feature"

    def __init__(self, tasks: Dict[str, int], num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False, eps=1e-6)
        self.heads = nn.ModuleDict({task: nn.Linear(num_features, num_classes) for task, num_classes in tasks.items()})

    def forward(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        feature = self.bn(feature)
        return {task: head(feature) for task, head in self.heads.items()}


def build_head(tasks: Dict[str, int], native_feature_dim: int) -> nn.Module:
    if native_feature_dim <= 0:
        raise ValueError(f"native_linear requires native_feature_dim > 0, got {native_feature_dim}")
    return NativeLinearHead(tasks, num_features=native_feature_dim)
