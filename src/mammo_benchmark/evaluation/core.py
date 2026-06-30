from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.stats import spearmanr
from torchmetrics import Metric


EXAM_LEVEL_TASKS = {"Bi-Rads", "CancerStatus"}
PROB_EPS = 1e-12


class ExpectedScoreSpearman(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        probs = torch.softmax(preds, dim=-1)
        weights = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
        self.preds.append((probs * weights).sum(dim=-1))
        self.target.append(target.float())

    def compute(self):
        if isinstance(self.preds, torch.Tensor):
            preds = self.preds
            target = self.target
        else:
            if len(self.preds) == 0:
                return torch.tensor(float("nan"), device=self.device)
            preds = torch.cat(self.preds)
            target = torch.cat(self.target)

        if preds.numel() < 2:
            return torch.tensor(float("nan"), device=preds.device)

        preds_np = preds.cpu().numpy()
        target_np = target.cpu().numpy()
        if np.all(preds_np == preds_np[0]) or np.all(target_np == target_np[0]):
            return torch.tensor(float("nan"), device=preds.device)

        corr, _ = spearmanr(preds_np, target_np)
        return torch.tensor(corr, device=preds.device)


@dataclass
class EvalBuffer:
    logits: list[torch.Tensor] = field(default_factory=list)
    probs: list[torch.Tensor] = field(default_factory=list)
    labels: list[torch.Tensor] = field(default_factory=list)
    exam_ids: list[str] = field(default_factory=list)
    export_probs: Optional[torch.Tensor] = None
    export_labels: Optional[torch.Tensor] = None
    export_exam_ids: Optional[list[str]] = None
    export_num_images: Optional[list[int]] = None


def eval_scope(task: str) -> str:
    return "exam" if task in EXAM_LEVEL_TASKS else "image"


def batch_ids(batch: Dict[str, Any], key: str, batch_size: int) -> list[str]:
    values = batch.get(key)
    if values is None:
        return [""] * batch_size
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values]
    if torch.is_tensor(values):
        flat = values.detach().cpu().view(-1).tolist()
        return [str(value) for value in flat]
    return [str(values)] * batch_size


def record_batch(buffer: EvalBuffer, task: str, logits: torch.Tensor, labels: torch.Tensor, exam_ids: list[str]) -> None:
    buffer.labels.append(labels.detach().cpu())
    buffer.exam_ids.extend(exam_ids)
    if eval_scope(task) == "exam":
        buffer.logits.append(logits.detach().cpu())
    else:
        buffer.probs.append(torch.softmax(logits.detach(), dim=1).cpu())


def max_prob_log_probs(image_logits: torch.Tensor) -> torch.Tensor:
    """Aggregate image logits into log P(exam = class) via max probability pooling."""
    probs = torch.softmax(image_logits.float(), dim=1)
    exam_probs = probs.max(dim=0).values
    exam_probs = exam_probs / exam_probs.sum().clamp_min(PROB_EPS)
    return exam_probs.clamp_min(PROB_EPS).log()


def aggregate_exams(buffer: EvalBuffer) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
    if not buffer.logits:
        return torch.empty(0), torch.empty(0, dtype=torch.long), [], []
    logits = torch.cat(buffer.logits, dim=0)
    labels = torch.cat(buffer.labels, dim=0).long()

    groups: Dict[str, list[int]] = {}
    for index, exam_id in enumerate(buffer.exam_ids):
        groups.setdefault(exam_id, []).append(index)

    exam_logits = []
    exam_labels = []
    exam_ids = []
    num_images = []
    for exam_id, indices in groups.items():
        idx = torch.tensor(indices, dtype=torch.long)
        exam_logits.append(max_prob_log_probs(logits.index_select(0, idx)))
        exam_labels.append(labels.index_select(0, idx).max())
        exam_ids.append(exam_id)
        num_images.append(len(indices))

    return torch.stack(exam_logits, dim=0), torch.stack(exam_labels, dim=0), exam_ids, num_images


def update_exam_metrics(buffer: EvalBuffer, metrics: Any, device: torch.device) -> None:
    logits, labels, _, _ = aggregate_exams(buffer)
    if labels.numel() > 0:
        metrics.update(logits.to(device), labels.to(device))


def finalize_task(
    buffer: EvalBuffer,
    task: str,
    metrics: Any,
    confmat: Any,
    device: torch.device,
) -> bool:
    if eval_scope(task) == "exam":
        logits, labels, exam_ids, num_images = aggregate_exams(buffer)
        if labels.numel() == 0:
            return False
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        metrics.update(logits.to(device), labels.to(device))
        confmat.update(preds.to(device), labels.to(device))
        buffer.export_probs = probs
        buffer.export_labels = labels
        buffer.export_exam_ids = exam_ids
        buffer.export_num_images = num_images
        return True

    if not buffer.probs:
        return False
    labels = torch.cat(buffer.labels, dim=0).long()
    buffer.export_probs = torch.cat(buffer.probs, dim=0)
    buffer.export_labels = labels
    buffer.export_exam_ids = list(buffer.exam_ids)
    buffer.export_num_images = [1] * int(labels.numel())
    return True


def write_predictions_csv(csv_path: Path, buffer: EvalBuffer) -> None:
    if buffer.export_probs is None or buffer.export_labels is None:
        return
    probs = buffer.export_probs.numpy()
    labels = buffer.export_labels.numpy().astype(np.int64)
    exam_ids = buffer.export_exam_ids or [""] * len(labels)
    num_images = buffer.export_num_images or [1] * len(labels)
    header = ",".join(["exam_id", "num_images"] + [f"prob_{i}" for i in range(probs.shape[1])] + ["label"])
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for i in range(labels.shape[0]):
            prob_str = ",".join(f"{p:.6e}" for p in probs[i])
            handle.write(f"{exam_ids[i]},{int(num_images[i])},{prob_str},{int(labels[i])}\n")
