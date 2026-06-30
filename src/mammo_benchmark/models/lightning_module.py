import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
)

from mammo_benchmark.evaluation.core import (
    EvalBuffer,
    ExpectedScoreSpearman,
    batch_ids,
    eval_scope,
    finalize_task,
    record_batch,
    update_exam_metrics,
    write_predictions_csv,
)
from mammo_benchmark.models.backbones import load_backbone_bundle
from mammo_benchmark.models.heads import build_head


def num_classes_from_mapping(mapping: Dict[Any, int]) -> int:
    return len(set(mapping.values()))


class PositiveClassAveragePrecision(BinaryAveragePrecision):
    """Average precision for the positive class from two-class logits."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.ndim == 2:
            if preds.shape[1] != 2:
                raise ValueError(f"Expected binary logits/probabilities with shape [N,2], got {tuple(preds.shape)}")
            preds = torch.softmax(preds.float(), dim=1)[:, 1]
        return super().update(preds, target)


class PositiveClassAUROC(BinaryAUROC):
    """AUROC for the positive class from two-class logits."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.ndim == 2:
            if preds.shape[1] != 2:
                raise ValueError(f"Expected binary logits/probabilities with shape [N,2], got {tuple(preds.shape)}")
            preds = torch.softmax(preds.float(), dim=1)[:, 1]
        return super().update(preds, target)


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        model_key: str,
        model_spec: Dict[str, Any],
        label_mappings: Dict[str, Dict[Any, int]],
        input_size: Tuple[int, int] = (1024, 768),
    ):
        super().__init__()

        backbone_bundle = load_backbone_bundle(
            model_key=model_key,
            model_spec=model_spec,
            input_size=input_size,
        )
        self.backbone = backbone_bundle.backbone
        self.layer_ids = backbone_bundle.layer_ids
        self.native_feature_dim = backbone_bundle.native_feature_dim

        pretrain_norm = model_spec.get("pretrain_norm", None)
        if pretrain_norm is not None:
            mean = torch.tensor(pretrain_norm["mean"], dtype=torch.float32).view(1, -1, 1, 1)
            std = torch.tensor(pretrain_norm["std"], dtype=torch.float32).view(1, -1, 1, 1)
            self.register_buffer("pretrain_norm_mean", mean, persistent=False)
            self.register_buffer("pretrain_norm_std", std, persistent=False)
            self.apply_pretrain_norm = True
            print(f"[Normalize] model={model_key} pretrain_norm mean={pretrain_norm['mean']} std={pretrain_norm['std']}")
        else:
            self.apply_pretrain_norm = False
            print(f"[Normalize] model={model_key} pretrain_norm=None (backbone handles normalization internally)")

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        if not hasattr(self.backbone, "forward_native_feature"):
            raise ValueError(f"head_type=native_linear requires backbone {model_key} to implement forward_native_feature()")

        tasks = {task: num_classes_from_mapping(mapping) for task, mapping in label_mappings.items()}
        self.classifier = build_head(
            tasks=tasks,
            native_feature_dim=backbone_bundle.native_feature_dim,
        )
        print(
            f"[Head] type=native_linear token_dim={backbone_bundle.token_dim} num_layers={len(self.layer_ids)} "
            f"native_feature_dim={backbone_bundle.native_feature_dim}"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"Expected image-level x as [B,3,H,W], got shape {tuple(x.shape)}")
        if self.apply_pretrain_norm:
            x = (x - self.pretrain_norm_mean) / self.pretrain_norm_std
        with torch.no_grad():
            feature = self.backbone.forward_native_feature(x)  # [B, D_native]
        return self.classifier(feature)


class FoundationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_key: str,
        model_spec: Dict[str, Any],
        label_mappings: Dict[str, Dict[Any, int]],
        backbone_input_size: Tuple[int, int],
        head_lr: float,
        head_wd: float,
        eval_output_dir: str | None = None,
    ):
        super().__init__()
        if len(label_mappings) != 1:
            raise ValueError(
                "FoundationLightningModule is single-task only. "
                f"Got tasks={list(label_mappings.keys())}; run one task per job."
            )
        self.save_hyperparameters(ignore=["label_mappings", "model_spec"])
        self.label_mappings = label_mappings
        self.eval_output_dir = Path(eval_output_dir) if eval_output_dir else None

        self.model = MultiTaskModel(
            model_key=model_key,
            model_spec=model_spec,
            label_mappings=label_mappings,
            input_size=backbone_input_size,
        )

        self.criteria = nn.ModuleDict(
            {
                task: nn.CrossEntropyLoss()
                for task, mapping in label_mappings.items()
            }
        )
        print("[Loss] name=ce")
        self.val_metrics = nn.ModuleDict({task: self.build_metrics(num_classes_from_mapping(mapping)) for task, mapping in label_mappings.items()})
        self.val_predictions = {task: EvalBuffer() for task in label_mappings}
        self.test_metrics = []
        self.test_dataset_names = []
        self.test_label_mappings = []
        self.test_predictions = []

    @staticmethod
    def sanitize_name(value: str) -> str:
        return value.replace(" ", "_").replace("-", "_")

    @staticmethod
    def build_metrics(num_classes: int) -> MetricCollection:
        if num_classes == 2:
            return MetricCollection(
                {
                    "auc": PositiveClassAUROC(),
                    "auprc": PositiveClassAveragePrecision(),
                }
            )
        else:
            return MetricCollection(
                {
                    "auc": MulticlassAUROC(num_classes=num_classes, average="macro"),
                    "spearman": ExpectedScoreSpearman(),
                }
            )

    @staticmethod
    def primary_metric_name(num_classes: int) -> str:
        return "auc"

    @staticmethod
    def metric_dict_to_float(metric_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        result = {}
        for name, value in metric_dict.items():
            value = value.detach().float()
            if value.numel() == 1 and torch.isfinite(value).all():
                result[name] = float(value.cpu().item())
        return result

    @staticmethod
    def build_cosine_warmup_lambda(total_steps: int, warmup_fraction: float, min_lr_ratio: float):
        total_steps = max(1, int(total_steps))
        warmup_steps = max(1, int(total_steps * warmup_fraction))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            if total_steps == warmup_steps:
                return 1.0

            progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return lr_lambda

    def configure_optimizers(self):
        head_params = list(self.model.classifier.parameters())
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_fraction = 0.05
        min_lr_ratio = 0.10
        full_lambda = self.build_cosine_warmup_lambda(total_steps, warmup_fraction, min_lr_ratio)

        param_groups = [{"name": "head", "params": head_params, "lr": self.hparams.head_lr, "weight_decay": self.hparams.head_wd}]
        print(f"[Optim] LP param group: head_lr={self.hparams.head_lr:.2e}, head_wd={self.hparams.head_wd:.2e}")

        lr_lambdas = [full_lambda] * len(param_groups)
        print(f"[Scheduler] Single-stage LP schedule: total_steps={total_steps}, warmup_fraction={warmup_fraction:.2f}")
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def shared_step(self, batch: Dict[str, Any], stage: str, dataloader_idx: int = 0) -> torch.Tensor:
        images = batch["images"].float()
        labels = {task: values.long() for task, values in batch["labels"].items()}
        exam_ids = batch_ids(batch, "exam_id", images.shape[0])
        logits = self.model(images)

        total_loss = torch.zeros((), device=images.device)
        for task, task_logits in logits.items():
            if task not in labels:
                if stage == "test":
                    continue
                raise KeyError(f"Missing label for task={task} during {stage}")
            task_labels = labels[task]
            total_loss = total_loss + self.criteria[task](task_logits, task_labels)

            probs = torch.softmax(task_logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            if stage == "train":
                self.log(
                    f"train_{task}_acc",
                    (preds == task_labels).float().mean(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=images.shape[0],
                )
            elif stage == "val":
                if eval_scope(task) == "exam":
                    record_batch(self.val_predictions[task], task, task_logits, task_labels, exam_ids)
                else:
                    self.val_metrics[task].update(task_logits.detach(), task_labels.detach())
            elif stage == "test":
                task_state = self.test_metrics[dataloader_idx][task]
                pred_buf = self.test_predictions[dataloader_idx][task]
                record_batch(pred_buf, task, task_logits, task_labels, exam_ids)
                if eval_scope(task) == "image":
                    task_state["metrics"].update(task_logits.detach(), task_labels.detach())
                    task_state["confmat"].update(preds.detach(), task_labels.detach())

        self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=(stage != "train"), batch_size=images.shape[0])
        return total_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        if batch_idx == 0 and self.logger:
            images = batch["images"][: min(4, batch["images"].shape[0])].detach().cpu()
            grid = torchvision.utils.make_grid(images, nrow=2, normalize=True)
            self.logger.experiment.add_image("train_images", grid, self.global_step)
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self.shared_step(batch, stage="val")

    def on_validation_epoch_end(self) -> None:
        # Final downstream jobs use a single device; exam-level validation needs
        # complete exam groups before max-probability aggregation.
        for task, metrics in self.val_metrics.items():
            if eval_scope(task) == "exam":
                update_exam_metrics(self.val_predictions[task], metrics, self.device)
            values = self.metric_dict_to_float(metrics.compute())
            primary_metric = self.primary_metric_name(num_classes_from_mapping(self.label_mappings[task]))
            for metric_name, metric_value in values.items():
                self.log(f"val_{self.sanitize_name(task)}_{metric_name}", metric_value, prog_bar=(metric_name == primary_metric), sync_dist=True)
            metrics.reset()
            self.val_predictions[task] = EvalBuffer()

    def on_test_start(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "get_test_dataset_names"):
            self.test_dataset_names = datamodule.get_test_dataset_names()
        else:
            self.test_dataset_names = ["test"]

        self.test_metrics = []
        self.test_label_mappings = []
        self.test_predictions = []
        for dataset_name in self.test_dataset_names:
            if datamodule is not None and dataset_name != getattr(datamodule, "dataset_name", None) and hasattr(datamodule, "ood_label_mappings"):
                task_mappings = datamodule.ood_label_mappings[dataset_name]
            else:
                task_mappings = self.label_mappings
            self.test_label_mappings.append(task_mappings)
            dataset_metrics = {}
            dataset_preds = {}
            for task, mapping in task_mappings.items():
                nc = num_classes_from_mapping(mapping)
                dataset_metrics[task] = {
                    "metrics": self.build_metrics(nc).to(self.device),
                    "confmat": MulticlassConfusionMatrix(num_classes=nc).to(self.device),
                }
                dataset_preds[task] = EvalBuffer()
            self.test_metrics.append(dataset_metrics)
            self.test_predictions.append(dataset_preds)

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        self.shared_step(batch, stage="test", dataloader_idx=dataloader_idx)

    def on_test_epoch_end(self) -> None:
        summary = {"datasets": {}}
        all_auc = []

        for dataset_index, (dataset_name, dataset_metrics, task_mappings) in enumerate(zip(self.test_dataset_names, self.test_metrics, self.test_label_mappings)):
            dataset_summary = {}
            for task, task_state in dataset_metrics.items():
                pred_buf = self.test_predictions[dataset_index][task]
                level = eval_scope(task)
                if not finalize_task(pred_buf, task, task_state["metrics"], task_state["confmat"], self.device):
                    continue

                values = self.metric_dict_to_float(task_state["metrics"].compute())
                confmat = task_state["confmat"].compute().detach().cpu().to(torch.int64)
                num_samples = int(confmat.sum().item())
                if num_samples == 0:
                    continue

                reverse_mapping = {value: key for key, value in task_mappings[task].items()}
                class_names = [str(reverse_mapping.get(i, i)) for i in range(confmat.shape[0])]

                dataset_summary[task] = {
                    "eval_level": level,
                    "logit_aggregation": "max_prob" if level == "exam" else None,
                    "num_samples": num_samples,
                    "metrics": values,
                    "confusion_matrix": confmat.tolist(),
                    "classes": class_names,
                }
                primary_metric = self.primary_metric_name(num_classes_from_mapping(task_mappings[task]))
                for metric_name, metric_value in values.items():
                    self.log(
                        f"test_{self.sanitize_name(dataset_name)}_{self.sanitize_name(task)}_{metric_name}",
                        metric_value,
                        prog_bar=(metric_name == primary_metric),
                        sync_dist=True,
                    )
                if "auc" in values:
                    all_auc.append(values["auc"])

                if self.trainer.is_global_zero:
                    print(
                        f"[Test][{dataset_name}][{task}] "
                        f"AUC={values.get('auc', float('nan')):.4f} "
                        f"PositiveAP={values.get('auprc', float('nan')):.4f} "
                        f"Spearman={values.get('spearman', float('nan')):.4f}"
                    )

            if dataset_summary:
                summary["datasets"][dataset_name] = dataset_summary

        avg_auc = sum(all_auc) / len(all_auc) if all_auc else 0.0
        self.log("test_avg_auc", avg_auc, prog_bar=True, sync_dist=True)

        if self.trainer.is_global_zero:
            output_dir = self.resolve_eval_output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = output_dir / "test_metrics_summary.json"
            import re
            summary_str = json.dumps(summary, indent=2)
            summary_str = re.sub(r'\[\s*((?:\d+,\s*)*\d+)\s*\]', lambda m: '[' + re.sub(r'\s+', '', m.group(1)).replace(',', ', ') + ']', summary_str)
            with summary_path.open("w", encoding="utf-8") as handle:
                handle.write(summary_str)
            print(f"[Test] Saved evaluation summary to {summary_path}")

            for dataset_name, dataset_preds in zip(self.test_dataset_names, self.test_predictions):
                for task, buf in dataset_preds.items():
                    csv_path = output_dir / f"predictions_{self.sanitize_name(dataset_name)}_{self.sanitize_name(task)}.csv"
                    write_predictions_csv(csv_path, buf)
            print(f"[Test] Saved per-(dataset,task) prediction CSVs to {output_dir}")

    def resolve_eval_output_dir(self) -> Path:
        if self.eval_output_dir is not None:
            return self.eval_output_dir
        if self.logger is not None and hasattr(self.logger, "log_dir") and self.logger.log_dir:
            return Path(self.logger.log_dir) / "eval"
        return Path.cwd() / "eval"
