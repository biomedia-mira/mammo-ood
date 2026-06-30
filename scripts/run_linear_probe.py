#!/usr/bin/env python
"""Run the frozen-backbone linear-probe benchmark from the paper."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from mammo_benchmark.config import DATASET_CONFIG, MODEL_SPECS, resolve_model_specs
from mammo_benchmark.data.datamodule import UniversalMammoDataModule
from mammo_benchmark.models.lightning_module import FoundationLightningModule


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "birads": {
        "dataset": "VinDr-Mammo-breast",
        "task": "Bi-Rads",
        "ood": "BMCD,CBIS-DDSM-breast,CDD-CESM,INbreast,KAU-BCMD,DMID,EMBED",
        "ood_test_only": "EMBED,VinDr-Mammo-breast",
        "epochs": 20,
        "head_lr": 1e-3,
        "batch_alpha": 0.0,
        "early_stop_patience": 8,
    },
    "density": {
        "dataset": "EMBED",
        "task": "Composition",
        "ood": "VinDr-Mammo-breast,BMCD,CBIS-DDSM-breast,CDD-CESM,INbreast,KAU-BCMD,OPTIMAM,DMID",
        "ood_test_only": "EMBED,OPTIMAM,VinDr-Mammo-breast",
        "epochs": 20,
        "head_lr": 1e-3,
        "batch_alpha": 0.0,
        "early_stop_patience": 8,
    },
    "cancer": {
        "dataset": "RSNA-site2",
        "task": "CancerStatus",
        "ood": "VinDr-Mammo-breast,EMBED,CBIS-DDSM-breast,CDD-CESM,CMMD,DBT,DMID,MIAS,MM,NLBS,OPTIMAM",
        "ood_test_only": "OPTIMAM,RSNA-site2",
        "epochs": 15,
        "head_lr": 1e-3,
        "batch_alpha": 0.3,
        "early_stop_patience": 7,
    },
}


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_input_size(args: argparse.Namespace, default_size: tuple[int, int]) -> tuple[int, int]:
    if args.input_height is None and args.input_width is None:
        return default_size
    if args.input_height is None or args.input_width is None:
        raise ValueError("Provide --input_height and --input_width together.")
    return (int(args.input_height), int(args.input_width))


def make_datamodule(args: argparse.Namespace, cfg: dict[str, Any], input_size: tuple[int, int]) -> UniversalMammoDataModule:
    return UniversalMammoDataModule(
        dataset_name=cfg["dataset"],
        task_name=cfg["task"],
        info_root=args.info_root,
        cache_image_size=input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ood_dataset_names=parse_csv_list(cfg["ood"]),
        ood_test_only_datasets=parse_csv_list(cfg["ood_test_only"]),
        batch_alpha=float(cfg["batch_alpha"]),
        seed=args.seed,
    )


def make_trainer(
    args: argparse.Namespace,
    max_epochs: int,
    output_name: str,
    monitor_metric: str,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    logger = TensorBoardLogger(save_dir=args.output_root, name=output_name)
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        filename=f"best-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
    )
    callbacks: list[Any] = [checkpoint_callback, LearningRateMonitor(logging_interval="step")]
    if args.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=monitor_metric,
                mode="max",
                patience=args.early_stop_patience,
                check_on_train_epoch_end=False,
            )
        )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    limit = args.limit_batches if args.limit_batches > 0 else None
    return (
        pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=args.devices if accelerator == "gpu" else 1,
            precision=args.precision,
            logger=logger,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            gradient_clip_val=1.0,
            limit_train_batches=limit,
            limit_val_batches=limit,
            limit_test_batches=limit,
        ),
        checkpoint_callback,
    )


def run_one(args: argparse.Namespace, experiment: str, model_specs: dict[str, dict[str, Any]]) -> None:
    cfg = dict(EXPERIMENTS[experiment])
    if args.max_epochs is not None:
        cfg["epochs"] = args.max_epochs
    if args.head_lr is not None:
        cfg["head_lr"] = args.head_lr
    if args.early_stop_patience is not None:
        cfg["early_stop_patience"] = args.early_stop_patience

    spec = model_specs[args.model]
    checkpoint_path = Path(spec["weight_file"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {args.model}: {checkpoint_path}")

    input_size = resolve_input_size(args, tuple(spec["default_input_size"]))
    datamodule = make_datamodule(args, cfg, input_size)
    datamodule.prepare_data()

    model = FoundationLightningModule(
        model_key=args.model,
        model_spec=spec,
        label_mappings=datamodule.label_mappings,
        backbone_input_size=input_size,
        head_lr=float(cfg["head_lr"]),
        head_wd=args.head_wd,
    )
    safe_task = FoundationLightningModule.sanitize_name(cfg["task"])
    monitor_metric = f"val_{safe_task}_auc"
    run_name = args.output_name or f"{cfg['dataset']}--{cfg['task']}--{args.model}--native_linear--lp--ce"
    if args.experiment == "all":
        run_name = f"{run_name}--{experiment}"

    print("=" * 80)
    print(f"[Run] experiment={experiment} dataset={cfg['dataset']} task={cfg['task']} model={args.model}")
    print(f"[Run] checkpoint={checkpoint_path}")
    print(f"[Run] input_size={input_size} output_root={args.output_root} run_name={run_name}")
    print(f"[Run] epochs={cfg['epochs']} head_lr={cfg['head_lr']} head_wd={args.head_wd} limit_batches={args.limit_batches}")
    print("=" * 80)

    args.early_stop_patience = int(cfg["early_stop_patience"])
    trainer, checkpoint_callback = make_trainer(args, int(cfg["epochs"]), run_name, monitor_metric)
    trainer.fit(model=model, datamodule=datamodule)
    best_ckpt = checkpoint_callback.best_model_path
    if not best_ckpt:
        raise RuntimeError("Training did not produce a best checkpoint.")

    eval_dir = Path(trainer.logger.log_dir) / "eval"
    eval_model = FoundationLightningModule.load_from_checkpoint(
        best_ckpt,
        label_mappings=datamodule.label_mappings,
        model_spec=spec,
        eval_output_dir=str(eval_dir),
    )
    trainer.validate(model=eval_model, datamodule=datamodule)
    trainer.test(model=eval_model, datamodule=datamodule)
    print(f"[Done] best_ckpt={best_ckpt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["all", *EXPERIMENTS.keys()], required=True)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--info_root", default=str(REPO_ROOT / "data" / "classification_data"))
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--adapted_checkpoint_root", default=None)
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs" / "linear_probe"))
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--head_wd", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--limit_batches", type=int, default=0, help="Limit train/val/test batches for quick validation; 0 means full run.")
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    model_specs = resolve_model_specs(args.checkpoint_root, args.adapted_checkpoint_root)
    experiments = EXPERIMENTS.keys() if args.experiment == "all" else [args.experiment]
    for experiment in experiments:
        run_one(args, experiment, model_specs)


if __name__ == "__main__":
    main()
