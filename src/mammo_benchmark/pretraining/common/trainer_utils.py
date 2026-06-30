from __future__ import annotations

import os
from typing import Any, Iterable

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy


def build_output_dir(output_root: str, output_name: str) -> str:
    output_dir = os.path.join(output_root, output_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def build_strategy(num_devices: int, *, find_unused_parameters: bool = False) -> Any:
    return DDPStrategy(find_unused_parameters=find_unused_parameters) if num_devices > 1 else "auto"


def build_logger(output_root: str, output_name: str) -> TensorBoardLogger:
    return TensorBoardLogger(output_root, name=output_name)


def build_trainer(
    *,
    max_epochs: int,
    num_devices: int,
    strategy: Any,
    callbacks: Iterable[Any],
    logger: TensorBoardLogger,
    precision: str = "16-mixed",
    sync_batchnorm: bool = False,
    accumulate_grad_batches: int = 1,
    num_sanity_val_steps: int = 0,
    limit_train_batches: int | float | None = None,
    limit_val_batches: int | float | None = None,
) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        accelerator="auto",
        devices=num_devices,
        strategy=strategy,
        sync_batchnorm=sync_batchnorm,
        precision=precision,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        logger=logger,
        callbacks=list(callbacks),
    )
