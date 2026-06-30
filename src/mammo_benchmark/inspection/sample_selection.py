"""Utilities for fixed model-inspection plotting samples."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_SAMPLE_COLUMNS = {"study_id", "image_id"}


def validate_sample_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Inspection sample CSV not found: {path}")
    sample = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_SAMPLE_COLUMNS.difference(sample.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    sample["study_id"] = sample["study_id"].astype(str)
    sample["image_id"] = sample["image_id"].astype(str)
    return sample.drop_duplicates(["study_id", "image_id"]).reset_index(drop=True)

