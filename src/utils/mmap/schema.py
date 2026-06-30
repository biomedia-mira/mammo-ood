"""Metadata schema for mmap-backed mammography datasets.

Each dataset produces a `metadata.parquet` file with one row per image. The
row schema is fixed across datasets so the downstream loader can treat them
uniformly. Columns may be NaN where the dataset does not provide that field.

The image pixel data lives in a separate `images_<H>x<W>.npy` mmap, indexed by
the `mmap_idx` column.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

# Column name -> pandas dtype. `object` is used for list-valued columns
# (e.g. multi-label finding). `string` columns can be NaN.
METADATA_COLUMNS: Dict[str, str] = {
    "mmap_idx": "int64",
    "sample_name": "string",
    "exam_id": "string",
    "patient_id": "string",
    "side": "string",                # 'L' / 'R' or NaN
    "view": "string",                # 'CC' / 'MLO' or NaN
    "split": "string",               # 'Train' / 'Eval' / 'Test'
    "bi_rads": "string",             # 'Bi-Rads 0'..'Bi-Rads 5' or NaN
    "density": "string",             # 'Level A'..'Level D' or NaN
    "cancer_status": "string",       # 'Cancer' / 'Non-cancer' or NaN
    "finding": "object",             # list[str] or NaN
    "manufacturer": "string",        # NaN if unknown (e.g., 'HOLOGIC, Inc.')
    "manufacturer_model": "string",  # NaN if unknown (e.g., 'Selenia Dimensions')
    "age": "Int64",                  # nullable integer, years; NaN if unknown
    "study_date": "string",          # ISO 'YYYY-MM-DD' or NaN
    "race_ethnicity": "string",      # NaN if unknown (e.g., EMBED 'White')
    "site_id": "string",             # NaN if unknown (e.g., RSNA site_id, OPTIMAM site)
}

ALLOWED_SIDES = {"L", "R", None}
ALLOWED_VIEWS = {"CC", "MLO", None}
ALLOWED_SPLITS = {"Train", "Eval", "Test"}
ALLOWED_BIRADS = {None, *(f"Bi-Rads {i}" for i in range(6))}
ALLOWED_DENSITY = {None, "Level A", "Level B", "Level C", "Level D"}
ALLOWED_CANCER = {None, "Cancer", "Non-cancer"}


def empty_metadata_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical schema and dtypes."""
    df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in METADATA_COLUMNS.items()})
    return df


def rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a metadata DataFrame from a list of row dicts.

    Missing columns are filled with NaN. Extra columns raise. dtypes are
    coerced to match the schema.
    """
    df = pd.DataFrame(rows)
    extras = set(df.columns) - set(METADATA_COLUMNS)
    if extras:
        raise ValueError(f"Unexpected metadata columns: {sorted(extras)}")
    for col, dtype in METADATA_COLUMNS.items():
        if col not in df.columns:
            df[col] = pd.NA if dtype == "object" else pd.Series([pd.NA] * len(df), dtype=dtype)
    df = df[list(METADATA_COLUMNS)]
    for col, dtype in METADATA_COLUMNS.items():
        if dtype == "object":
            continue
        df[col] = df[col].astype(dtype)
    return df


def validate_metadata(df: pd.DataFrame, n_images: int) -> None:
    """Validate a metadata DataFrame against the schema.

    Raises ValueError on any inconsistency.
    """
    missing_cols = set(METADATA_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing metadata columns: {sorted(missing_cols)}")

    if len(df) != n_images:
        raise ValueError(f"Metadata rows ({len(df)}) != mmap rows ({n_images})")

    if df["mmap_idx"].duplicated().any():
        dupes = df.loc[df["mmap_idx"].duplicated(), "mmap_idx"].unique()[:10]
        raise ValueError(f"Duplicate mmap_idx values: {list(dupes)}")

    expected_indices = set(range(n_images))
    actual_indices = set(df["mmap_idx"].astype(int))
    if expected_indices != actual_indices:
        missing = sorted(expected_indices - actual_indices)[:10]
        extra = sorted(actual_indices - expected_indices)[:10]
        raise ValueError(
            f"mmap_idx must cover [0..{n_images-1}] exactly. "
            f"Missing: {missing}, extra: {extra}"
        )

    def _check_vocab(col: str, allowed: set) -> None:
        bad = set(df[col].dropna().unique()) - {v for v in allowed if v is not None}
        if bad:
            raise ValueError(f"Invalid {col} values: {sorted(bad)}")

    _check_vocab("side", ALLOWED_SIDES)
    _check_vocab("view", ALLOWED_VIEWS)
    _check_vocab("split", ALLOWED_SPLITS)
    _check_vocab("bi_rads", ALLOWED_BIRADS)
    _check_vocab("density", ALLOWED_DENSITY)
    _check_vocab("cancer_status", ALLOWED_CANCER)
