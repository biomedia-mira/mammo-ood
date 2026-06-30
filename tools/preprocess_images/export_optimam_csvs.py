from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from optimam_utils import (
    OPTIMAM_CSV_ROOT,
    build_optimam_classification_dataframe,
    build_optimam_dataframe,
    build_optimam_marks_dataframe,
    filter_optimam_for_presentation,
)


def build_patient_split_map(
    df: pd.DataFrame,
    *,
    patient_col: str = "client_id",
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, str]:
    patient_ids = df[patient_col].dropna().astype(str).drop_duplicates().tolist()
    patient_ids = list(np.random.default_rng(seed).permutation(patient_ids))

    n_patients = len(patient_ids)
    n_train = int(n_patients * train_frac)
    n_val = int(n_patients * val_frac)
    n_test = max(n_patients - n_train - n_val, 0)

    split_map: dict[str, str] = {}
    for patient_id in patient_ids[:n_train]:
        split_map[patient_id] = "train"
    for patient_id in patient_ids[n_train:n_train + n_val]:
        split_map[patient_id] = "val"
    for patient_id in patient_ids[n_train + n_val:n_train + n_val + n_test]:
        split_map[patient_id] = "test"
    return split_map


def apply_patient_split(
    df: pd.DataFrame,
    split_map: dict[str, str],
    *,
    patient_col: str = "client_id",
) -> pd.DataFrame:
    df_out = df.copy()
    if df_out.empty:
        df_out["split"] = pd.Series(dtype="object")
        return df_out
    df_out["split"] = df_out[patient_col].astype(str).map(split_map)
    return df_out


def resolve_output_path(base_path: Path, *, is_full_export: bool, row_count: int) -> Path:
    if is_full_export:
        return base_path
    return base_path.with_name(f"{base_path.stem}_sample_{row_count}rows{base_path.suffix}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OPTIMAM master, classification, and mark CSVs.")
    parser.add_argument("--limit-patients", type=int, default=None, help="Limit to the first N patients.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patient-level splits.")
    parser.add_argument("--output-dir", type=Path, default=OPTIMAM_CSV_ROOT)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    is_full_export = args.limit_patients is None

    df_parsed = build_optimam_dataframe(limit_patients=args.limit_patients, show_progress=True)
    df_master = filter_optimam_for_presentation(df_parsed)

    df_classification = build_optimam_classification_dataframe(df_parsed)
    classification_split_map = build_patient_split_map(
        df_classification,
        train_frac=0.7,
        val_frac=0.1,
        test_frac=0.2,
        seed=args.seed,
    )
    df_classification = apply_patient_split(df_classification, classification_split_map)

    master_path = resolve_output_path(output_dir / "optimam.csv", is_full_export=is_full_export, row_count=len(df_master))
    classification_path = resolve_output_path(
        output_dir / "optimam_classification.csv",
        is_full_export=is_full_export,
        row_count=len(df_classification),
    )

    save_csv(df_master, master_path)
    save_csv(df_classification, classification_path)
    print(f"Saved master table: {master_path} ({len(df_master):,} rows)")
    print(f"Saved classification table: {classification_path} ({len(df_classification):,} rows)")
    if "split" in df_classification.columns:
        print("Classification split patients:", df_classification.groupby("split")["client_id"].nunique().to_dict())

    df_marks = build_optimam_marks_dataframe(df_master)
    marks_path = resolve_output_path(output_dir / "optimam_marks.csv", is_full_export=is_full_export, row_count=len(df_marks))
    save_csv(df_marks, marks_path)
    print(f"Saved marks table: {marks_path} ({len(df_marks):,} rows)")


if __name__ == "__main__":
    main()
