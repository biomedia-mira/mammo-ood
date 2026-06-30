"""Build DBT (BCS-DBT) mmap + metadata.parquet at exam-level granularity.

Source:
  - /path/to/Mammo/DBT/metadata/BCS-DBT-labels-{train,validation-PHASE-2,test-PHASE-2}*.csv
  - /path/to/Mammo/DBT/metadata/BCS-DBT-file-paths-*.csv (for image paths)
  - /path/to/Mammo/DBT/pngs/<H>x<W>/{train,validation,test}/<PatientID>_<StudyUID>_<view>.png

`View` column encodes both side and view as one of: lcc, lmlo, rcc, rmlo. We
split into (side, view). exam_id = StudyUID; patient_id = PatientID.
Patient-level split 70/10/20 by PatientID with seed=42 (this overrides the
source train/val/test partitioning so that all 3 partitions get represented
across our Train/Eval/Test).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.mmap import MMapBuilder, make_patient_split

DATASET = "DBT"
SRC_META_ROOT = Path("/path/to/Mammo/DBT/metadata")
SRC_PNG_ROOT_FMT = "/path/to/Mammo/DBT/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/DBT")
DEFAULT_IMG_SIZE = (1024, 768)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.1, 0.2)

SPLIT_FILES = {
    "train": ("BCS-DBT-labels-train-v2.csv", "BCS-DBT-file-paths-train-v2.csv"),
    "validation": ("BCS-DBT-labels-validation-PHASE-2-Jan-2024.csv", "BCS-DBT-file-paths-validation-v2.csv"),
    "test": ("BCS-DBT-labels-test-PHASE-2.csv", "BCS-DBT-file-paths-test-v2.csv"),
}

VIEW_DECODE = {
    "lcc": ("L", "CC"), "lmlo": ("L", "MLO"),
    "rcc": ("R", "CC"), "rmlo": ("R", "MLO"),
}


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    all_rows: List[pd.DataFrame] = []
    for source_split, (lbl, _paths) in SPLIT_FILES.items():
        df = pd.read_csv(SRC_META_ROOT / lbl)
        df["source_split"] = source_split
        all_rows.append(df)
    df = pd.concat(all_rows, ignore_index=True)
    print(f"[{DATASET}] {len(df)} rows total")

    patient_split = make_patient_split(df["PatientID"].astype(str).unique(), seed=SPLIT_SEED, ratios=SPLIT_RATIOS)

    image_paths: List[str] = []
    rows: List[Dict] = []
    missing: List[str] = []
    bad_views: List[str] = []

    for r in df.itertuples(index=False):
        view_raw = str(r.View).strip().lower()
        if view_raw not in VIEW_DECODE:
            bad_views.append(view_raw)
            continue
        side, view = VIEW_DECODE[view_raw]

        patient_id = str(r.PatientID)
        study_uid = str(r.StudyUID)
        sample_id = f"{patient_id}_{study_uid}_{view_raw}"

        png_path = src_png_root / r.source_split / f"{sample_id}.png"
        if not png_path.is_file():
            missing.append(str(png_path))
            continue

        if int(r.Cancer) == 1:
            cancer_status = "Cancer"
        else:
            cancer_status = "Non-cancer"

        image_paths.append(str(png_path))
        rows.append({
            "mmap_idx": -1,
            "sample_name": sample_id,
            "exam_id": study_uid,
            "patient_id": patient_id,
            "side": side,
            "view": view,
            "split": patient_split[patient_id],
            "bi_rads": None,
            "density": None,
            "cancer_status": cancer_status,
            "finding": None,
            "manufacturer": None,
            "manufacturer_model": None,
            "age": None,
            "study_date": None,
            "race_ethnicity": None,
            "site_id": None,
        })

    if bad_views:
        print(f"[{DATASET}] WARNING: {len(bad_views)} unparseable View values")
    if missing:
        print(f"[{DATASET}] WARNING: {len(missing)} PNGs missing (skipped)")
        for p in missing[:5]:
            print(f"  {p}")
    if not image_paths:
        raise RuntimeError("No images resolved.")
    return image_paths, rows


def main(
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
    out_root: Path = DEFAULT_OUT_ROOT,
    num_workers: Optional[int] = None,
    force_rebuild: bool = False,
) -> None:
    h, w = img_size
    out_dir = Path(out_root) / f"mmap_{h}x{w}"
    mmap_path = out_dir / f"images_{h}x{w}.npy"
    metadata_path = out_dir / "metadata.parquet"

    print(f"[{DATASET}] target {img_size}, out {out_dir}")
    image_paths, rows = build_metadata_rows(img_size)
    print(f"[{DATASET}] resolved {len(image_paths)} images")

    builder = MMapBuilder(mmap_path=mmap_path, metadata_path=metadata_path, img_size=img_size)
    result = builder.build(
        image_paths=image_paths,
        metadata_rows=rows,
        num_workers=num_workers,
        force_rebuild=force_rebuild,
    )
    print(f"[{DATASET}] done: total={result.n_total} written={result.n_written} "
          f"skipped={result.n_skipped} failed={result.n_failed}")
    if result.n_failed == 0:
        builder.verify(sample_size=32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Build {DATASET} mmap")
    parser.add_argument("--img-size", nargs=2, type=int, default=list(DEFAULT_IMG_SIZE), metavar=("H", "W"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    main(tuple(args.img_size), args.out_root, args.num_workers, args.force_rebuild)
