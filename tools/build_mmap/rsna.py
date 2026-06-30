"""Build RSNA mmap + metadata.parquet at exam-level granularity.

Source:
  - /path/to/Mammo/RSNA/train.csv
  - /path/to/Mammo/RSNA/pngs/<H>x<W>/<patient_id>/<image_id>.png

Output:
  - /path/to/Mammo/classification_data/RSNA/mmap_<H>x<W>/

Patient-level split 70/10/20 by patient_id with seed=42 (matches existing
notebook). Cancer label is per-image in CSV but empirically breast-level
(CC+MLO of same patient+side share 100%); aggregation is left for downstream.
exam_id = patient_id (1 visit per patient).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.mmap import MMapBuilder, make_patient_split

DATASET = "RSNA"
SRC_CSV = Path("/path/to/Mammo/RSNA/train.csv")
SRC_PNG_ROOT_FMT = "/path/to/Mammo/RSNA/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/RSNA")
DEFAULT_IMG_SIZE = (1024, 768)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.1, 0.2)


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    df = pd.read_csv(SRC_CSV)
    df["patient_id"] = df["patient_id"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    if df["image_id"].duplicated().any():
        dupes = df.loc[df["image_id"].duplicated(), "image_id"].unique()[:10]
        raise ValueError(f"Duplicate image_id in RSNA train.csv: {list(dupes)}")
    print(f"[{DATASET}] {len(df)} images, {df['patient_id'].nunique()} patients")

    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    patient_split = make_patient_split(df["patient_id"].unique(), seed=SPLIT_SEED, ratios=SPLIT_RATIOS)

    image_paths: List[str] = []
    rows: List[Dict] = []
    missing: List[str] = []
    df = df.sort_values(["patient_id", "image_id"], kind="stable").reset_index(drop=True)

    for r in df.itertuples(index=False):
        png_path = src_png_root / str(r.patient_id) / f"{r.image_id}.png"
        if not png_path.is_file():
            missing.append(str(png_path))
            continue

        side = str(r.laterality).strip().upper() if pd.notna(r.laterality) else None
        if side not in ("L", "R"):
            side = None
        view = str(r.view).strip().upper() if pd.notna(r.view) else None
        if view not in ("CC", "MLO"):
            view = None

        cancer_status = "Cancer" if int(r.cancer) == 1 else "Non-cancer"

        density_raw = str(r.density).strip().upper() if pd.notna(r.density) else None
        density = f"Level {density_raw}" if density_raw in ("A", "B", "C", "D") else None

        bi_rads = None
        if pd.notna(r.BIRADS):
            try:
                bi_rads_int = int(r.BIRADS)
                if 0 <= bi_rads_int <= 5:
                    bi_rads = f"Bi-Rads {bi_rads_int}"
            except (TypeError, ValueError):
                bi_rads = None

        age = None
        if pd.notna(r.age):
            try:
                age = int(r.age)
            except (TypeError, ValueError):
                age = None

        machine_id = getattr(r, "machine_id", None)
        site_id = str(int(getattr(r, "site_id"))) if pd.notna(getattr(r, "site_id", None)) else None

        image_paths.append(str(png_path))
        rows.append({
            "mmap_idx": -1,
            "sample_name": str(r.image_id),
            "exam_id": str(r.patient_id),  # 1 visit/patient typical
            "patient_id": str(r.patient_id),
            "side": side,
            "view": view,
            "split": patient_split[str(r.patient_id)],
            "bi_rads": bi_rads,
            "density": density,
            "cancer_status": cancer_status,
            "finding": None,
            "manufacturer": None,         # RSNA CSV has machine_id (numeric); manufacturer string not exposed
            "manufacturer_model": str(machine_id) if pd.notna(machine_id) else None,
            "age": age,
            "study_date": None,
            "race_ethnicity": None,
            "site_id": site_id,
        })

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
        builder.verify(sample_size=64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Build {DATASET} mmap")
    parser.add_argument("--img-size", nargs=2, type=int, default=list(DEFAULT_IMG_SIZE), metavar=("H", "W"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    main(tuple(args.img_size), args.out_root, args.num_workers, args.force_rebuild)
