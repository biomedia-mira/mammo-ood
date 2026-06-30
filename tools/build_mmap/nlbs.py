"""Build NLBS mmap + metadata.parquet at exam-level granularity.

Source:
  - /path/to/Mammo/NLBS/NLBSD_Metadata.csv
  - /path/to/Mammo/NLBS/pngs/<H>x<W>/<cat>/<pid>/<lat>/<view>/<file>.png

File Path format: '<cat>/<pid>/<lat>/<view>/<file>.dcm'. Cancer flag in CSV
is patient/screening-event level (per README) — not per-image abnormality.

Patient-level split 70/10/20 by patient_id with seed=42. exam_id = patient_id
(no exam date column in CSV; all values are NaN).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.mmap import MMapBuilder, make_patient_split

DATASET = "NLBS"
SRC_CSV = Path("/path/to/Mammo/NLBS/NLBSD_Metadata.csv")
SRC_PNG_ROOT_FMT = "/path/to/Mammo/NLBS/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/NLBS")
DEFAULT_IMG_SIZE = (1024, 768)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.1, 0.2)

CATEGORY_NORMALIZE = {"abnormal": "abnormal", "normal": "normal", "False Positive": "false_positive"}


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    df = pd.read_csv(SRC_CSV)
    df["File Path"] = df["File Path"].astype(str).str.replace("\\", "/", regex=False).str.strip("/")
    parts = df["File Path"].str.split("/")
    df["cat_raw"] = parts.str[0].str.strip()
    df["pid"] = parts.str[1].str.strip()
    df["lat_dir"] = parts.str[2].str.strip()
    df["view_dir"] = parts.str[3].str.strip()
    df["fname"] = parts.str[4].str.strip()
    df["lat"] = df["Image Laterality"].astype(str).str.upper().str.strip()
    df["view"] = df["View Position"].astype(str).str.upper().str.strip()
    df["cat"] = df["cat_raw"].map(CATEGORY_NORMALIZE)
    df = df[(df["pid"] != "") & df["lat"].isin(["L", "R"]) & df["view"].isin(["CC", "MLO"]) & df["cat"].notna()].copy()
    print(f"[{DATASET}] {len(df)} rows after filter")

    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    patient_split = make_patient_split(df["pid"].unique(), seed=SPLIT_SEED, ratios=SPLIT_RATIOS)

    image_paths: List[str] = []
    rows: List[Dict] = []
    missing: List[str] = []

    for r in df.itertuples(index=False):
        png_name = r.fname.replace(".dcm", ".png")
        png_path = src_png_root / r.cat / r.pid / r.lat_dir / r.view_dir / png_name
        if not png_path.is_file():
            missing.append(str(png_path))
            continue

        cancer_status = "Cancer" if int(r.Cancer) == 1 else "Non-cancer"
        age = None
        if pd.notna(getattr(r, "Age", None)):
            try:
                age = int(getattr(r, "Age"))
            except (TypeError, ValueError):
                age = None

        sample_id = f"{r.cat}-{r.pid}-{r.lat}-{r.view}-{r.fname.replace('.dcm','')}"
        image_paths.append(str(png_path))
        rows.append({
            "mmap_idx": -1,
            "sample_name": sample_id,
            "exam_id": r.pid,         # no exam date in CSV; treat patient as exam
            "patient_id": r.pid,
            "side": r.lat,
            "view": r.view,
            "split": patient_split[r.pid],
            "bi_rads": None,
            "density": None,
            "cancer_status": cancer_status,
            "finding": None,
            "manufacturer": None,     # GE Senograph Essential per README, but no per-image col
            "manufacturer_model": None,
            "age": age,
            "study_date": None,       # all NaN in source CSV
            "race_ethnicity": None,
            "site_id": None,
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
