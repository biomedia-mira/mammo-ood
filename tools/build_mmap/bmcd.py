"""Build BMCD mmap + metadata.parquet at exam-level granularity (RECENT only).

Source:
  - /path/to/Mammo/BMCD/Description.xlsx (Normal_cases + Suspicious_cases)
  - /path/to/Mammo/BMCD/pngs/<H>x<W>/{Normal_cases,Suspicious_cases}/<folder>/{CC,MLO}_recent.png

Each case folder = 1 patient = 1 breast (per README) with 4 images: CC_prior,
CC_recent, MLO_prior, MLO_recent. Per the README the prior was acquired ~2.2
years before the recent and was BI-RADS 1 or 2 in suspicious cases — but the
xlsx only stores the RECENT BI-RADS. To keep labels clean we only use the 2
recent images per case (~25% of total but with reliable labels).

Patient-level split 70/10/20 by case folder with seed=42. Each "patient"
contributes exactly one (CC_recent, MLO_recent) pair to its split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.mmap import MMapBuilder, make_patient_split

DATASET = "BMCD"
SRC_XLSX = Path("/path/to/Mammo/BMCD/Description.xlsx")
SRC_PNG_ROOT_FMT = "/path/to/Mammo/BMCD/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/BMCD")
DEFAULT_IMG_SIZE = (1024, 768)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.1, 0.2)

DENSITY_MAP = {"a": "Level A", "b": "Level B", "c": "Level C", "d": "Level D"}
BIRADS_MAP = {
    "0": "Bi-Rads 0", "1": "Bi-Rads 1", "2": "Bi-Rads 2", "3": "Bi-Rads 3",
    "4a": "Bi-Rads 4", "4b": "Bi-Rads 4", "4c": "Bi-Rads 4", "5": "Bi-Rads 5",
}


def _load_cases() -> pd.DataFrame:
    norm = pd.read_excel(SRC_XLSX, sheet_name="Normal_cases", header=2).dropna(subset=["Folder #"])
    sus = pd.read_excel(SRC_XLSX, sheet_name="Suspicious_cases", header=2).dropna(subset=["Folder #"])
    norm["case_type"] = "Normal_cases"
    sus["case_type"] = "Suspicious_cases"
    df = pd.concat([norm, sus], ignore_index=True)
    df["folder"] = df["Folder #"].astype(int).astype(str)
    df["age"] = pd.to_numeric(df["Age (at the time of the recent mammogram)"], errors="coerce")
    # Excel reads numeric BI-RADS values as float (e.g. "1.0", "2.0"); strip the
    # ".0" suffix so the map below catches them.
    df["density_raw"] = df["BI-RADS categories for breast density"].astype(str).str.strip().str.lower()
    df["birads_raw"] = (
        df["BI-RADS categories for classification "].astype(str).str.strip().str.lower()
        .str.replace(r"\.0$", "", regex=True)
    )
    df["side"] = df["Breast (Right/Left)"].astype(str).str.strip().str.upper().map({"RIGHT": "R", "LEFT": "L"})
    return df


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    df = _load_cases()
    print(f"[{DATASET}] {len(df)} cases (Normal+Suspicious)")

    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    case_keys = [f"{r.case_type}_{r.folder}" for r in df.itertuples(index=False)]
    patient_split = make_patient_split(case_keys, seed=SPLIT_SEED, ratios=SPLIT_RATIOS)

    image_paths: List[str] = []
    rows: List[Dict] = []
    missing: List[str] = []

    for r in df.itertuples(index=False):
        case_key = f"{r.case_type}_{r.folder}"
        density = DENSITY_MAP.get(r.density_raw)
        bi_rads = BIRADS_MAP.get(r.birads_raw)

        for view in ("CC", "MLO"):
            png_path = src_png_root / r.case_type / r.folder / f"{view}_recent.png"
            if not png_path.is_file():
                missing.append(str(png_path))
                continue
            image_paths.append(str(png_path))
            rows.append({
                "mmap_idx": -1,
                "sample_name": f"{case_key}_{view}_recent",
                "exam_id": case_key,         # 1 case = 1 patient = 1 breast
                "patient_id": case_key,
                "side": r.side,
                "view": view,
                "split": patient_split[case_key],
                "bi_rads": bi_rads,
                "density": density,
                "cancer_status": None,
                "finding": None,
                "manufacturer": None,
                "manufacturer_model": None,
                "age": int(r.age) if pd.notna(r.age) else None,
                "study_date": None,
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
        builder.verify(sample_size=16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Build {DATASET} mmap (recent images only)")
    parser.add_argument("--img-size", nargs=2, type=int, default=list(DEFAULT_IMG_SIZE), metavar=("H", "W"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    main(tuple(args.img_size), args.out_root, args.num_workers, args.force_rebuild)
