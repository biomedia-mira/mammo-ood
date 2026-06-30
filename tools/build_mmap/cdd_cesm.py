"""Build CDD-CESM mmap + metadata.parquet at exam-level granularity.

Source:
  - /path/to/Mammo/CDD-CESM/Radiology-manual-annotations.xlsx ('all' sheet)
  - /path/to/Mammo/CDD-CESM/pngs/<H>x<W>/<Image_name>.png

Image_name format 'P{n}_{L|R}_DM_{CC|MLO}'. Each row is one image; we keep
only DM (digital mammography) rows. exam_id = (Patient_ID, Side); patient
split 70/10/20 by Patient_ID.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.mmap import MMapBuilder, make_patient_split

DATASET = "CDD-CESM"
SRC_XLSX = Path("/path/to/Mammo/CDD-CESM/Radiology-manual-annotations.xlsx")
SRC_PNG_ROOT_FMT = "/path/to/Mammo/CDD-CESM/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/CDD-CESM")
DEFAULT_IMG_SIZE = (1024, 768)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.1, 0.2)

PATHOLOGY_TO_CANCER = {"Normal": "Non-cancer", "Benign": "Non-cancer", "Malignant": "Cancer"}


def _parse_birads(value) -> Optional[str]:
    if pd.isna(value):
        return None
    raw = str(value).strip()
    if raw in ("", "_", "nan", "None"):
        return None
    try:
        m = max(int(float(p)) for p in raw.split("$"))
    except ValueError:
        return None
    # Match notebook: only keep BI-RADS 1..5 (BI-RADS 0 dropped).
    return f"Bi-Rads {m}" if 1 <= m <= 5 else None


def _parse_image_name(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    m = re.match(r"^(P\d+)_([LR])_DM_(CC|MLO)$", name.strip())
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    df = pd.read_excel(SRC_XLSX, sheet_name="all")
    df["Image_name"] = df["Image_name"].astype(str).str.strip()
    df = df[df["Image_name"].str.contains(r"_DM_(CC|MLO)$", regex=True)].copy()
    # Rename columns with spaces/special chars to safe identifiers so itertuples
    # gives clean attribute access (column names with spaces become '_N' otherwise).
    df = df.rename(columns={
        "Breast density (ACR)": "density_raw",
        "Pathology Classification/ Follow up": "pathology_raw",
    })
    print(f"[{DATASET}] {len(df)} DM rows")

    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    parsed = df["Image_name"].apply(_parse_image_name)
    df["patient_id"], df["side"], df["view"] = zip(*parsed)
    df = df[df["patient_id"].notna()].reset_index(drop=True)

    patient_split = make_patient_split(df["patient_id"].unique(), seed=SPLIT_SEED, ratios=SPLIT_RATIOS)

    image_paths: List[str] = []
    rows: List[Dict] = []
    missing: List[str] = []

    for r in df.itertuples(index=False):
        png_path = src_png_root / f"{r.Image_name}.png"
        if not png_path.is_file():
            missing.append(str(png_path))
            continue

        density_raw = str(r.density_raw).strip().upper() if pd.notna(r.density_raw) else ""
        density = f"Level {density_raw}" if density_raw in ("A", "B", "C", "D") else None

        bi_rads = _parse_birads(getattr(r, "BIRADS", None))
        path_raw = str(r.pathology_raw).strip() if pd.notna(r.pathology_raw) else ""
        cancer_status = PATHOLOGY_TO_CANCER.get(path_raw)

        age_val = getattr(r, "Age", None)
        age_int = int(age_val) if age_val is not None and not (isinstance(age_val, float) and pd.isna(age_val)) else None
        machine = getattr(r, "Machine", None)

        image_paths.append(str(png_path))
        rows.append({
            "mmap_idx": -1,
            "sample_name": r.Image_name,
            "exam_id": r.patient_id,  # patient-level exam (1 visit/patient)
            "patient_id": r.patient_id,
            "side": r.side,
            "view": r.view,
            "split": patient_split[r.patient_id],
            "bi_rads": bi_rads,
            "density": density,
            "cancer_status": cancer_status,
            "finding": None,  # CDD-CESM has Findings text but free-form; not aggregated here
            "manufacturer": None,
            "manufacturer_model": str(machine) if machine is not None and not (isinstance(machine, float) and pd.isna(machine)) else None,
            "age": age_int,
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
        builder.verify(sample_size=32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Build {DATASET} mmap")
    parser.add_argument("--img-size", nargs=2, type=int, default=list(DEFAULT_IMG_SIZE), metavar=("H", "W"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    main(tuple(args.img_size), args.out_root, args.num_workers, args.force_rebuild)
