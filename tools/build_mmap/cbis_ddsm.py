"""Build CBIS-DDSM mmap + metadata.parquet at exam-level granularity (case = patient+side).

Source:
  - mass/calc case description CSVs in /path/to/Mammo/CBIS-DDSM/
  - /path/to/Mammo/CBIS-DDSM/pngs/<H>x<W>/<data_name>/<n>.png
    (data_name = first path component of `image file path`, e.g.
     'Mass-Training_P_00001_LEFT_CC')

Each row in the description CSVs is one lesion in one image-view. We aggregate
all lesion rows for each `data_name` (a unique patient+side+view) into image-
level labels:
  - bi_rads:        max(assessment) across rows
  - density:        the unique nonzero breast_density value
  - cancer_status:  any MALIGNANT lesion -> Cancer, else Non-cancer
  - finding:        sorted unique abnormality types (mass / calcification)

Each PNG folder may contain multiple slices; for now we pick the single full
mammogram PNG (the existing convert script writes one). Patient-level split
70/10/20 by patient_id with seed=42.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.mmap import MMapBuilder, make_patient_split

DATASET = "CBIS-DDSM"
SRC_DIR = Path("/path/to/Mammo/CBIS-DDSM")
SRC_CSVS = [
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
]
SRC_PNG_ROOT_FMT = "/path/to/Mammo/CBIS-DDSM/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/CBIS-DDSM")
DEFAULT_IMG_SIZE = (1024, 768)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.1, 0.2)

DENSITY_MAP = {1: "Level A", 2: "Level B", 3: "Level C", 4: "Level D"}
PATHOLOGY_TO_CANCER = {"BENIGN": "Non-cancer", "BENIGN_WITHOUT_CALLBACK": "Non-cancer", "MALIGNANT": "Cancer"}


def _parse_data_name(data_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (patient_id, side, view) from 'Mass-Training_P_00001_LEFT_CC'."""
    m = re.match(r"^(?:Mass|Calc)-(?:Training|Test)_(P_\d+)_(LEFT|RIGHT)_(CC|MLO)$", data_name)
    if not m:
        return None, None, None
    return m.group(1), {"LEFT": "L", "RIGHT": "R"}[m.group(2)], m.group(3)


def _load_aggregated() -> pd.DataFrame:
    parts = []
    for f in SRC_CSVS:
        df = pd.read_csv(SRC_DIR / f)
        df.columns = df.columns.str.strip()
        # The 4 source CSVs use either 'breast density' (mass) or 'breast_density'
        # (calc). Normalize to a single name.
        df = df.rename(columns={"breast density": "breast_density"})
        df["data_name"] = df["image file path"].astype(str).str.split("/").str[0].str.strip()
        parts.append(df[["data_name", "patient_id", "left or right breast", "image view",
                         "abnormality type", "assessment", "pathology", "breast_density"]])
    return pd.concat(parts, ignore_index=True)


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    df = _load_aggregated()
    print(f"[{DATASET}] {len(df)} lesion rows over {df['data_name'].nunique()} images")

    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    image_paths: List[str] = []
    rows: List[Dict] = []
    missing: List[str] = []
    bad_format: List[str] = []
    seen_patients: set = set()
    pending_rows: List[Dict] = []

    for data_name, group in df.groupby("data_name", sort=True):
        patient_id, side, view = _parse_data_name(data_name)
        if patient_id is None:
            bad_format.append(data_name)
            continue
        seen_patients.add(patient_id)

        # Aggregate lesion rows
        assessments = sorted({int(v) for v in group["assessment"].dropna()})
        bi_rads = f"Bi-Rads {max(assessments)}" if assessments else None
        # CBIS-DDSM density is the same value across all lesion rows of an image
        # (verified empirically: 0 cases have conflicting nonzero density).
        densities = {int(v) for v in group["breast_density"].dropna() if int(v) != 0}
        density = DENSITY_MAP.get(next(iter(densities))) if densities else None
        pathologies = {str(v).strip().upper() for v in group["pathology"].dropna()}
        if "MALIGNANT" in pathologies:
            cancer_status = "Cancer"
        elif pathologies & {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}:
            cancer_status = "Non-cancer"
        else:
            cancer_status = None
        findings = sorted({str(v).strip().lower() for v in group["abnormality type"].dropna()
                           if str(v).strip().lower() in ("mass", "calcification")})

        png_dir = src_png_root / data_name
        if not png_dir.is_dir():
            missing.append(str(png_dir))
            continue
        png_candidates = sorted(p for p in png_dir.glob("*.png") if not p.name.startswith("._"))
        if not png_candidates:
            missing.append(str(png_dir))
            continue
        png_path = png_candidates[0]
        if len(png_candidates) > 1:
            print(f"[{DATASET}] WARNING: multiple PNGs in {png_dir}, using {png_path.name}")

        pending_rows.append({
            "image_path": str(png_path),
            "row": {
                "mmap_idx": -1,
                "sample_name": data_name,
                "exam_id": patient_id,  # patient-level exam (1 visit/patient in CBIS-DDSM)
                "patient_id": patient_id,
                "side": side,
                "view": view,
                "split": None,  # filled below
                "bi_rads": bi_rads,
                "density": density,
                "cancer_status": cancer_status,
                "finding": findings if findings else None,
                "manufacturer": None,
                "manufacturer_model": None,
                "age": None,
                "study_date": None,
                "race_ethnicity": None,
                "site_id": None,
            },
        })

    if bad_format:
        print(f"[{DATASET}] WARNING: {len(bad_format)} data_names did not match expected pattern")
        for d in bad_format[:5]:
            print(f"  {d}")
    if missing:
        print(f"[{DATASET}] WARNING: {len(missing)} PNG dirs missing or empty (skipped)")

    patient_split = make_patient_split(seen_patients, seed=SPLIT_SEED, ratios=SPLIT_RATIOS)
    for entry in pending_rows:
        entry["row"]["split"] = patient_split[entry["row"]["patient_id"]]
        image_paths.append(entry["image_path"])
        rows.append(entry["row"])

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
