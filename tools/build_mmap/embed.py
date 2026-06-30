"""Build EMBED mmap + metadata.parquet at exam-level granularity.

Source:
  - data/embed_pretrain.csv
    (one row per image; many columns including filtering flags + clinical)
  - /path/to/Mammo/EMBED/pngs/<H>x<W>/<image_path>

Output:
  - /path/to/Mammo/classification_data/EMBED/mmap_<H>x<W>/

Filter (matches `utils/create_info_dict/EMBED.ipynb` and the existing
`pre-training/util/offline_mmap_creator.py`):
  - FinalImageType == '2D'
  - GENDER_DESC == 'Female'
  - tissueden in [1, 4]
  - ViewPosition in {'MLO', 'CC'}
  - spot_mag is NaN
  - asses in {'N', 'B', 'P', 'S', 'M'}

EMBED is exam-level: within the same `acc_anon` (exam), all 4 views (L-CC,
L-MLO, R-CC, R-MLO) share the same `asses` and `tissueden` (verified
empirically: 100% match). exam_id = acc_anon, patient_id = empi_anon.

Patient-level split 70/10/20 by `empi_anon` with seed=42, identical to the
existing notebook to keep run reproducibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.mmap import MMapBuilder, make_patient_split

DATASET = "EMBED"
SRC_CSV = Path("data/embed_pretrain.csv")
SRC_PNG_ROOT_FMT = "/path/to/Mammo/EMBED/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/EMBED")
DEFAULT_IMG_SIZE = (1024, 768)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.1, 0.2)

ASSESS_TO_BIRADS = {"N": "Bi-Rads 1", "B": "Bi-Rads 2", "P": "Bi-Rads 3", "S": "Bi-Rads 4", "M": "Bi-Rads 5"}
DENSITY_TO_LEVEL = {1: "Level A", 2: "Level B", 3: "Level C", 4: "Level D"}
PATH_SEVERITY_TO_CANCER = {0: "Cancer", 1: "Cancer", 4: "Non-cancer", 5: "Non-cancer"}

USECOLS = [
    "empi_anon", "acc_anon", "FinalImageType", "GENDER_DESC", "tissueden",
    "ViewPosition", "spot_mag", "asses", "image_path", "image_id",
    "ImageLateralityFinal", "bside", "path_severity",
    "Manufacturer", "ManufacturerModelName",
    "age_at_study", "study_date_anon_x", "RACE_DESC", "ETHNIC_GROUP_DESC",
    "site",  # may not always exist; we read with usecols-fallback
]


def _read_filtered() -> pd.DataFrame:
    # `site` column is not always present in older exports; fall back without it.
    try:
        df = pd.read_csv(SRC_CSV, usecols=USECOLS, low_memory=False)
    except (ValueError, KeyError):
        cols = [c for c in USECOLS if c != "site"]
        df = pd.read_csv(SRC_CSV, usecols=cols, low_memory=False)
        df["site"] = pd.NA
    df = df[df["FinalImageType"] == "2D"]
    df = df[df["GENDER_DESC"] == "Female"]
    df = df[df["tissueden"].notna()]
    df = df[df["tissueden"] < 5]
    df = df[df["ViewPosition"].isin(["MLO", "CC"])]
    df = df[df["spot_mag"].isna()]
    df = df[df["asses"].isin(ASSESS_TO_BIRADS)].copy()
    df["empi_anon"] = df["empi_anon"].astype(str)
    df["acc_anon"] = df["acc_anon"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    if df["image_id"].duplicated().any():
        dupes = df.loc[df["image_id"].duplicated(), "image_id"].unique()[:10]
        raise ValueError(f"Duplicate image_id after filtering: {list(dupes)}")
    return df


def _race_ethnicity(row) -> Optional[str]:
    race = row.get("RACE_DESC")
    eth = row.get("ETHNIC_GROUP_DESC")
    if pd.isna(race) and pd.isna(eth):
        return None
    if pd.isna(race):
        return f"ethnicity={eth}"
    if pd.isna(eth):
        return str(race)
    return f"{race} | {eth}"


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    df = _read_filtered()
    print(f"[{DATASET}] {len(df)} images after filtering", flush=True)

    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    patient_split = make_patient_split(df["empi_anon"].unique(), seed=SPLIT_SEED, ratios=SPLIT_RATIOS)

    image_paths: List[str] = []
    rows: List[Dict] = []

    df = df.sort_values(["empi_anon", "acc_anon", "image_id"], kind="stable").reset_index(drop=True)
    for r in df.itertuples(index=False):
        png_path = src_png_root / str(r.image_path)

        bi_rads = ASSESS_TO_BIRADS[r.asses]
        density = DENSITY_TO_LEVEL[int(r.tissueden)]
        side = str(r.ImageLateralityFinal).strip().upper() if pd.notna(r.ImageLateralityFinal) else None
        if side not in ("L", "R"):
            side = None
        view = str(r.ViewPosition).strip().upper() if pd.notna(r.ViewPosition) else None
        if view not in ("CC", "MLO"):
            view = None

        # CancerStatus only when pathology side matches image laterality (or 'B' = bilateral)
        cancer_status = None
        bside = str(r.bside).strip().upper() if pd.notna(r.bside) else ""
        if (side and bside == side) or bside == "B":
            sev = r.path_severity
            if pd.notna(sev):
                cancer_status = PATH_SEVERITY_TO_CANCER.get(int(sev))

        sd = getattr(r, "study_date_anon_x", None)
        study_date = None
        if sd is not None and not (isinstance(sd, float) and pd.isna(sd)):
            study_date = str(sd)[:10]  # YYYY-MM-DD

        age = getattr(r, "age_at_study", None)
        age_int = int(age) if age is not None and not (isinstance(age, float) and pd.isna(age)) else None

        site = getattr(r, "site", None)
        site_id = str(site) if site is not None and not (isinstance(site, float) and pd.isna(site)) else None

        image_paths.append(str(png_path))
        rows.append({
            "mmap_idx": -1,
            "sample_name": str(r.image_id),
            "exam_id": str(r.acc_anon),
            "patient_id": str(r.empi_anon),
            "side": side,
            "view": view,
            "split": patient_split[str(r.empi_anon)],
            "bi_rads": bi_rads,
            "density": density,
            "cancer_status": cancer_status,
            "finding": None,  # EMBED has lesion boxes elsewhere; not loaded here
            "manufacturer": str(r.Manufacturer) if pd.notna(r.Manufacturer) else None,
            "manufacturer_model": str(r.ManufacturerModelName) if pd.notna(r.ManufacturerModelName) else None,
            "age": age_int,
            "study_date": study_date,
            "race_ethnicity": _race_ethnicity(r._asdict()),
            "site_id": site_id,
        })

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
