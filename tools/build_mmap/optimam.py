"""Build OPTIMAM mmap + metadata.parquet at exam-level granularity.

Source:
  - <OPTIMAM_ROOT>/optimam_classification.csv
  - <OPTIMAM_ROOT>/optimam_marks.csv      (lesion bounding boxes)
  - <OPTIMAM_ROOT>/meta/density/studies.csv (Volpara VDG 5thEd)
  - <OPTIMAM_ROOT>/meta/ethnicity.csv
  - <OPTIMAM_ROOT>/pngs/<H>x<W>/...image_path

Output:
  - /path/to/Mammo/classification_data/OPTIMAM/mmap_<H>x<W>/

OPTIMAM ships with a `split` column (train/val/test). We keep that mapping
verbatim (val -> Eval). exam_id = study_uid (1 imaging session); patient_id
= client_id.

Filter (analogous to EMBED's strict screening filter):
  - ViewPosition in {CC, MLO}
  - ImageLateralityFinal in {L, R}
  - split in {train, val, test}
  - study_event_type == 'screening'   (drop biopsy/surgery/assessment/NaN)
  - study has exactly 4 images, one per (side, view) combo
    (drop incomplete and extra-view studies)

After filtering, every study has the standard 4-view set
(L-CC, L-MLO, R-CC, R-MLO) — perfect for exam-level 4-input model. About
93.1% of source studies remain; equivalent to EMBED's `acc_anon` exam unit.

CancerStatus uses the patient-level `is_cancer` flag from the source CSV
(derived by the official OPTIMAM pipeline as `episode_status in {M, CI}`).
For a screening study in a cancer episode, the cancer was actually present
on those images at acquisition (whether radiologist detected it or not for
interval cancers). That is the standard OPTIMAM convention and matches what
we want for exam-level prediction (model sees all 4 views L-CC/L-MLO/R-CC/
R-MLO and outputs one decision per visit).

OPTIMAM episode_status codes:
  N  = Normal
  B  = Benign
  M  = Malignant         (biopsy-confirmed cancer in this episode)
  CI = Interval Cancer   (cancer found between screening rounds)
  NA = Normal with assessment (recalled but cleared)
is_cancer = 1 iff episode_status in {M, CI}; else 0.

BI-RADS is NOT populated for OPTIMAM. The dataset uses NHS NBSS opinion
codes (P1-P5) which are not directly equivalent to ACR BI-RADS, and there
is no official mapping between them. We leave `bi_rads = None` so downstream
training is not corrupted by a fake mapping.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.mmap import MMapBuilder

DATASET = "OPTIMAM"
OPTIMAM_ROOT = Path(os.environ.get("OPTIMAM_ROOT", "/path/to/optimam"))
SRC_CSV = OPTIMAM_ROOT / "optimam_classification.csv"
SRC_PNG_ROOT_FMT = str(OPTIMAM_ROOT / "pngs" / "{h}x{w}")
SRC_DENSITY_CSV = OPTIMAM_ROOT / "meta" / "density" / "studies.csv"
SRC_ETHNICITY_CSV = OPTIMAM_ROOT / "meta" / "ethnicity.csv"
SRC_MARKS_CSV = OPTIMAM_ROOT / "optimam_marks.csv"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/OPTIMAM")
DEFAULT_IMG_SIZE = (1024, 768)

# Volpara Density Grade 5th edition -> ACR BI-RADS A-D mapping.
VDG5_TO_LEVEL = {1: "Level A", 2: "Level B", 3: "Level C", 4: "Level D"}

SPLIT_NORMALIZE = {"train": "Train", "val": "Eval", "test": "Test"}

USECOLS = [
    "client_id", "site", "study_uid", "study_date", "study_event_type_text",
    "image_path", "image_id",
    "ImageLateralityFinal", "ViewPosition",
    "Manufacturer", "ManufacturerModelName",
    "AgeAtScreening", "is_cancer", "split",
]


def _load_density_lookup() -> Dict[str, str]:
    """study_uid -> 'Level A'..'Level D' from Volpara VDG 5th edition."""
    if not SRC_DENSITY_CSV.exists():
        print(f"[{DATASET}] WARNING: density CSV not found, density column will be NaN")
        return {}
    df = pd.read_csv(SRC_DENSITY_CSV, usecols=["Path", "VDG 5thEd"])
    out: Dict[str, str] = {}
    for path, vdg in df.itertuples(index=False):
        if pd.isna(vdg):
            continue
        level = VDG5_TO_LEVEL.get(int(vdg))
        if level:
            out[str(path)] = level
    print(f"[{DATASET}] loaded density for {len(out):,} studies")
    return out


FINDING_FLAG_TO_NAME = [
    # (column, finding_name) — order determines deterministic sort within image
    ("Mass", "mass"),
    ("SuspiciousCalcifications", "calcification"),
    ("Calcifications", "calcification"),
    ("FocalAsymmetry", "asymmetry"),
    ("ArchitecturalDistortion", "architectural distortion"),
]
FINDING_ORDER = ["mass", "calcification", "asymmetry", "architectural distortion"]


def _is_true_flag(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def _load_finding_lookup() -> Dict[str, List[str]]:
    """sop_uid -> sorted unique list of mapped findings.

    Aggregates marks per image: union over all bounding-box rows of the four
    supported finding flags.
    """
    if not SRC_MARKS_CSV.exists():
        print(f"[{DATASET}] WARNING: marks CSV not found, finding column will be NaN")
        return {}
    cols = ["sop_uid"] + [c for c, _ in FINDING_FLAG_TO_NAME]
    df = pd.read_csv(SRC_MARKS_CSV, usecols=cols, low_memory=False)

    out: Dict[str, set] = {}
    for r in df.itertuples(index=False):
        names: set = set()
        for col, name in FINDING_FLAG_TO_NAME:
            if _is_true_flag(getattr(r, col, None)):
                names.add(name)
        if names:
            out.setdefault(str(r.sop_uid), set()).update(names)
    final = {k: sorted(v, key=FINDING_ORDER.index) for k, v in out.items()}
    print(f"[{DATASET}] loaded findings for {len(final):,} images")
    return final


def _load_ethnicity_lookup() -> Dict[str, str]:
    """client_id -> EthnicCategory string."""
    if not SRC_ETHNICITY_CSV.exists():
        print(f"[{DATASET}] WARNING: ethnicity CSV not found, race_ethnicity will be NaN")
        return {}
    df = pd.read_csv(SRC_ETHNICITY_CSV, usecols=["ClientID", "EthnicCategory"])
    df = df[df["EthnicCategory"].notna()]
    out = {str(c): str(e) for c, e in df.itertuples(index=False)}
    print(f"[{DATASET}] loaded ethnicity for {len(out):,} clients")
    return out


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    df = pd.read_csv(SRC_CSV, usecols=USECOLS, low_memory=False)
    print(f"[{DATASET}] read {len(df):,} rows")

    df = df[df["ViewPosition"].isin(["CC", "MLO"])]
    df = df[df["ImageLateralityFinal"].isin(["L", "R"])]
    df = df[df["split"].isin(SPLIT_NORMALIZE)]
    # Keep only screening studies — cleanest for cancer detection label.
    df = df[df["study_event_type_text"].astype(str).str.strip().str.lower() == "screening"]
    df["client_id"] = df["client_id"].astype(str)
    df["study_uid"] = df["study_uid"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    print(f"[{DATASET}] {len(df):,} rows after CC/MLO + L/R + screening")

    # Strict clean filter: keep only studies that contain exactly one image per
    # (side, view) combination — the standard screening exam of 4 images
    # (L-CC, L-MLO, R-CC, R-MLO). Drops incomplete exams and studies with
    # extra spot/magnification views (~7% drop, retains 93.1%).
    df = df.assign(_sv=df["ImageLateralityFinal"] + "-" + df["ViewPosition"])
    REQUIRED = frozenset(("L-CC", "L-MLO", "R-CC", "R-MLO"))
    g = df.groupby("study_uid")
    has_all_4 = g["_sv"].agg(lambda s: frozenset(s) == REQUIRED)
    exactly_4 = g.size() == 4
    clean_studies = has_all_4.index[(has_all_4 & exactly_4).values]
    df = df[df["study_uid"].isin(clean_studies)].drop(columns="_sv")
    print(f"[{DATASET}] {len(df):,} rows / {df['study_uid'].nunique():,} studies after clean 4-view filter")

    if df["image_id"].duplicated().any():
        dupes = df.loc[df["image_id"].duplicated(), "image_id"].unique()[:10]
        raise ValueError(f"Duplicate image_id in OPTIMAM CSV: {list(dupes)}")

    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    density_by_study = _load_density_lookup()
    ethnicity_by_client = _load_ethnicity_lookup()
    finding_by_sop = _load_finding_lookup()

    image_paths: List[str] = []
    rows: List[Dict] = []
    missing: List[str] = []

    df = df.sort_values(["client_id", "study_uid", "image_id"], kind="stable").reset_index(drop=True)

    for r in df.itertuples(index=False):
        png_path = src_png_root / str(r.image_path)
        if not png_path.is_file():
            missing.append(str(png_path))
            continue

        side = str(r.ImageLateralityFinal).strip().upper()
        view = str(r.ViewPosition).strip().upper()

        cancer_status = "Cancer" if int(r.is_cancer) == 1 else "Non-cancer"

        sd = getattr(r, "study_date", None)
        study_date = None
        if sd is not None and not (isinstance(sd, float) and pd.isna(sd)):
            study_date = str(sd)[:10]

        age = getattr(r, "AgeAtScreening", None)
        age_int = int(age) if age is not None and not (isinstance(age, float) and pd.isna(age)) else None

        site_val = getattr(r, "site", None)
        site_id = str(site_val) if site_val is not None and not (isinstance(site_val, float) and pd.isna(site_val)) else None

        image_paths.append(str(png_path))
        rows.append({
            "mmap_idx": -1,
            "sample_name": str(r.image_id),
            "exam_id": str(r.study_uid),
            "patient_id": str(r.client_id),
            "side": side,
            "view": view,
            "split": SPLIT_NORMALIZE[str(r.split).strip().lower()],
            "bi_rads": None,  # OPTIMAM uses NHS NBSS, not ACR BI-RADS — leave unmapped
            "density": density_by_study.get(str(r.study_uid)),
            "cancer_status": cancer_status,
            "finding": finding_by_sop.get(str(r.image_id)),
            "manufacturer": str(r.Manufacturer) if pd.notna(r.Manufacturer) else None,
            "manufacturer_model": str(r.ManufacturerModelName) if pd.notna(r.ManufacturerModelName) else None,
            "age": age_int,
            "study_date": study_date,
            "race_ethnicity": ethnicity_by_client.get(str(r.client_id)),
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
