"""Build VinDr-Mammo mmap + metadata.parquet at exam-level granularity.

Source:
  - /path/to/Mammo/VinDr-Mammo/breast-level_annotations.csv
    (one row per image; columns: study_id, image_id, laterality,
     view_position, breast_birads, breast_density, split)
  - /path/to/Mammo/VinDr-Mammo/finding_annotations.csv (lesion-level
     finding boxes; aggregated to image-level multi-label)
  - /path/to/Mammo/VinDr-Mammo/pngs/<H>x<W>/<study_id>/<image_id>.png

Output:
  - /path/to/Mammo/classification_data/VinDr-Mammo/mmap_<H>x<W>/
        ├── images_<H>x<W>.npy        # float32, shape (N, 1, H, W)
        └── metadata.parquet          # canonical schema

Notes:
  - VinDr-Mammo BI-RADS and density are breast-level: CC and MLO of the same
    breast share identical labels (verified empirically: 100% match within
    study_id+laterality). exam_id = study_id; patient_id = study_id (1
    study/patient).
  - CancerStatus is a pseudo-label derived from BI-RADS (1/2 -> Non-cancer,
    5 -> Cancer, 3/4 left as NaN).
  - Source split column ('training' | 'test') is mapped to canonical
    Train/Test, then 12.5% of training patients (study_id) are carved out
    deterministically (seed=42) into Eval to match the 70/10/20 convention
    used elsewhere. Test split is preserved as-is.
  - Findings are aggregated per image_id from finding_annotations.csv.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from utils.mmap import MMapBuilder, make_patient_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET = "VinDr-Mammo"
SRC_BREAST_CSV = Path("/path/to/Mammo/VinDr-Mammo/breast-level_annotations.csv")
SRC_FINDING_CSV = Path("/path/to/Mammo/VinDr-Mammo/finding_annotations.csv")
SRC_METADATA_CSV = Path("/path/to/Mammo/VinDr-Mammo/metadata.csv")
SRC_PNG_ROOT_FMT = "/path/to/Mammo/VinDr-Mammo/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/VinDr-Mammo")
DEFAULT_IMG_SIZE = (1024, 768)

# Carve an Eval split out of the source 'training' patients (Test stays as-is).
# 12.5% of training -> ~2000 Eval, ~14000 Train, with Test=4000 this matches
# the project's 70/10/20 convention (utils/mmap/splits.DEFAULT_RATIOS).
EVAL_FRACTION_OF_TRAIN = 0.125
SPLIT_SEED = 42

BIRADS_NORMALIZE = {f"BI-RADS {i}": f"Bi-Rads {i}" for i in range(1, 6)}
DENSITY_NORMALIZE = {f"DENSITY {c}": f"Level {c}" for c in "ABCD"}
BIRADS_TO_CANCER = {
    "Bi-Rads 1": "Non-cancer",
    "Bi-Rads 2": "Non-cancer",
    "Bi-Rads 5": "Cancer",
    # 0/3/4 -> NaN (not enough information for binary cancer status)
}
SPLIT_NORMALIZE = {
    "training": "Train",
    "train": "Train",
    "test": "Test",
}
SUPPORTED_FINDINGS = {
    "Mass": "mass",
    "Suspicious Calcification": "calcification",
    "Asymmetry": "asymmetry",
    "Focal Asymmetry": "asymmetry",
    "Global Asymmetry": "asymmetry",
    "Architectural Distortion": "architectural distortion",
}
FINDING_ORDER = ["mass", "calcification", "asymmetry", "architectural distortion"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_finding_per_image() -> Dict[str, List[str]]:
    """Aggregate finding_annotations.csv into image_id -> sorted list."""
    df = pd.read_csv(SRC_FINDING_CSV)
    df["finding_categories"] = df["finding_categories"].apply(ast.literal_eval)

    out: Dict[str, set] = {}
    for image_id, group in df.groupby("image_id"):
        labels: set = set()
        for cats in group["finding_categories"]:
            for cat in cats:
                if cat == "No Finding":
                    continue
                if cat in SUPPORTED_FINDINGS:
                    labels.add(SUPPORTED_FINDINGS[cat])
        out[str(image_id)] = labels
    return {k: sorted(v, key=FINDING_ORDER.index) for k, v in out.items()}


def _parse_dicom_age(raw) -> Optional[int]:
    """Parse DICOM-style age string like '053Y' -> 53. Returns None if unparseable."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip().upper()
    if not s:
        return None
    # Strip trailing unit letter (Y/M/W/D); keep only leading digits
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    try:
        val = int(digits)
    except ValueError:
        return None
    return val if 0 <= val <= 120 else None


def _load_per_image_metadata() -> Dict[str, Dict[str, Optional[object]]]:
    """Read VinDr metadata.csv -> image_id -> {manufacturer, manufacturer_model, age}.

    Column names in VinDr metadata.csv contain spaces and apostrophes (e.g.
    "Manufacturer's Model Name"); we rename them up front so .itertuples
    gives clean attribute access.
    """
    if not SRC_METADATA_CSV.exists():
        return {}
    df = pd.read_csv(SRC_METADATA_CSV)
    rename = {
        "SOP Instance UID": "sop_uid",
        "Manufacturer": "manufacturer",
        "Manufacturer's Model Name": "manufacturer_model",
        "Patient's Age": "age_raw",
    }
    missing = set(rename) - set(df.columns)
    if missing:
        print(f"[{DATASET}] WARNING: VinDr metadata.csv missing columns {sorted(missing)}; "
              f"those fields will be NaN.")
    df = df[[c for c in rename if c in df.columns]].rename(columns=rename)

    out: Dict[str, Dict[str, Optional[object]]] = {}
    for r in df.itertuples(index=False):
        sop = getattr(r, "sop_uid", None)
        if sop is None or (isinstance(sop, float) and pd.isna(sop)):
            continue
        rec: Dict[str, Optional[object]] = {}
        mfr = getattr(r, "manufacturer", None)
        if mfr is not None and not (isinstance(mfr, float) and pd.isna(mfr)):
            rec["manufacturer"] = str(mfr)
        model = getattr(r, "manufacturer_model", None)
        if model is not None and not (isinstance(model, float) and pd.isna(model)):
            rec["manufacturer_model"] = str(model)
        age = _parse_dicom_age(getattr(r, "age_raw", None))
        if age is not None:
            rec["age"] = age
        out[str(sop)] = rec
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_metadata_rows(img_size) -> tuple[List[str], List[Dict]]:
    breast_df = pd.read_csv(SRC_BREAST_CSV)
    required = {"study_id", "image_id", "laterality", "view_position",
                "breast_birads", "breast_density", "split"}
    missing = required - set(breast_df.columns)
    if missing:
        raise ValueError(f"breast-level CSV missing columns: {sorted(missing)}")

    if breast_df["image_id"].duplicated().any():
        dupes = breast_df.loc[breast_df["image_id"].duplicated(), "image_id"].unique()[:10]
        raise ValueError(f"Duplicate image_id in breast CSV: {list(dupes)}")

    findings = _load_finding_per_image()
    per_image = _load_per_image_metadata()

    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    image_paths: List[str] = []
    rows: List[Dict] = []
    missing_pngs: List[str] = []

    breast_df = breast_df.sort_values(["study_id", "image_id"], kind="stable").reset_index(drop=True)

    # Carve Eval out of training: pick a deterministic patient subset (= study_id
    # in VinDr, since 1 study/patient) using EVAL_FRACTION_OF_TRAIN.
    train_studies = breast_df.loc[
        breast_df["split"].str.strip().str.lower().eq("training"), "study_id"
    ].astype(str).unique()
    eval_ratio = float(EVAL_FRACTION_OF_TRAIN)
    train_eval_split = make_patient_split(
        train_studies,
        seed=SPLIT_SEED,
        ratios=(1.0 - eval_ratio, eval_ratio, 0.0),
    )
    # make_patient_split returns Train/Eval/Test for the *given* patients only;
    # we just use Train -> stays Train, Eval -> becomes our carved Eval.
    eval_studies = {sid for sid, sp in train_eval_split.items() if sp == "Eval"}
    print(f"[{DATASET}] carved {len(eval_studies)}/{len(train_studies)} training studies into Eval (seed={SPLIT_SEED})")

    for r in breast_df.itertuples(index=False):
        study_id = str(r.study_id)
        image_id = str(r.image_id)
        png_path = src_png_root / study_id / f"{image_id}.png"
        if not png_path.is_file():
            missing_pngs.append(str(png_path))
            continue

        bi_rads = BIRADS_NORMALIZE.get(str(r.breast_birads).strip())
        density = DENSITY_NORMALIZE.get(str(r.breast_density).strip())
        if bi_rads is None:
            raise ValueError(f"Unknown breast_birads '{r.breast_birads}' for {image_id}")
        if density is None:
            raise ValueError(f"Unknown breast_density '{r.breast_density}' for {image_id}")

        cancer_status = BIRADS_TO_CANCER.get(bi_rads)  # NaN for 0/3/4
        split = SPLIT_NORMALIZE.get(str(r.split).strip().lower())
        if split is None:
            raise ValueError(f"Unknown split value '{r.split}' for {image_id}")
        if split == "Train" and study_id in eval_studies:
            split = "Eval"

        side = str(r.laterality).strip().upper()
        view = str(r.view_position).strip().upper()
        if side not in ("L", "R"):
            raise ValueError(f"Unexpected laterality '{r.laterality}' for {image_id}")
        if view not in ("CC", "MLO"):
            raise ValueError(f"Unexpected view_position '{r.view_position}' for {image_id}")

        finding_list = findings.get(image_id, [])
        finding_value = list(finding_list) if finding_list else None

        per_img = per_image.get(image_id, {})

        image_paths.append(str(png_path))
        rows.append({
            "mmap_idx": -1,  # filled in by MMapBuilder
            "sample_name": image_id,
            "exam_id": study_id,
            "patient_id": study_id,
            "side": side,
            "view": view,
            "split": split,
            "bi_rads": bi_rads,
            "density": density,
            "cancer_status": cancer_status,
            "finding": finding_value,
            "manufacturer": per_img.get("manufacturer"),
            "manufacturer_model": per_img.get("manufacturer_model"),
            "age": per_img.get("age"),  # int or None; pandas casts to Int64
            "study_date": None,         # not in VinDr metadata.csv
            "race_ethnicity": None,     # not provided
            "site_id": None,            # VinDr is from 2 hospitals but no per-image site col
        })

    if missing_pngs:
        print(f"WARNING: {len(missing_pngs)} PNGs missing under {src_png_root} (skipped)")
        for p in missing_pngs[:5]:
            print(f"  {p}")

    if not image_paths:
        raise RuntimeError("No images resolved. Check SRC_PNG_ROOT_FMT and image size.")

    return image_paths, rows


def main(
    img_size=DEFAULT_IMG_SIZE,
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

    builder = MMapBuilder(
        mmap_path=mmap_path,
        metadata_path=metadata_path,
        img_size=img_size,
    )
    result = builder.build(
        image_paths=image_paths,
        metadata_rows=rows,
        num_workers=num_workers,
        force_rebuild=force_rebuild,
    )
    print(
        f"[{DATASET}] done: total={result.n_total} "
        f"written={result.n_written} skipped={result.n_skipped} failed={result.n_failed}"
    )
    if result.n_failed == 0:
        builder.verify(sample_size=32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Build {DATASET} mmap")
    parser.add_argument("--img-size", nargs=2, type=int, default=list(DEFAULT_IMG_SIZE),
                        metavar=("H", "W"), help="Output image size (height width)")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    main(
        img_size=tuple(args.img_size),
        out_root=args.out_root,
        num_workers=args.num_workers,
        force_rebuild=args.force_rebuild,
    )
