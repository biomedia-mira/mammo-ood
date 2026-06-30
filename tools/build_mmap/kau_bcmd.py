"""Build KAU-BCMD mmap + metadata.parquet at exam-level granularity.

Source:
  - /path/to/Mammo/KAU-BCMD-DICOM/correctSheetlast.xlsx (per-image density)
  - /path/to/Mammo/KAU-BCMD-DICOM/pngs/<H>x<W>/BIRAD {1,3,4,5}/{file}.png

Filename pattern '{year}_{patientID}_{view}_{side}', e.g.
'2018_BC005421_CC_R'. BIRADS comes from folder name; density from xlsx
lookup. exam_id = (patient, side); patient split 70/10/20.

There are 4 known duplicate sample IDs that appear in both BIRAD 4 and BIRAD
5 folders (or BIRAD 1 and 5); we resolve to BIRAD 5 per the existing notebook
convention.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.mmap import MMapBuilder, make_patient_split

DATASET = "KAU-BCMD"
SRC_XLSX = Path("/path/to/Mammo/KAU-BCMD-DICOM/correctSheetlast.xlsx")
SRC_PNG_ROOT_FMT = "/path/to/Mammo/KAU-BCMD-DICOM/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/KAU-BCMD")
DEFAULT_IMG_SIZE = (1024, 768)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.1, 0.2)

CATEGORY_TO_BIRADS = {"BIRAD 1": "Bi-Rads 1", "BIRAD 3": "Bi-Rads 3",
                      "BIRAD 4": "Bi-Rads 4", "BIRAD 5": "Bi-Rads 5"}
DENSITY_TO_LEVEL = {"0%-25%": "Level A", "26%-50%": "Level B",
                    "51%-75%": "Level C", ">75%": "Level D"}
DENSITY_COL = "Percentage of\n grandular tissue(density)"
SPECIAL_BIRADS_OVERRIDE = "Bi-Rads 5"
SPECIAL_CASES = {"2018_BC005421_CC_R", "2018_BC005421_MLO_R",
                 "2018_BC0022482_CC_R", "2018_BC0022482_MLO_R"}


def _parse_stem(stem: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (patient_id, view, side)."""
    parts = stem.replace(" ", "").split("_")
    if len(parts) < 4:
        return None, None, None
    patient_id = f"{parts[0]}_{parts[1]}"
    view = parts[2].upper() if parts[2].upper() in ("CC", "MLO") else None
    side = parts[3].upper() if parts[3].upper() in ("L", "R") else None
    return patient_id, view, side


def _load_density_lookup(valid_stems: set) -> Dict[str, str]:
    if not SRC_XLSX.exists():
        print(f"[{DATASET}] WARNING: density xlsx not found, density will be NaN")
        return {}
    meta = pd.read_excel(SRC_XLSX, sheet_name="correctSheet", usecols=["Image path", DENSITY_COL])
    meta = meta[meta["Image path"].notna()].copy()
    # Rename DENSITY_COL (which has a newline + parens) to a safe identifier so
    # itertuples gives clean attribute access.
    meta = meta.rename(columns={DENSITY_COL: "density_raw"})
    meta["sample_id"] = meta["Image path"].astype(str).map(lambda p: Path(p).stem.replace(" ", ""))
    meta = meta[meta["sample_id"].isin(valid_stems)]
    out: Dict[str, str] = {}
    for r in meta.itertuples(index=False):
        if pd.isna(r.density_raw):
            continue
        mapped = DENSITY_TO_LEVEL.get(str(r.density_raw).strip())
        if mapped:
            out[str(r.sample_id)] = mapped
    return out


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    # Gather PNGs grouped by sample_id; resolve duplicates with BIRADS 5 override
    candidates: Dict[str, List[Tuple[Path, str]]] = defaultdict(list)
    for cat_name, birads in CATEGORY_TO_BIRADS.items():
        cat_dir = src_png_root / cat_name
        if not cat_dir.exists():
            continue
        for png in cat_dir.glob("*.png"):
            if png.name.startswith("._"):
                continue
            stem = png.stem.replace(" ", "")
            candidates[stem].append((png, birads))

    print(f"[{DATASET}] {sum(len(v) for v in candidates.values())} PNGs across {len(candidates)} unique stems")

    density_lookup = _load_density_lookup(set(candidates))

    image_paths: List[str] = []
    rows: List[Dict] = []
    bad_format: List[str] = []
    seen_patients: set = set()

    for stem, items in candidates.items():
        if len(items) > 1:
            if stem in SPECIAL_CASES:
                items = [(p, SPECIAL_BIRADS_OVERRIDE) for (p, b) in items if b == SPECIAL_BIRADS_OVERRIDE]
                if not items:
                    items = [(p, SPECIAL_BIRADS_OVERRIDE) for (p, _) in candidates[stem][:1]]
            else:
                # Unexpected duplicate - skip with warning
                print(f"[{DATASET}] WARNING: unexpected duplicate stem {stem}: {[b for _, b in items]} (skipped)")
                continue

        png_path, birads = items[0]
        patient_id, view, side = _parse_stem(stem)
        if patient_id is None or view is None or side is None:
            bad_format.append(stem)
            continue
        seen_patients.add(patient_id)
        density = density_lookup.get(stem)

        image_paths.append(str(png_path))
        rows.append({
            "mmap_idx": -1,
            "sample_name": stem,
            "exam_id": patient_id,  # patient-level exam (1 visit/patient)
            "patient_id": patient_id,
            "side": side,
            "view": view,
            "split": None,  # filled below
            "bi_rads": birads,
            "density": density,
            "cancer_status": None,
            "finding": None,
            "manufacturer": None,
            "manufacturer_model": None,
            "age": None,
            "study_date": None,
            "race_ethnicity": None,
            "site_id": None,
        })

    if bad_format:
        print(f"[{DATASET}] WARNING: {len(bad_format)} stems with unparseable name format")

    patient_split = make_patient_split(seen_patients, seed=SPLIT_SEED, ratios=SPLIT_RATIOS)
    for r in rows:
        r["split"] = patient_split[r["patient_id"]]

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
