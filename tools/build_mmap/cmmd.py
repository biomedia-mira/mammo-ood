"""Build CMMD mmap + metadata.parquet at exam-level granularity.

Source layout (verified):
  - /path/to/Mammo/CMMD/<study_uid>/1-{N}.dcm    (raw DICOMs, by study UID)
  - /path/to/Mammo/CMMD/CMMD_clinicaldata_revision.xlsx
      (1 row per breast-side: ID1, LeftRight, abnormality, classification)
  - /path/to/Mammo/CMMD/pngs/<H>x<W>/<patient_id>/1-{N}.png
      (PNGs by PatientID, e.g. 'D1-0501/1-1.png')

The PNG folders use PatientID (D1-XXXX) while DICOM folders use study UIDs;
we bridge with a one-pass DICOM scan that records the {study_uid, instance,
side, view, patient_id} of every .dcm file in parallel. Each DICOM gives:
  - ImageLaterality (L/R)
  - ViewCodeSequence (SNOMED-CT 399162004 = CC, 399368009 = MLO)
  - PatientID (D1-XXXX)

CMMD covers 1775 patients (single Siemens Mammomat Inspiration site). Patient
labels live in the clinical xlsx per (ID1, LeftRight). Patient-level split
70/10/20 by PatientID with seed=42.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pydicom
from tqdm import tqdm

from utils.mmap import MMapBuilder, make_patient_split

DATASET = "CMMD"
SRC_DICOM_ROOT = Path("/path/to/Mammo/CMMD")
SRC_XLSX = Path("/path/to/Mammo/CMMD/CMMD_clinicaldata_revision.xlsx")
SRC_PNG_ROOT_FMT = "/path/to/Mammo/CMMD/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/CMMD")
DEFAULT_IMG_SIZE = (1024, 768)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.1, 0.2)

VIEWCODE_TO_VIEW = {"399162004": "CC", "399368009": "MLO"}
CLASSIFICATION_TO_CANCER = {"Benign": "Non-cancer", "Malignant": "Cancer"}
ABNORMALITY_TO_FINDING = {
    # Lowercase to match the convention used by VinDr / DMID / MIAS / CBIS-DDSM.
    "mass": ["mass"],
    "calcification": ["calcification"],
    "both": ["mass", "calcification"],
}


def _read_one_dcm(dcm_path_str: str) -> Optional[Dict]:
    try:
        ds = pydicom.dcmread(dcm_path_str, stop_before_pixels=True)
    except Exception:
        return None
    side = getattr(ds, "ImageLaterality", None)
    side = str(side).strip().upper() if side else None
    if side not in ("L", "R"):
        side = None

    view = None
    seq = getattr(ds, "ViewCodeSequence", None)
    if seq:
        for item in seq:
            code = getattr(item, "CodeValue", None)
            if code is not None:
                v = VIEWCODE_TO_VIEW.get(str(code).strip())
                if v:
                    view = v
                    break

    pid = getattr(ds, "PatientID", None)
    pid = str(pid).strip() if pid else None
    return {"path": dcm_path_str, "side": side, "view": view, "patient_id": pid}


def _build_dicom_index(num_workers: Optional[int] = None) -> Dict[Tuple[str, str], Dict]:
    """One-pass scan of all DICOMs. Returns {(patient_id, stem) -> info}.

    Stem here is e.g. '1-1' (from filename '1-1.dcm'). Done in parallel.
    """
    print(f"[{DATASET}] scanning DICOMs in {SRC_DICOM_ROOT}...")
    dcm_paths: List[str] = []
    for study_dir in SRC_DICOM_ROOT.iterdir():
        if study_dir.is_dir() and study_dir.name.startswith("1.3"):
            for dcm in study_dir.glob("*.dcm"):
                dcm_paths.append(str(dcm))
    print(f"[{DATASET}] found {len(dcm_paths)} DICOM files")

    if num_workers is None:
        try:
            num_workers = len(os.sched_getaffinity(0))
        except AttributeError:
            num_workers = mp.cpu_count()

    index: Dict[Tuple[str, str], Dict] = {}
    with mp.Pool(processes=num_workers) as pool:
        for info in tqdm(
            pool.imap_unordered(_read_one_dcm, dcm_paths, chunksize=64),
            total=len(dcm_paths),
            desc=f"[{DATASET}] reading DICOM tags",
        ):
            if info is None or info["patient_id"] is None:
                continue
            stem = Path(info["path"]).stem  # '1-1'
            index[(info["patient_id"], stem)] = info
    print(f"[{DATASET}] indexed {len(index)} (patient_id, stem) entries")
    return index


def _load_clinical() -> Dict[Tuple[str, str], Dict]:
    df = pd.read_excel(SRC_XLSX)
    out: Dict[Tuple[str, str], Dict] = {}
    for r in df.itertuples(index=False):
        pid = str(r.ID1).strip()
        side = str(r.LeftRight).strip().upper()
        if side not in ("L", "R"):
            continue
        classif = str(r.classification).strip()
        abn = str(r.abnormality).strip().lower()
        out[(pid, side)] = {
            "cancer_status": CLASSIFICATION_TO_CANCER.get(classif),
            "finding": ABNORMALITY_TO_FINDING.get(abn),
        }
    return out


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    clinical = _load_clinical()
    print(f"[{DATASET}] {len(clinical)} clinical (patient, side) entries")

    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    dcm_index = _build_dicom_index()

    image_paths: List[str] = []
    rows: List[Dict] = []
    missing_dicom_info: List[str] = []
    missing_clinical: List[Tuple[str, str]] = []
    incomplete_tags: List[str] = []
    seen_patients: set = set()

    for patient_dir in sorted(p for p in src_png_root.iterdir() if p.is_dir()):
        pid = patient_dir.name  # e.g. 'D1-0501'
        for png_path in sorted(patient_dir.glob("*.png")):
            stem = png_path.stem  # e.g. '1-1'
            info = dcm_index.get((pid, stem))
            if info is None:
                missing_dicom_info.append(str(png_path))
                continue
            side = info["side"]
            view = info["view"]
            if side is None or view is None:
                incomplete_tags.append(str(png_path))
                continue

            seen_patients.add(pid)
            clin = clinical.get((pid, side))
            if clin is None:
                missing_clinical.append((pid, side))
                continue

            image_paths.append(str(png_path))
            rows.append({
                "mmap_idx": -1,
                "sample_name": f"{pid}-{side}-{stem}",
                "exam_id": pid,
                "patient_id": pid,
                "side": side,
                "view": view,
                "split": None,
                "bi_rads": None,
                "density": None,
                "cancer_status": clin.get("cancer_status"),
                "finding": clin.get("finding"),
                "manufacturer": "SIEMENS",
                "manufacturer_model": "Mammomat Inspiration",
                "age": None,
                "study_date": None,
                "race_ethnicity": None,
                "site_id": None,
            })

    if missing_dicom_info:
        print(f"[{DATASET}] WARNING: {len(missing_dicom_info)} PNGs without matching DICOM index entry")
        for p in missing_dicom_info[:5]:
            print(f"  {p}")
    if incomplete_tags:
        print(f"[{DATASET}] WARNING: {len(incomplete_tags)} DICOMs with missing side/view tags")
    if missing_clinical:
        unique = sorted(set(missing_clinical))
        print(f"[{DATASET}] WARNING: {len(unique)} (patient, side) pairs missing clinical entry")
    if not image_paths:
        raise RuntimeError("No images resolved.")

    patient_split = make_patient_split(seen_patients, seed=SPLIT_SEED, ratios=SPLIT_RATIOS)
    for row in rows:
        row["split"] = patient_split[row["patient_id"]]

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
