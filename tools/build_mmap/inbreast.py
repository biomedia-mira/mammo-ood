"""Build INbreast mmap + metadata.parquet at exam-level granularity.

Source:
  - /path/to/Mammo/INbreast/INbreast Release 1.0/INbreast.xls
  - /path/to/Mammo/INbreast/INbreast Release 1.0/AllDICOMs/<file_id>_<patient_hash>_MG_<L|R>_<view>_ANON.dcm
  - /path/to/Mammo/INbreast/pngs/<H>x<W>/<file_id>.png

The INbreast xls 'Patient ID' column is anonymized (single value 'removed').
We extract the real patient_hash from each DICOM filename. exam_id =
(patient_hash, side); patient split 70/10/20 by patient_hash.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.mmap import MMapBuilder, make_patient_split

DATASET = "INbreast"
SRC_XLS = Path("/path/to/Mammo/INbreast/INbreast Release 1.0/INbreast.xls")
SRC_DICOMS_DIR = Path("/path/to/Mammo/INbreast/INbreast Release 1.0/AllDICOMs")
SRC_PNG_ROOT_FMT = "/path/to/Mammo/INbreast/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/INbreast")
DEFAULT_IMG_SIZE = (1024, 768)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.1, 0.2)

ACR_MAP = {1: "Level A", 2: "Level B", 3: "Level C", 4: "Level D",
           "1": "Level A", "2": "Level B", "3": "Level C", "4": "Level D"}


def _map_birads(v) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s in ("4a", "4b", "4c"):
        return "Bi-Rads 4"
    if s == "6":
        return None
    if s in ("0", "1", "2", "3", "5"):
        return f"Bi-Rads {s}"
    return None


def _build_finding(row) -> List[str]:
    """Build finding list from per-column flags. Returns empty list if no
    findings — caller treats empty as None (no 'Normal' marker, matching
    the convention of VinDr / CBIS-DDSM / DMID / MIAS in the new pipeline).
    """
    out = []
    for col, name in [("Mass", "mass"), ("Micros", "calcification"),
                      ("Distortion", "architectural distortion"), ("Asymmetry", "asymmetry")]:
        if col in row.index and str(row[col]).strip() == "X":
            out.append(name)
    return out


def _index_dicoms() -> Dict[str, Tuple[str, str, str, str]]:
    """file_id -> (patient_hash, side, view, dicom_stem).

    INbreast filenames look like '{file_id}_{patient_hash}_MG_{L|R}_{CC|ML|MLO}_ANON.dcm'.
    The dataset uses both 'ML' and 'MLO' for the mediolateral oblique view (per
    the INbreast paper, the two standard projections are CC and MLO; 'ML' here
    is just the shorter form used inconsistently in filenames). We map both
    to 'MLO' for the schema.
    """
    out: Dict[str, Tuple[str, str, str, str]] = {}
    if not SRC_DICOMS_DIR.exists():
        return out
    pat = re.compile(r"^(\d+)_([0-9a-f]+)_MG_([RL])_(CC|MLO|ML)_")
    for f in SRC_DICOMS_DIR.iterdir():
        m = pat.match(f.name)
        if m:
            view = "MLO" if m.group(4) in ("ML", "MLO") else "CC"
            out[m.group(1)] = (m.group(2), m.group(3), view, f.stem)
    return out


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    df = pd.read_excel(SRC_XLS)
    df.columns = df.columns.str.strip()
    # Rename columns with spaces so .itertuples gives clean attribute access
    # (pandas sanitizes 'File Name' -> '_2' or similar inside namedtuples).
    df = df.rename(columns={"File Name": "file_name", "Bi-Rads": "bi_rads_raw", "ACR": "acr"})
    df["file_name"] = df["file_name"].astype(str).str.split(".").str[0].str.strip()
    file_index = _index_dicoms()
    print(f"[{DATASET}] xls rows: {len(df)}, indexed DICOMs: {len(file_index)}")

    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    patient_hashes = {pid for (pid, _, _, _) in file_index.values()}
    patient_split = make_patient_split(patient_hashes, seed=SPLIT_SEED, ratios=SPLIT_RATIOS)

    image_paths: List[str] = []
    rows: List[Dict] = []
    missing: List[str] = []
    no_match: List[str] = []

    for r in df.itertuples(index=False):
        file_name = str(r.file_name)
        info = file_index.get(file_name)
        if info is None:
            no_match.append(file_name)
            continue
        patient_hash, side, view, dicom_stem = info

        png_path = src_png_root / f"{dicom_stem}.png"
        if not png_path.is_file():
            missing.append(str(png_path))
            continue

        density = ACR_MAP.get(r.acr)
        bi_rads = _map_birads(r.bi_rads_raw)
        finding = _build_finding(pd.Series(r._asdict()))

        image_paths.append(str(png_path))
        rows.append({
            "mmap_idx": -1,
            "sample_name": file_name,
            "exam_id": patient_hash,  # patient-level exam (1 visit/patient)
            "patient_id": patient_hash,
            "side": side,
            "view": view,
            "split": patient_split[patient_hash],
            "bi_rads": bi_rads,
            "density": density,
            "cancer_status": None,
            "finding": finding if finding else None,
            "manufacturer": None,
            "manufacturer_model": None,
            "age": None,
            "study_date": None,
            "race_ethnicity": None,
            "site_id": None,
        })

    if no_match:
        print(f"[{DATASET}] WARNING: {len(no_match)} xls rows without DICOM filename match")
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
    parser = argparse.ArgumentParser(description=f"Build {DATASET} mmap")
    parser.add_argument("--img-size", nargs=2, type=int, default=list(DEFAULT_IMG_SIZE), metavar=("H", "W"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    main(tuple(args.img_size), args.out_root, args.num_workers, args.force_rebuild)
