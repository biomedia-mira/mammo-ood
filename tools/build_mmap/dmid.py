"""Build DMID mmap + metadata.parquet at exam-level granularity (= per-image fallback).

Source:
  - /path/to/Mammo/DMID/Metadata.xlsx
  - /path/to/Mammo/DMID/Reports/Img{nnn}.txt (per-image radiology reports)
  - /path/to/Mammo/DMID/pngs/<H>x<W>/IMG{nnn}.png

DMID has no patient_id — each IMG{nnn} is a separate image. We use the image
ID itself as exam_id and patient_id (effectively per-image). View column
encodes both side and view as 4-char codes (CCLT/CCRT/MLOLT/MLORT). Patient-
level split 70/10/20 by image ID with seed=42.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.mmap import MMapBuilder, make_patient_split

DATASET = "DMID"
SRC_XLSX = Path("/path/to/Mammo/DMID/Metadata.xlsx")
SRC_REPORT_ROOT = Path("/path/to/Mammo/DMID/Reports")
SRC_PNG_ROOT_FMT = "/path/to/Mammo/DMID/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/DMID")
DEFAULT_IMG_SIZE = (1024, 768)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.1, 0.2)

PATHOLOGY_TO_CANCER = {"M": "Cancer", "B": "Non-cancer", "N": "Non-cancer"}
FINDING_MAP = {
    "CALC": "calcification", "CLAC": "calcification",
    "CIRC": "mass", "SPIC": "mass", "MISC": "mass",
    "ARCH": "architectural distortion", "ASYM": "asymmetry",
}

VIEW_DECODE = {
    "CCLT": ("L", "CC"), "CCRT": ("R", "CC"),
    "MLOLT": ("L", "MLO"), "MLORT": ("R", "MLO"),
}


def _parse_report(image_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (bi_rads, density) parsed from per-image report file."""
    n = image_id.replace("IMG", "")
    path = SRC_REPORT_ROOT / f"Img{n}.txt"
    if not path.exists():
        return None, None
    text = path.read_text(errors="replace")
    density_match = re.search(r"ACR\s*[-:]?\s*([A-D])", text, flags=re.IGNORECASE)
    density = f"Level {density_match.group(1).upper()}" if density_match else None
    birads_match = re.search(r"BI[RA]+DS\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    bi_rads = None
    if birads_match:
        digits = re.findall(r"[1-5]", birads_match.group(1))
        if digits:
            bi_rads = f"Bi-Rads {max(int(d) for d in digits)}"
    return bi_rads, density


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    df = pd.read_excel(SRC_XLSX, header=None, skiprows=31,
                       names=["ID", "view", "bg", "abnormality", "pathology", "x", "y", "radius"])
    df["ID"] = df["ID"].astype(str).str.strip()
    df = df[df["ID"].str.startswith("IMG")].copy()
    print(f"[{DATASET}] {len(df)} rows over {df['ID'].nunique()} images")

    patient_split = make_patient_split(df["ID"].unique(), seed=SPLIT_SEED, ratios=SPLIT_RATIOS)

    image_paths: List[str] = []
    rows: List[Dict] = []
    missing: List[str] = []

    for image_id, group in df.groupby("ID", sort=True):
        png_path = src_png_root / f"{image_id}.png"
        if not png_path.is_file():
            missing.append(str(png_path))
            continue

        view_raw = str(group["view"].iloc[0]).strip().upper().replace(" ", "")
        side, view = VIEW_DECODE.get(view_raw, (None, None))

        # CancerStatus: any Malignant -> Cancer, else Non-cancer if any B/N
        paths = {str(v).strip().upper() for v in group["pathology"].dropna()}
        cancer_status = "Cancer" if "M" in paths else ("Non-cancer" if paths & {"B", "N"} else None)

        # Finding: union of mapped abnormality codes (split on '+')
        findings: set = set()
        for raw in group["abnormality"].dropna():
            for tok in str(raw).strip().upper().replace(" ", "").split("+"):
                if tok and tok != "NORM" and tok in FINDING_MAP:
                    findings.add(FINDING_MAP[tok])
        finding_list = sorted(findings) if findings else None

        bi_rads, density = _parse_report(image_id)

        image_paths.append(str(png_path))
        rows.append({
            "mmap_idx": -1,
            "sample_name": image_id,
            "exam_id": image_id,           # per-image fallback (no patient grouping in DMID)
            "patient_id": image_id,
            "side": side,
            "view": view,
            "split": patient_split[image_id],
            "bi_rads": bi_rads,
            "density": density,
            "cancer_status": cancer_status,
            "finding": finding_list,
            "manufacturer": None,
            "manufacturer_model": None,
            "age": None,
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
    parser = argparse.ArgumentParser(description=f"Build {DATASET} mmap")
    parser.add_argument("--img-size", nargs=2, type=int, default=list(DEFAULT_IMG_SIZE), metavar=("H", "W"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    main(tuple(args.img_size), args.out_root, args.num_workers, args.force_rebuild)
