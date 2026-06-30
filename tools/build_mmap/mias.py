"""Build MIAS mmap + metadata.parquet at exam-level granularity.

Source:
  - /path/to/Mammo/MIAS/Info.txt
  - /path/to/Mammo/MIAS/pngs/<H>x<W>/mdb<NNN>.png

MIAS has 322 images = 161 patients x 2 breasts. Each refnum (mdb001..mdb322)
is a single MLO view of one breast. Side is implicit from refnum parity:
odd = right, even = left (per peipa.essex.ac.uk docs). Multiple lesion rows
per refnum are aggregated to image level.

Note: BG codes (F/G/D) in Info.txt are *background tissue* (Fatty / Fatty-
glandular / Dense-glandular), NOT BI-RADS A-D density and not directly
mappable. We leave the density column NaN for MIAS so the canonical schema
vocab is preserved. If you need MIAS background tissue specifically, add a
separate column to the schema.

Patient-level split 70/10/20 by patient (a "patient" = consecutive refnum
pair, e.g. mdb001+mdb002 = patient 1).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.mmap import MMapBuilder, make_patient_split

DATASET = "MIAS"
SRC_INFO = Path("/path/to/Mammo/MIAS/Info.txt")
SRC_PNG_ROOT_FMT = "/path/to/Mammo/MIAS/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/MIAS")
DEFAULT_IMG_SIZE = (1024, 768)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.1, 0.2)

FINDING_MAP = {
    "CIRC": "mass", "SPIC": "mass", "MISC": "mass",
    "CALC": "calcification", "ASYM": "asymmetry", "ARCH": "architectural distortion",
}


def _read_info() -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    with SRC_INFO.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("REFNUM"):
                continue
            parts = line.split()
            ref = parts[0]
            bg = parts[1] if len(parts) > 1 else None
            cls = parts[2] if len(parts) > 2 else None
            sev = parts[3] if cls and cls != "NORM" and len(parts) > 3 else None
            out.setdefault(ref, []).append({"bg": bg, "cls": cls, "sev": sev})
    return out


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    info = _read_info()
    print(f"[{DATASET}] {len(info)} unique refnums")

    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    # Patient ID derived from refnum parity: (n+1)//2
    patient_ids = set()
    for ref in info:
        try:
            n = int(ref.replace("mdb", ""))
            patient_ids.add(str((n + 1) // 2))
        except ValueError:
            pass
    patient_split = make_patient_split(patient_ids, seed=SPLIT_SEED, ratios=SPLIT_RATIOS)

    image_paths: List[str] = []
    rows: List[Dict] = []
    missing: List[str] = []

    for ref in sorted(info):
        try:
            n = int(ref.replace("mdb", ""))
        except ValueError:
            continue
        side = "R" if n % 2 == 1 else "L"
        patient_id = str((n + 1) // 2)

        png_path = src_png_root / f"{ref}.png"
        if not png_path.is_file():
            missing.append(str(png_path))
            continue

        rows_for_ref = info[ref]
        # MIAS BG codes (F/G/D) are background tissue, not ACR density. We do
        # NOT populate the density column to keep the canonical vocab intact.
        density = None

        sevs = {r["sev"] for r in rows_for_ref if r["sev"]}
        if "M" in sevs:
            cancer_status = "Cancer"
        elif sevs:
            cancer_status = "Non-cancer"
        elif any(r["cls"] == "NORM" for r in rows_for_ref):
            cancer_status = "Non-cancer"
        else:
            cancer_status = None

        finds: set = set()
        for r in rows_for_ref:
            cls = r["cls"]
            if cls and cls != "NORM" and cls in FINDING_MAP:
                finds.add(FINDING_MAP[cls])
        finding = sorted(finds) if finds else None

        image_paths.append(str(png_path))
        rows.append({
            "mmap_idx": -1,
            "sample_name": ref,
            "exam_id": f"patient_{patient_id}_{side}",
            "patient_id": f"patient_{patient_id}",
            "side": side,
            "view": "MLO",   # MIAS is single-view (MLO only)
            "split": patient_split[patient_id],
            "bi_rads": None,
            "density": density,
            "cancer_status": cancer_status,
            "finding": finding,
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
