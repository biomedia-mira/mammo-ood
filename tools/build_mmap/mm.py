"""Build MM (Mammogram Mastery) mmap + metadata.parquet.

Source:
  - /path/to/Mammo/MM/pngs/<H>x<W>/{Cancer,Non-Cancer}/IMG (n).png

MM has NO metadata file; only folder Cancer / Non-Cancer with IMG(n).jpg.
There is no patient_id, side, view, age, manufacturer — anything beyond the
binary cancer label. We use the image filename as both sample_name, exam_id
and patient_id (per-image fallback like DMID). Patient-level split 70/10/20
on these synthetic IDs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.mmap import MMapBuilder, make_patient_split

DATASET = "MM"
SRC_PNG_ROOT_FMT = "/path/to/Mammo/MM/pngs/{h}x{w}"
DEFAULT_OUT_ROOT = Path("/path/to/Mammo/classification_data/MM")
DEFAULT_IMG_SIZE = (1024, 768)
SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.1, 0.2)

CLASS_TO_CANCER = {"Cancer": "Cancer", "Non-Cancer": "Non-cancer"}


def build_metadata_rows(img_size: Tuple[int, int]) -> Tuple[List[str], List[Dict]]:
    src_png_root = Path(SRC_PNG_ROOT_FMT.format(h=img_size[0], w=img_size[1]))
    if not src_png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {src_png_root}")

    candidates: List[Tuple[Path, str]] = []
    for cls_name in ("Cancer", "Non-Cancer"):
        cls_dir = src_png_root / cls_name
        if not cls_dir.is_dir():
            print(f"[{DATASET}] WARNING: missing class folder {cls_dir}")
            continue
        for png in sorted(cls_dir.glob("*.png")):
            if png.name.startswith("._"):
                continue
            candidates.append((png, CLASS_TO_CANCER[cls_name]))

    print(f"[{DATASET}] {len(candidates)} PNGs across Cancer + Non-Cancer folders")

    sample_names = [f"{cs}_{p.stem}" for p, cs in candidates]
    patient_split = make_patient_split(sample_names, seed=SPLIT_SEED, ratios=SPLIT_RATIOS)

    image_paths: List[str] = []
    rows: List[Dict] = []

    for (png_path, cancer_status), name in zip(candidates, sample_names):
        image_paths.append(str(png_path))
        rows.append({
            "mmap_idx": -1,
            "sample_name": name,
            "exam_id": name,           # per-image fallback (no metadata)
            "patient_id": name,
            "side": None,
            "view": None,
            "split": patient_split[name],
            "bi_rads": None,
            "density": None,
            "cancer_status": cancer_status,
            "finding": None,
            "manufacturer": None,
            "manufacturer_model": None,
            "age": None,
            "study_date": None,
            "race_ethnicity": None,
            "site_id": None,
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
        builder.verify(sample_size=16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Build {DATASET} mmap")
    parser.add_argument("--img-size", nargs=2, type=int, default=list(DEFAULT_IMG_SIZE), metavar=("H", "W"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    main(tuple(args.img_size), args.out_root, args.num_workers, args.force_rebuild)
