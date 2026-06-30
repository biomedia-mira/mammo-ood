"""Build the EMBED `.npy` mmap used by SSL pretraining.

The pretraining dataloaders read `embed_pretrain.csv`, apply the same EMBED
filters below, then assign `global_index = 0..N-1`. For the mmap indices to
match training, this script must preserve the filtered CSV row order exactly.

Output shape is `(N, 1, H, W)` with float32 pixels. Pixel values keep the
original PNG dynamic range; background outside the largest foreground component
is zeroed, matching the preprocessing used for the paper pretraining branch.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte
from tqdm import tqdm


DEFAULT_CSV = Path("data/embed_pretrain.csv")
DEFAULT_IMAGE_ROOT = Path("/path/to/Mammo/EMBED/pngs/1024x768")
DEFAULT_OUTPUT = Path("data/embed_processed_512x384_pretrain.npy")
DEFAULT_IMAGE_SIZE = (512, 384)

_WORKER_MMAP: Optional[np.ndarray] = None
_WORKER_IMAGE_SIZE: Optional[Tuple[int, int]] = None


def read_filtered_embed_csv(csv_file: Path) -> pd.DataFrame:
    """Read and filter EMBED rows exactly as the pretraining dataloader does."""
    df = pd.read_csv(csv_file, low_memory=False)
    required = {
        "FinalImageType",
        "GENDER_DESC",
        "tissueden",
        "ViewPosition",
        "spot_mag",
        "image_path",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_file} is missing required columns: {missing}")

    df = df[df["FinalImageType"] == "2D"]
    df = df[df["GENDER_DESC"] == "Female"]
    df = df[df["tissueden"].notna()]
    df = df[df["tissueden"] < 5]
    df = df[df["ViewPosition"].isin(["MLO", "CC"])]
    df = df[df["spot_mag"].isna()]
    return df.copy().reset_index(drop=True)


def preprocess_pretrain_image(image_path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    """Load one PNG and return a float32 array of shape `(1, H, W)`."""
    image = imread(str(image_path)).astype(np.float32)
    if image.ndim == 3:
        image = image[..., 0]

    if image.shape != image_size:
        image = resize(image, output_shape=image_size, preserve_range=True)

    img_norm = image - float(np.min(image))
    img_norm = img_norm / (float(np.max(img_norm)) + 1e-8)
    thresh = cv2.threshold(img_as_ubyte(img_norm), 5, 255, cv2.THRESH_BINARY)[1]

    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=4)
    if nb_components > 1:
        max_label, _ = max(
            ((i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)),
            key=lambda x: x[1],
        )
        image[output != max_label] = 0.0

    return image[None, :, :].astype(np.float32, copy=False)


def init_worker(output_mmap: str, shape: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> None:
    global _WORKER_MMAP, _WORKER_IMAGE_SIZE
    _WORKER_MMAP = np.lib.format.open_memmap(output_mmap, mode="r+", dtype="float32", shape=shape)
    _WORKER_IMAGE_SIZE = image_size


def process_one(task: Tuple[int, str]) -> Tuple[int, bool, str]:
    idx, image_path = task
    try:
        if _WORKER_MMAP is None or _WORKER_IMAGE_SIZE is None:
            raise RuntimeError("worker mmap was not initialized")
        arr = preprocess_pretrain_image(Path(image_path), _WORKER_IMAGE_SIZE)
        _WORKER_MMAP[idx] = arr
        return idx, True, ""
    except Exception as exc:  # noqa: BLE001 - report any preprocessing failure
        return idx, False, f"{type(exc).__name__}: {exc}"


def default_num_workers() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return mp.cpu_count()


def write_failed_csv(path: Path, failures: Iterable[Tuple[int, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "image_path", "error"])
        writer.writeheader()
        for idx, image_path, error in failures:
            writer.writerow({"index": idx, "image_path": image_path, "error": error})


def build_pretrain_mmap(
    *,
    csv_file: Path,
    image_root: Path,
    output_mmap: Path,
    image_size: Tuple[int, int],
    num_workers: int,
    force: bool,
    max_images: Optional[int],
    filtered_csv_out: Optional[Path],
    failed_csv: Path,
    flush_every: int,
) -> None:
    df = read_filtered_embed_csv(csv_file)
    if max_images is not None:
        df = df.head(max_images).copy().reset_index(drop=True)

    if filtered_csv_out is not None:
        filtered_csv_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filtered_csv_out, index=False)
        print(f"Wrote filtered CSV: {filtered_csv_out} ({len(df):,} rows)")

    image_paths: List[str] = [str(image_root / str(path)) for path in df["image_path"].values]
    missing = [(idx, path, "file does not exist") for idx, path in enumerate(image_paths) if not Path(path).exists()]
    if missing:
        write_failed_csv(failed_csv, missing)
        raise FileNotFoundError(f"{len(missing):,} image files are missing. See {failed_csv}")

    n_images = len(image_paths)
    if n_images == 0:
        raise ValueError("No EMBED images remain after pretraining filters")

    shape = (n_images, 1, image_size[0], image_size[1])
    output_mmap.parent.mkdir(parents=True, exist_ok=True)
    if output_mmap.exists():
        if not force:
            raise FileExistsError(f"{output_mmap} already exists. Use --force to overwrite.")
        output_mmap.unlink()

    print(f"Filtered images: {n_images:,}")
    print(f"Allocating mmap: {output_mmap}")
    print(f"Shape: {shape}; dtype=float32")
    mmap_main = np.lib.format.open_memmap(output_mmap, mode="w+", dtype="float32", shape=shape)
    mmap_main.flush()

    failures: List[Tuple[int, str, str]] = []
    tasks = [(idx, path) for idx, path in enumerate(image_paths)]
    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(str(output_mmap), shape, image_size),
    ) as pool:
        for n_done, (idx, ok, error) in enumerate(
            tqdm(pool.imap_unordered(process_one, tasks, chunksize=64), total=n_images),
            start=1,
        ):
            if not ok:
                failures.append((idx, image_paths[idx], error))
            if n_done % flush_every == 0:
                mmap_main.flush()

    mmap_main.flush()
    del mmap_main

    if failures:
        write_failed_csv(failed_csv, failures)
        raise RuntimeError(f"{len(failures):,} images failed. See {failed_csv}")

    arr = np.load(output_mmap, mmap_mode="r")
    if arr.shape != shape or arr.dtype != np.dtype("float32"):
        raise RuntimeError(f"Unexpected output mmap: shape={arr.shape}, dtype={arr.dtype}")
    print(f"Done: {output_mmap}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EMBED pretraining image mmap.")
    parser.add_argument("--csv-file", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-mmap", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, nargs=2, default=DEFAULT_IMAGE_SIZE, metavar=("H", "W"))
    parser.add_argument("--num-workers", type=int, default=default_num_workers())
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output mmap.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit on the number of images.")
    parser.add_argument(
        "--filtered-csv-out",
        type=Path,
        default=None,
        help="Optional CSV matching the rows written to the mmap. Useful with --max-images.",
    )
    parser.add_argument(
        "--failed-csv",
        type=Path,
        default=Path("outputs/embed_pretrain_mmap_failed.csv"),
        help="CSV path for missing/failed image records.",
    )
    parser.add_argument("--flush-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_pretrain_mmap(
        csv_file=args.csv_file,
        image_root=args.image_root,
        output_mmap=args.output_mmap,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        num_workers=int(args.num_workers),
        force=bool(args.force),
        max_images=args.max_images,
        filtered_csv_out=args.filtered_csv_out,
        failed_csv=args.failed_csv,
        flush_every=int(args.flush_every),
    )


if __name__ == "__main__":
    main()
