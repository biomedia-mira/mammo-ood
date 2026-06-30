import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from skimage.io import imsave
from tqdm import tqdm

from image_utils import convert_dicom_to_png, left_align_mammo, resize_image
from optimam_utils import OPTIMAM_CSV_ROOT, OPTIMAM_IMAGE_ROOT, OPTIMAM_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OPTIMAM DICOM images to 1024x768 PNGs.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes to use.")
    return parser.parse_args()


def convert_one_image(args: tuple[str, str, str]) -> tuple[str, str | None]:
    dcm_rel_path, png_rel_path, target_root = args
    try:
        target_path = Path(target_root) / png_rel_path
        if target_path.exists():
            return "skipped", None

        orig_path = OPTIMAM_IMAGE_ROOT / dcm_rel_path
        if not orig_path.exists():
            raise FileNotFoundError(f"Missing DICOM file: {orig_path}")

        png = convert_dicom_to_png(str(orig_path))
        if png is None:
            raise ValueError(f"Could not decode DICOM pixel data: {orig_path}")
        png = left_align_mammo(png)
        png = resize_image(image=png, center_fill=False, target_size=(1024, 768))

        target_path.parent.mkdir(exist_ok=True, parents=True)
        tmp_path = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp.png")
        imsave(str(tmp_path), png)
        os.replace(tmp_path, target_path)
        return "converted", None
    except Exception as exc:
        return "error", f"Error processing {dcm_rel_path}: {exc}"


if __name__ == "__main__":
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    # Convert all presentation images; classification filtering happens downstream.
    csv_path = OPTIMAM_CSV_ROOT / "optimam.csv"

    target_dir = OPTIMAM_ROOT / "pngs"
    target_root = str(target_dir / "1024x768")
    error_log_path = Path(__file__).resolve().with_name("error_log_optimam.txt")

    df = pd.read_csv(csv_path, low_memory=False)
    image_rows = (
        df[["dcm_path", "image_path"]]
        .dropna(subset=["dcm_path", "image_path"])
        .drop_duplicates(subset=["image_path"])
        .reset_index(drop=True)
    )
    work_items = [
        (str(row.dcm_path), str(row.image_path), target_root)
        for row in image_rows.itertuples(index=False)
    ]

    counts = {"converted": 0, "skipped": 0, "error": 0}
    with open(error_log_path, "w", encoding="utf-8") as error_file:
        if args.num_workers == 1:
            results = map(convert_one_image, work_items)
        else:
            executor = ProcessPoolExecutor(max_workers=args.num_workers)
            results = executor.map(convert_one_image, work_items, chunksize=16)

        try:
            for status, message in tqdm(results, total=len(work_items)):
                counts[status] += 1
                if message:
                    error_file.write(f"{message}\n")
        finally:
            if args.num_workers != 1:
                executor.shutdown(cancel_futures=True)

    print(f"[OPTIMAM] converted={counts['converted']} skipped={counts['skipped']} errors={counts['error']}")
