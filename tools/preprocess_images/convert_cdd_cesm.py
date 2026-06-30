import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import resize_image


if __name__ == "__main__":
    # CDD-CESM: JPEG 8-bit (no DICOM)
    # We only process "Low energy images" (type=DM) — standard mammogram equivalent.
    # Subtracted (CESM) images are not used for standard mammography classification.
    # Filename pattern: P{id}_{side}_{type}_{view}.jpg  e.g. P1_L_DM_CC.jpg
    # Convention: L breast stored tissue-LEFT, R breast stored tissue-RIGHT.
    # → flip R images horizontally so all output has tissue on the LEFT.
    base_dir = Path("/path/to/Mammo/CDD-CESM/PKG - CDD-CESM/CDD-CESM")
    low_dir = base_dir / "Low energy images of CDD-CESM"
    target_dir = Path("/path/to/Mammo/CDD-CESM/pngs")

    error_log_path = "./error_log_cdd_cesm.txt"

    img_files = sorted([f for f in low_dir.glob("*.jpg") if f.name != "desktop.ini"])

    with open(error_log_path, "w") as error_file:
        for img_path in tqdm(img_files, total=len(img_files)):
            try:
                target_path = target_dir / "1024x768" / img_path.name.replace('.jpg', '.png')
                if not os.path.exists(target_path):
                    from image_utils import convert_image_to_png
                    img = convert_image_to_png(str(img_path))

                    # Save full resolution png
                    from image_utils import left_align_mammo
                    target_full = target_dir / "full_size" / img_path.name.replace('.jpg', '.png')
                    target_full.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_full), img)

                    # Align and Resize
                    img = left_align_mammo(img)
                    img = resize_image(image=img, center_fill=False, target_size=(1024, 768))
                    target_path.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_path), img)

            except Exception as e:
                error_file.write(f"Error processing {img_path}: {e}\n")
