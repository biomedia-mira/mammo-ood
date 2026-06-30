import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import resize_image, left_align_mammo, crop_black_border, convert_image_to_png


if __name__ == "__main__":
    # KAU-BCMD: JPEG 8-bit images (no DICOM), label from folder name
    # Filename pattern: {year}_BC{patientID}_ {view}_{side}.jpg
    orig_dir = Path("/path/to/Mammo/KAU-BCMD")
    target_dir = Path("/path/to/Mammo/KAU-BCMD/pngs")

    # inner subfolder for each BIRADS class
    inner_dirs = {'BIRAD1': 'b1', 'Birad3': 'b3', 'Birad4': 'b4', 'Birad5': 'Birad5'}

    error_log_path = "./error_log_kau.txt"

    with open(error_log_path, "w") as error_file:
        for birad_folder, inner in inner_dirs.items():
            src_dir = orig_dir / birad_folder / inner
            img_files = sorted(src_dir.glob("*.jpg"))
            for img_file in tqdm(img_files, desc=birad_folder, total=len(img_files)):
                try:
                    target_path = target_dir / "1024x768" / birad_folder / img_file.name.replace('.jpg', '.png')
                    if not os.path.exists(target_path):
                        img = convert_image_to_png(str(img_file))

                        # Save full resolution png (before any cropping)
                        target_full = target_dir / "full_size" / birad_folder / img_file.name.replace('.jpg', '.png')
                        target_full.parent.mkdir(exist_ok=True, parents=True)
                        imsave(str(target_full), img)

                        # Crop large black borders common in KAU-BCMD raw JPEGs
                        img = crop_black_border(img)

                        img = left_align_mammo(img)
                        img = resize_image(image=img, center_fill=False, target_size=(1024, 768))
                        target_path.parent.mkdir(exist_ok=True, parents=True)
                        imsave(str(target_path), img)

                except Exception as e:
                    error_file.write(f"Error processing {img_file}: {e}\n")
