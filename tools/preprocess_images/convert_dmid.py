import os
import numpy as np
import pydicom
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import resize_image
from image_utils import convert_dicom_to_png, convert_rgb_dicom_to_png


if __name__ == "__main__":
    # DMID: 510 flat DCM files (IMG001.dcm … IMG510.dcm), RGB Secondary Capture
    # No standard mammography DICOM tags or laterality parsing needed (handled by left_align_mammo)
    orig_dir    = Path("/path/to/Mammo/DMID/DICOM Images")
    target_dir  = Path("/path/to/Mammo/DMID/pngs")

    dcm_files = sorted(orig_dir.glob("*.dcm"))
    error_log_path = "./error_log_dmid.txt"

    with open(error_log_path, "w") as error_file:
        for dcm_path in tqdm(dcm_files, total=len(dcm_files)):
            try:
                img_id = dcm_path.stem  # e.g. IMG001

                target_path = target_dir / "full_size" / (img_id + ".png")
                if not os.path.exists(target_path):
                    # Convert RGB DICOM → uint16 grayscale
                    png = convert_rgb_dicom_to_png(str(dcm_path))

                    # Save full resolution
                    target_path.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_path), png)

                    # Align and Resize
                    from image_utils import left_align_mammo
                    png = left_align_mammo(png)

                    png = resize_image(image=png, center_fill=False, target_size=(1024, 768))
                    target_small = target_dir / "1024x768" / (img_id + ".png")
                    target_small.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_small), png)

            except Exception as e:
                error_file.write(f"Error {dcm_path}: {e}\n")
