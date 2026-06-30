import os
import pydicom
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import convert_dicom_to_png, resize_image, left_align_mammo


if __name__ == "__main__":
    # CMMD: DICOM stored in SeriesInstanceUID folders
    # Each folder may contain multiple DCM files (1-1.dcm, 1-2.dcm, ... for multi-frame)
    # PatientID is read from DICOM header (format: D{number}-{id}, e.g. D2-0512)
    # Laterality is read from DICOM ImageLaterality tag (reliable, verified on dataset)
    cmmd_dir = Path("/path/to/Mammo/CMMD")
    target_dir = Path("/path/to/Mammo/CMMD/pngs")

    # Collect all series dirs (skip non-directory entries)
    series_dirs = sorted([d for d in cmmd_dir.iterdir() if d.is_dir()])

    error_log_path = "./error_log_cmmd.txt"

    with open(error_log_path, "w") as error_file:
        for series_dir in tqdm(series_dirs, total=len(series_dirs)):
            for dcm_path in sorted(series_dir.glob("*.dcm")):
                try:
                    # Read PatientID and Laterality from DICOM header (no pixel data needed)
                    dcm_meta = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
                    patient_id = str(dcm_meta.get("PatientID", series_dir.name))
                    laterality = str(dcm_meta.get("ImageLaterality") or
                                     dcm_meta.get("Laterality") or "")

                    target_path = target_dir / "full_size" / patient_id / (dcm_path.stem + ".png")
                    if not os.path.exists(target_path):
                        # save full resolution png
                        png = convert_dicom_to_png(str(dcm_path))
                        target_path.parent.mkdir(exist_ok=True, parents=True)
                        imsave(str(target_path), png)

                        # save 1024x768 resized png
                        png = left_align_mammo(png)

                        png = resize_image(image=png, center_fill=False, target_size=(1024, 768))
                        target_small = target_dir / "1024x768" / patient_id / (dcm_path.stem + ".png")
                        target_small.parent.mkdir(exist_ok=True, parents=True)
                        imsave(str(target_small), png)

                except Exception as e:
                    error_file.write(f"Error {dcm_path}: {e}\n")
