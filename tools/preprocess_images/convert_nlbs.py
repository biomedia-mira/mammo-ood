import os
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import convert_dicom_to_png, resize_image, left_align_mammo


if __name__ == "__main__":
    # NLBS: Full field digital mammography DICOMs, 3 label classes
    # Structure: {abnormal,False Positive,normal}/p_XXXX/{left,left-c,right}/{CC,MLO}/IM-*.dcm
    # Label from top-level folder: abnormal / false_positive / normal
    # Note: 'left-c' means the left breast was the cancerous/abnormal side
    orig_dir = Path("/path/to/Mammo/NLBS")
    target_dir = Path("/path/to/Mammo/NLBS/pngs")

    # Map folder names to clean label names for output
    class_dirs = {
        "abnormal": "abnormal",
        "False Positive": "false_positive",
        "normal": "normal",
    }

    error_log_path = "./error_log_nlbs.txt"

    with open(error_log_path, "w") as error_file:
        for src_name, label in class_dirs.items():
            src_dir = orig_dir / src_name
            patient_dirs = sorted(src_dir.iterdir())
            for patient_dir in tqdm(patient_dirs, desc=label):
                # Each patient has variable side subfolders (left, left-c, right)
                # each containing CC/ and MLO/ with one .dcm file
                dicom_files = list(patient_dir.rglob("*.dcm"))
                for dicom_file in dicom_files:
                    try:
                        # Preserve relative subpath: p_XXXX/side/view/filename.png
                        rel = dicom_file.relative_to(src_dir)
                        target_path = target_dir / "1024x768" / label / rel.with_suffix(".png")
                        if not os.path.exists(target_path):
                            img = convert_dicom_to_png(str(dicom_file))

                            # Save full resolution png
                            target_full = target_dir / "full_size" / label / rel.with_suffix(".png")
                            target_full.parent.mkdir(exist_ok=True, parents=True)
                            imsave(str(target_full), img)

                            img = left_align_mammo(img)
                            img = resize_image(image=img, center_fill=False, target_size=(1024, 768))
                            target_path.parent.mkdir(exist_ok=True, parents=True)
                            imsave(str(target_path), img)

                    except Exception as e:
                        error_file.write(f"Error processing {dicom_file}: {e}\n")
