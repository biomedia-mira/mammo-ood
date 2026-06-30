import os
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import convert_dicom_to_png, resize_image, left_align_mammo, crop_black_border


if __name__ == "__main__":
    # KAU-BCMD DICOM version (v5): same structure as JPG version
    # Filename pattern: {year}_BC{patientID}_{view}_{side}.dcm
    orig_dir = Path("/path/to/Mammo/KAU-BCMD-DICOM")
    target_dir = Path("/path/to/Mammo/KAU-BCMD-DICOM/pngs")

    # DICOM version uses all-caps with space: "BIRAD 1", "BIRAD 3", "BIRAD 4", "BIRAD 5"
    inner_dirs = {'BIRAD 1': 'BIRAD 1', 'BIRAD 3': 'BIRAD 3', 'BIRAD 4': 'BIRAD 4', 'BIRAD 5': 'BIRAD 5'}

    error_log_path = "./error_log_kau_dicom.txt"

    with open(error_log_path, "w") as error_file:
        for birad_folder, inner in inner_dirs.items():
            src_dir = orig_dir / birad_folder / inner
            img_files = sorted(src_dir.glob("*.dcm"))
            for img_file in tqdm(img_files, desc=birad_folder, total=len(img_files)):
                try:
                    target_path = target_dir / "1024x768" / birad_folder / img_file.name.replace('.dcm', '.png')
                    if not os.path.exists(target_path):
                        img = convert_dicom_to_png(str(img_file))

                        # Save full resolution png (before any cropping)
                        target_full = target_dir / "full_size" / birad_folder / img_file.name.replace('.dcm', '.png')
                        target_full.parent.mkdir(exist_ok=True, parents=True)
                        imsave(str(target_full), img)

                        # Crop black borders (same as JPG version)
                        # img = crop_black_border(img) -> dicom version dont need this

                        img = left_align_mammo(img)
                        img = resize_image(image=img, center_fill=False, target_size=(1024, 768))
                        target_path.parent.mkdir(exist_ok=True, parents=True)
                        imsave(str(target_path), img)

                except Exception as e:
                    error_file.write(f"Error processing {img_file}: {e}\n")
