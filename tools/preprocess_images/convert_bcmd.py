import os
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import convert_dicom_to_png, resize_image, left_align_mammo


if __name__ == "__main__":
    # BCMD: DICOM mammograms + JPG ground-truth annotation overlays
    # Structure: Dataset_v3/{Normal_cases,Suspicious_cases}/{1..50}/
    #   - CC_prior.dcm, CC_recent.dcm, MLO_prior.dcm, MLO_recent.dcm  <- process these
    #   - CC_prior_GT.jpg, CC_recent_GT.jpg, ...                       <- skip (color annotation)
    # Note: some patients have uppercase .DCM extension (e.g. patient 46)
    orig_dir = Path("/path/to/Mammo/BCMD/Dataset_v3")
    target_dir = Path("/path/to/Mammo/BCMD/pngs")

    splits = ["Normal_cases", "Suspicious_cases"]

    error_log_path = "./error_log_bcmd.txt"

    with open(error_log_path, "w") as error_file:
        for split in splits:
            split_dir = orig_dir / split
            patient_dirs = sorted(split_dir.iterdir(), key=lambda p: int(p.name))
            for patient_dir in tqdm(patient_dirs, desc=split):
                # Glob both .dcm and .DCM (case mismatch in dataset)
                dicom_files = list(patient_dir.glob("*.dcm")) + list(patient_dir.glob("*.DCM"))
                for dicom_file in dicom_files:
                    try:
                        target_path = target_dir / "1024x768" / split / patient_dir.name / f"{dicom_file.stem}.png"
                        if not os.path.exists(target_path):
                            img = convert_dicom_to_png(str(dicom_file))

                            # Save full resolution png
                            target_full = target_dir / "full_size" / split / patient_dir.name / f"{dicom_file.stem}.png"
                            target_full.parent.mkdir(exist_ok=True, parents=True)
                            imsave(str(target_full), img)

                            img = left_align_mammo(img)
                            img = resize_image(image=img, center_fill=False, target_size=(1024, 768))
                            target_path.parent.mkdir(exist_ok=True, parents=True)
                            imsave(str(target_path), img)

                    except Exception as e:
                        error_file.write(f"Error processing {dicom_file}: {e}\n")
