import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import convert_dbt_to_png, resize_image, left_align_mammo


if __name__ == "__main__":
    # Breast-Cancer-Screening-DBT: 2D digital mammograms (DM) from a DBT screening trial
    # Each series = one view per patient, stored as images/{SeriesUID}/1-1.dcm
    # Labels: Normal / Actionable / Benign / Cancer (per view, from labels CSV)
    # Split: train / validation / test (from v2 file-paths CSVs)
    images_dir = Path("/path/to/Mammo/DBT/images")
    meta_dir = Path("/path/to/Mammo/DBT/metadata")
    target_dir = Path("/path/to/Mammo/DBT/pngs")

    # v2 CSVs (latest) for all splits
    splits = {
        "train": "BCS-DBT-file-paths-train-v2.csv",
        "validation": "BCS-DBT-file-paths-validation-v2.csv",
        "test": "BCS-DBT-file-paths-test-v2.csv",
    }

    error_log_path = "./error_log_dbt.txt"

    with open(error_log_path, "w") as error_file:
        for split, csv_name in splits.items():
            df = pd.read_csv(meta_dir / csv_name)
            for _, row in tqdm(df.iterrows(), desc=split, total=len(df)):
                try:
                    # Extract SeriesUID = second-to-last component of classic_path
                    classic_path = row["classic_path"]
                    series_uid = classic_path.split("/")[-2]
                    dicom_file = images_dir / series_uid / "1-1.dcm"

                    if not dicom_file.exists():
                        error_file.write(f"Missing DICOM: {dicom_file}\n")
                        continue

                    # Output filename: {PatientID}_{StudyUID}_{View}.png
                    out_name = f"{row['PatientID']}_{row['StudyUID']}_{row['View']}.png"

                    target_path = target_dir / "1024x768" / split / out_name
                    if not os.path.exists(target_path):
                        img = convert_dbt_to_png(str(dicom_file), row['View'])

                        # Save full resolution png
                        target_full = target_dir / "full_size" / split / out_name
                        target_full.parent.mkdir(exist_ok=True, parents=True)
                        imsave(str(target_full), img)

                        img = left_align_mammo(img)
                        img = resize_image(image=img, center_fill=False, target_size=(1024, 768))
                        target_path.parent.mkdir(exist_ok=True, parents=True)
                        imsave(str(target_path), img)

                except Exception as e:
                    error_file.write(f"Error processing {row.get('PatientID', '?')} {row.get('View', '?')}: {e}\n")
