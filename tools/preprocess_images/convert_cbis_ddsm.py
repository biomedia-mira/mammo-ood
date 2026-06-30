import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import convert_dicom_to_png, resize_image, left_align_mammo


if __name__ == "__main__":
    # CBIS-DDSM: DICOM stored in SeriesUID folders
    # CSV path format: patient_folder/StudyUID/SeriesUID/000000.dcm
    #   parts[0] = patient_folder  (e.g. Mass-Training_P_00001_LEFT_CC)
    #   parts[1] = StudyUID        (does NOT exist as folder on disk)
    #   parts[2] = SeriesUID       (EXISTS as folder on disk, contains 1-1.dcm)
    #   parts[3] = 000000.dcm      (instance filename in CSV, actual file is 1-1.dcm on disk)
    # Laterality: use left_align_mammo (intensity-based) — filename convention not consistent.
    base_dir = Path("/path/to/Mammo/CBIS-DDSM")
    target_dir = Path("/path/to/Mammo/CBIS-DDSM/pngs")

    # Load all 4 CSVs (mass + calc, train + test)
    dfs = []
    for csv_file in [
        "mass_case_description_train_set.csv",
        "calc_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "calc_case_description_test_set.csv",
    ]:
        df = pd.read_csv(base_dir / csv_file)
        df["split"] = "train" if "train" in csv_file else "test"
        dfs.append(df)

    # Drop duplicate image paths (same image may appear in mass + calc CSVs)
    all_data = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["image file path"])

    error_log_path = "./error_log_cbis_ddsm.txt"

    with open(error_log_path, "w") as error_file:
        for _, row in tqdm(all_data.iterrows(), total=len(all_data)):
            try:
                # Parse CSV path: patient_folder/StudyUID/SeriesUID/000000.dcm
                img_path_parts = row["image file path"].strip().split("/")
                patient_folder = img_path_parts[0]   # e.g. Mass-Training_P_00001_LEFT_CC
                series_uid = img_path_parts[2]        # SeriesUID — exists as folder on disk

                # Locate DICOM: series folder contains 1-1.dcm (and sometimes 1-2.dcm)
                dcm_dir = base_dir / series_uid
                dcm_path = dcm_dir / "1-1.dcm"
                if not dcm_path.exists():
                    dcms = [f for f in dcm_dir.glob("*.dcm")] if dcm_dir.exists() else []
                    if not dcms:
                        error_file.write(f"Not found: {series_uid}\n")
                        continue
                    dcm_path = sorted(dcms)[0]

                target_path = target_dir / "full_size" / patient_folder / (series_uid + ".png")
                if not os.path.exists(target_path):
                    # save full resolution png
                    png = convert_dicom_to_png(dcm_path)
                    target_path.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_path), png)

                    # save 1024x768 resized png
                    png = left_align_mammo(png)
                    png = resize_image(image=png, center_fill=False, target_size=(1024, 768))
                    t = target_dir / "1024x768" / patient_folder / (series_uid + ".png")
                    t.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(t), png)

            except Exception as e:
                error_file.write(f"Error {row.get('image file path', '?')}: {e}\n")
