import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import convert_dicom_to_png, resize_image, left_align_mammo


if __name__ == "__main__":
    csv_path = "/path/to/Mammo/INbreast/INbreast Release 1.0/INbreast.csv"
    orig_dir = Path("/path/to/Mammo/INbreast/INbreast Release 1.0/AllDICOMs")
    target_dir = Path("/path/to/Mammo/INbreast/pngs")

    # INbreast.csv uses semicolon as delimiter
    # Columns: Patient ID; Patient age; Laterality; View; Acquisition date; File Name; ACR; Bi-Rads
    df = pd.read_csv(csv_path, sep=";")

    error_log_path = "./error_log_inbreast.txt"

    with open(error_log_path, "w") as error_file:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # File Name column is the numeric prefix of the DICOM filename
                file_id = str(int(row["File Name"]))

                # Locate DICOM: filename pattern is {file_id}_{hash}_MG_{side}_{view}_ANON.dcm
                matches = list(orig_dir.glob(f"{file_id}_*.dcm"))
                if not matches:
                    error_file.write(f"Not found: {file_id}\n")
                    continue
                dcm_path = matches[0]

                target_path = target_dir / "full_size" / (dcm_path.stem + ".png")
                if not os.path.exists(target_path):
                    # save full resolution png
                    png = convert_dicom_to_png(dcm_path)
                    target_path.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_path), png)

                    # save 1024x768 resized png
                    png = left_align_mammo(png)
                    png = resize_image(image=png, center_fill=False, target_size=(1024, 768))
                    target_path = target_dir / "1024x768" / (dcm_path.stem + ".png")
                    target_path.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_path), png)

            except Exception as e:
                error_file.write(f"Error processing {row.get('File Name', '?')}: {e}\n")
