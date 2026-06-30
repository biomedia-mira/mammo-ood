import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import convert_dicom_to_png, resize_image, left_align_mammo


if __name__ == "__main__":
    df = pd.read_csv("/path/to/Mammo/RSNA/train.csv")
    patient_ids = df.patient_id.astype(str).values
    image_ids = df.image_id.astype(str).values
    orig_dir = Path("/path/to/Mammo/RSNA/train_images")
    target_dir = Path("/path/to/Mammo/RSNA/pngs")

    error_log_path = "./error_log_rsna.txt"

    with open(error_log_path, "w") as error_file:
        for img_id, patient_id in tqdm(zip(image_ids, patient_ids), total=len(image_ids)):
            try:
                target_path = target_dir / "full_size" / patient_id / f"{img_id}.png"
                if not os.path.exists(target_path):
                    # save full resolution png
                    rel_path = Path(str(patient_id)) / str(img_id)
                    orig_path = orig_dir / patient_id / f"{img_id}.dcm"
                    png = convert_dicom_to_png(orig_path)                   
                    target_path.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_path), png)
                    
                    # save 1024x768 resized png
                    png = left_align_mammo(png)
                    png = resize_image(image=png, center_fill=False, target_size=(1024, 768))
                    target_path = target_dir / "1024x768" / patient_id / f"{img_id}.png"
                    target_path.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_path), png)
            except Exception as e:
                error_file.write(f"Error processing {img_id} in {patient_id}: {e}\n")
