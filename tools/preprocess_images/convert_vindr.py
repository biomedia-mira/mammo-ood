import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import convert_dicom_to_png, resize_image, left_align_mammo


if __name__ == "__main__":
    df = pd.read_csv("/path/to/Mammo/VinDr-Mammo/breast-level_annotations.csv")
    study_ids = df.study_id.values
    image_ids = df.image_id.values
    orig_dir = Path("/path/to/Mammo/VinDr-Mammo/images")
    target_dir = Path("/path/to/Mammo/VinDr-Mammo/pngs")

    error_log_path = "./error_log_vindr.txt"

    with open(error_log_path, "w") as error_file:
        for img_id, study_id in tqdm(zip(image_ids, study_ids), total=len(image_ids)):
            try:
                target_path = target_dir / "full_size" / study_id / f"{img_id}.png"
                if not os.path.exists(target_path):
                    # save full resolution png
                    rel_path = Path(study_id) / img_id
                    orig_path = orig_dir / study_id / f"{img_id}.dicom"
                    png = convert_dicom_to_png(orig_path)               
                    target_path.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_path), png)
                    
                    # save 1024x768 resized png                
                    png = left_align_mammo(png)
                    png = resize_image(image=png, center_fill=False, target_size=(1024, 768))            
                    target_path = target_dir / "1024x768" / study_id / f"{img_id}.png"
                    target_path.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_path), png)
            except Exception as e:
                error_file.write(f"Error processing {img_id} in {study_id}: {e}\n")
