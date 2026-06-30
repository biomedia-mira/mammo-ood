import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import convert_dicom_to_png, resize_image, left_align_mammo


if __name__ == "__main__":
    df = pd.read_csv('/path/to/Mammo/EMBED/tables/EMBED_OpenData_metadata_reduced.csv')
    image_paths = df.anon_dicom_path.values
    orig_dir = Path("/path/to/Mammo/EMBED/images")
    target_dir = Path("/path/to/Mammo/EMBED/pngs")

    error_log_path = "./error_log_embed.txt"

    with open(error_log_path, "w") as error_file:
        for img_path in tqdm(image_paths, total=len(image_paths)):
            try:            
                img_path = img_path.replace('/mnt/NAS2/mammo/anon_dicom/','')
                target_path = target_dir / "full_size" / img_path.replace('.dcm','.png')
                if not os.path.exists(target_path):
                    # save full resolution png                
                    orig_path = orig_dir / img_path
                    png = convert_dicom_to_png(orig_path)               
                    target_path.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_path), png)

                    # save 1024x768 resized png                
                    png = left_align_mammo(png)
                    png = resize_image(image=png, center_fill=False, target_size=(1024, 768))
                    target_path = target_dir / "1024x768" / img_path.replace('.dcm','.png')
                    target_path.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_path), png)
            except Exception as e:
                error_file.write(f"Error processing {img_path}: {e}\n")
