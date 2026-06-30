import os
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import resize_image, left_align_mammo, crop_black_border, convert_image_to_png


if __name__ == "__main__":
    # MIAS: 8-bit PGM images (1024x1024), no DICOM, no train/test split
    # Filename pattern: mdb{id}.pgm, label from Info.txt
    # Images have large black borders — crop_black_border required
    orig_dir = Path("/path/to/Mammo/MIAS/all-mias")
    target_dir = Path("/path/to/Mammo/MIAS/pngs")

    error_log_path = "./error_log_mias.txt"

    img_files = sorted(orig_dir.glob("*.pgm"))

    with open(error_log_path, "w") as error_file:
        for img_file in tqdm(img_files, total=len(img_files)):
            try:
                target_path = target_dir / "1024x768" / img_file.name.replace(".pgm", ".png")
                if not os.path.exists(target_path):
                    img = convert_image_to_png(str(img_file))

                    # Save full resolution png (before any cropping)
                    target_full = target_dir / "full_size" / img_file.name.replace(".pgm", ".png")
                    target_full.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_full), img)

                    # Crop black borders (MIAS images have significant black padding)
                    img = crop_black_border(img)

                    img = left_align_mammo(img)
                    img = resize_image(image=img, center_fill=False, target_size=(1024, 768))
                    target_path.parent.mkdir(exist_ok=True, parents=True)
                    imsave(str(target_path), img)

            except Exception as e:
                error_file.write(f"Error processing {img_file}: {e}\n")
