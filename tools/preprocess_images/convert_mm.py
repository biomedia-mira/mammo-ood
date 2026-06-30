import os
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
from image_utils import resize_image, left_align_mammo, crop_black_border, convert_image_to_png


if __name__ == "__main__":
    # MM (Mammogram Mastery): JPG images, label from folder name (Cancer / Non-Cancer)
    # Only process Original Dataset (Augmented Dataset is derived — skip to avoid double augmentation)
    orig_dir = Path("/path/to/Mammo/MM/Mammogram Mastery A Robust Dataset for Breast Cancer Detection and Medical Education/Breast Cancer Dataset/Original Dataset")
    target_dir = Path("/path/to/Mammo/MM/pngs")

    class_dirs = ["Cancer", "Non-Cancer"]

    error_log_path = "./error_log_mm.txt"

    with open(error_log_path, "w") as error_file:
        for class_name in class_dirs:
            src_dir = orig_dir / class_name
            img_files = sorted(src_dir.glob("*.jpg"))
            for img_file in tqdm(img_files, desc=class_name, total=len(img_files)):
                try:
                    target_path = target_dir / "1024x768" / class_name / img_file.name.replace(".jpg", ".png")
                    if not os.path.exists(target_path):
                        img = convert_image_to_png(str(img_file))

                        # Save full resolution png (before any cropping)
                        target_full = target_dir / "full_size" / class_name / img_file.name.replace(".jpg", ".png")
                        target_full.parent.mkdir(exist_ok=True, parents=True)
                        imsave(str(target_full), img)

                        # Crop large black borders
                        img = crop_black_border(img)

                        img = left_align_mammo(img)
                        img = resize_image(image=img, center_fill=False, target_size=(1024, 768))
                        target_path.parent.mkdir(exist_ok=True, parents=True)
                        imsave(str(target_path), img)

                except Exception as e:
                    error_file.write(f"Error processing {img_file}: {e}\n")
