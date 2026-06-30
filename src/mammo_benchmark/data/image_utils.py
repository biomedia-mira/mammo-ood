import cv2
import pydicom
import numpy as np
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from pydicom.pixel_data_handlers.util import apply_windowing


# Dicom to PNG conversion with windowing and Photometric Interpretation handling 
def convert_dicom_to_png(dicom_file: str) -> np.ndarray:
    data = pydicom.dcmread(dicom_file)
    max_16bit = 2**16 - 1

    try:
        img = data.pixel_array
    except:
        print(f'{dicom_file} Cannot get get pixel_array!')
        return

    # some images have pixel array stored as 8 bit, but all metadata is as if they 
    # were stored as higher bit images. Manually converting to the number of bits
    # expected by the metadata solves conversion issue.
    if data.pixel_array.max() < 256:
        # print(f'Max pixel intensity suggests 8-bit storage {dicom_file}')
        img = apply_windowing(data.pixel_array / 255 * (2 ** data.get('BitsStored') - 1), data, index=0)
    else:
        img = apply_windowing(data.pixel_array, data, index=0)
    
    # ensure the result is in bit 16
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max() 
    img *= max_16bit
    
    if data.get('PhotometricInterpretation') == 'MONOCHROME1':
        img = max_16bit - img

    return img.astype(np.uint16)


# Standard Image (JPEG/PNG) to PNG conversion with 16-bit min-max normalization mapping
def convert_image_to_png(image_path: str) -> np.ndarray:
    from skimage.io import imread
    img = imread(image_path)
    
    # Drop colour channels if RGB (mammograms are grayscale)
    if img.ndim == 3:
        img = img[:, :, 0]
        
    # Force 0 to 65535 dynamic range
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max() 
    img *= 65535.0
    
    return img.astype(np.uint16)

def convert_dbt_to_png(dicom_file: str, view: str, index: int = None) -> np.ndarray:
    """
    Read pixel array from DBT DICOM file (3D Tomosynthesis)
    and extract a specific slice as a 2D PNG. If index is None, extracts the middle slice.
    Applies orientation correction as per Duke's dataset repository.
    """
    ds = pydicom.dcmread(dicom_file)
    ds.decompress(handler_name="pylibjpeg")
    
    # 3D pixel array: (slices, height, width)
    pixel_array = ds.pixel_array
    
    # Extract slice
    if index is None:
        index = pixel_array.shape[0] // 2
    
    # Orientation correction to fix malformed DICOMs (per Duke's repo)
    # We check laterality on the slice we're actually extracting based on pixel columns sum
    left_edge = np.sum(pixel_array[index][:, 0])
    right_edge = np.sum(pixel_array[index][:, -1])
    image_laterality = "R" if left_edge < right_edge else "L"
    
    pixel_array = pixel_array[index]
    
    view_laterality = view[0].upper()
    if image_laterality != view_laterality:
        pixel_array = np.flip(pixel_array, axis=(-1, -2))
        
    # Windowing
    window_center = np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1050].value)
    window_width = np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1051].value)
    low = (2 * window_center - window_width) / 2
    high = (2 * window_center + window_width) / 2
    
    pixel_array = rescale_intensity(
        pixel_array, in_range=(low, high), out_range=(0, 65535)
    )
    
    # Photometric Interpretation
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array = np.invert(pixel_array.astype(np.uint16))
        
    # Return as 16-bit
    return pixel_array.astype(np.uint16)

# RGB/Secondary Capture Dicom to PNG conversion with Grayscale
def convert_rgb_dicom_to_png(dicom_file: str) -> np.ndarray:
    data = pydicom.dcmread(dicom_file)
    
    try:
        arr = data.pixel_array
    except:
        print(f'{dicom_file} Cannot get pixel_array!')
        return None

    # Handle RGB to Grayscale
    if arr.ndim == 3 and arr.shape[2] == 3:
        # standard RGB -> Grayscale luminance weights
        gray = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    elif arr.ndim == 3:
        gray = arr[:, :, 0] # fallback just take first channel
    else:
        gray = arr

    # Force 0 to 65535 dynamic range
    gray = gray.astype(np.float32)
    gray -= gray.min()
    if gray.max() > 0:
        gray /= gray.max() 
    gray *= 65535.0

    return gray.astype(np.uint16)


# Aspect ratio preserving resize ensuring that the image fits within the target size
def resize_image(image: np.ndarray, center_fill: bool, target_size) -> np.ndarray:
    
    if image.shape == target_size:
        return image
    else:
        new_image = np.zeros(target_size)
        
        # Original image size
        orig_height, orig_width = image.shape
        
        resize_width = np.ceil(target_size[0] / orig_height * orig_width).astype(np.uint16)

        if resize_width <= target_size[1]:
            resize_size = np.array((target_size[0], resize_width)).astype(np.uint16)
            
            # Resize image
            image_resized = resize(image, output_shape=resize_size, preserve_range=True)

            # Paste into target size with padding
            if center_fill:
                idx_start = np.ceil((target_size[1]-resize_width)/2).astype(np.uint16)
                new_image[:,idx_start:idx_start + resize_width] = image_resized
            else:
                new_image[:,0:resize_width] = image_resized
        else:
            resize_height = np.ceil(target_size[1] / orig_width * orig_height).astype(np.uint16)
            
            resize_size = np.array((resize_height, target_size[1])).astype(np.uint16)
            
            # Resize image
            image_resized = resize(image, output_shape=resize_size, preserve_range=True)

            # Paste into target size with padding
            if center_fill:
                idx_start = np.ceil((target_size[0]-resize_height)/2).astype(np.uint16)
                new_image[idx_start:idx_start + resize_height,:] = image_resized
            else:
                new_image[0:resize_height,:] = image_resized            

        return new_image.astype(image.dtype)


# Ensures left alignment of mammogram images
def left_align_mammo(image: np.ndarray) -> np.ndarray:
    l = np.mean(image[:,0:int(image.shape[1]/2)])
    r = np.mean(image[:,int(image.shape[1]/2)::])
    if l < r:
        image = image[:, ::-1].copy()

    return image


# Crops black borders from a mammogram using dynamic thresholding, morphological
# open/close, and connected components to isolate the largest breast region.
# Falls back to the original if the result is suspiciously small (<5% of image).
def crop_black_border(image: np.ndarray, margin: int = 3) -> np.ndarray:
    thresh = int(np.percentile(image, 5)) + 200
    mask = (image > thresh).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num < 2:
        return image

    best_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    x, y, w, h, area = stats[best_idx]

    r1, r2 = max(0, y - margin), min(image.shape[0], y + h + margin)
    c1, c2 = max(0, x - margin), min(image.shape[1], x + w + margin)

    cropped = image[r1:r2, c1:c2]
    return cropped if cropped.size > (image.size * 0.05) else image
