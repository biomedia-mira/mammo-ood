"""Image preprocessing for mmap-backed mammography datasets.

This matches `pre-training/util/offline_mmap_creator.py` exactly: pixel values
are stored as float32 with the original dynamic range preserved (no
normalization), and the breast region is segmented so background pixels are
zeroed out. Per-batch normalization happens at training time, after the mmap
read.

Storing float32 keeps full precision for 16-bit source PNGs (mammography PNGs
converted from DICOM by `utils/convert_*.py` are typically 16-bit). The mask
is computed on a uint8-normalized copy for thresholding, then applied back to
the original float32 image.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte


def _largest_breast_mask(image_uint8: np.ndarray) -> np.ndarray:
    """Return a binary mask of the largest connected foreground region.

    Threshold at 5/255, find connected components, keep the largest by area.
    Falls back to all-ones if thresholding finds nothing.
    """
    thresh = cv2.threshold(image_uint8, 5, 255, cv2.THRESH_BINARY)[1]
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=4)
    if nb_components <= 1:
        return np.ones_like(image_uint8, dtype=bool)
    max_label, _ = max(
        ((i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)),
        key=lambda x: x[1],
    )
    return output == max_label


def preprocess_image(image_path: str, output_size: Tuple[int, int]) -> np.ndarray:
    """Load an image from `image_path`, preprocess, return float32 (1, H, W).

    Steps (kept identical to `pre-training/util/offline_mmap_creator.py`):
      1. Read the image (PNG / TIFF / JPG via skimage), cast to float32.
      2. Reduce to single channel if RGB.
      3. Resize to `output_size = (H, W)` if not already that shape
         (preserve_range=True so original pixel values stay).
      4. Compute a min-max normalized uint8 copy and threshold it to find the
         largest connected component (the breast).
      5. Zero out background pixels on the ORIGINAL float32 image.
      6. Add a leading channel axis.

    Returns a numpy array of shape (1, H, W) and dtype float32.
    """
    image = imread(image_path).astype(np.float32)
    if image.ndim == 3:
        image = image[..., 0]

    if image.shape != output_size:
        image = resize(image, output_shape=output_size, preserve_range=True, anti_aliasing=True)

    img_norm = image - float(image.min())
    denom = float(img_norm.max()) + 1e-8
    img_norm = img_norm / denom
    image_uint8_for_mask = img_as_ubyte(np.clip(img_norm, 0.0, 1.0))

    mask = _largest_breast_mask(image_uint8_for_mask)
    image[~mask] = 0.0

    return image[None, :, :].astype(np.float32, copy=False)
