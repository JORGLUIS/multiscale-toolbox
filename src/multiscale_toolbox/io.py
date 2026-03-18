from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def load_grayscale_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def load_nifti_volume(path: Path, dtype=np.float32):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=dtype)
    return data.astype(dtype), img.affine, img.header
