from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0
