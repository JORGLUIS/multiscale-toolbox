from __future__ import annotations

import numpy as np


def hard_threshold(x: np.ndarray, t: float) -> np.ndarray:
    y = x.copy()
    y[np.abs(y) < t] = 0.0
    return y


def soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)
