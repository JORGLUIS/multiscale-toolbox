from __future__ import annotations

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mse(a, b)))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    err = mse(a, b)
    if err == 0:
        return float("inf")
    return float(20 * np.log10(data_range / np.sqrt(err)))
