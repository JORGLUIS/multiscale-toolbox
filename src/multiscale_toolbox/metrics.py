from __future__ import annotations

import numpy as np
from scipy import ndimage


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mse(a, b)))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    err = mse(a, b)
    if err == 0:
        return float("inf")
    return float(20 * np.log10(data_range / np.sqrt(err)))


def _mask_values(x: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return x.ravel()
    return x[np.asarray(mask) > 0]


def masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (a - b) ** 2
    values = _mask_values(diff, mask)
    return float(np.mean(values))


def masked_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    return float(np.sqrt(masked_mse(a, b, mask=mask)))


def masked_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    va = _mask_values(a, mask).astype(np.float64)
    vb = _mask_values(b, mask).astype(np.float64)
    if va.size == 0 or vb.size == 0:
        return float("nan")
    va = va - np.mean(va)
    vb = vb - np.mean(vb)
    denom = np.sqrt(np.sum(va**2) * np.sum(vb**2))
    if denom <= 0:
        return 0.0
    return float(np.sum(va * vb) / denom)


def laplacian_energy(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    lap = ndimage.laplace(x.astype(np.float32), mode="reflect")
    values = _mask_values(lap**2, mask)
    return float(np.mean(values))
