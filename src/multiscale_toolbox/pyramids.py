from __future__ import annotations

import numpy as np
from scipy import ndimage


def downsample2(x: np.ndarray) -> np.ndarray:
    return x[::2, ::2]


def upsample_by_factor(x: np.ndarray, factor) -> np.ndarray:
    factor_int = int(round(float(factor)))
    if factor_int < 1 or not np.isclose(factor, factor_int):
        raise ValueError("upsample_by_factor only supports positive integer factors")
    if factor_int == 1:
        return x.astype(np.float32).copy()

    h, w = x.shape
    up = np.zeros((h * factor_int, w * factor_int), dtype=np.float32)
    up[::factor_int, ::factor_int] = x.astype(np.float32)

    ramp = np.arange(1, factor_int + 1, dtype=np.float32) / float(factor_int)
    kernel_1d = np.concatenate([ramp, ramp[-2::-1]])
    kernel_2d = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    return ndimage.convolve(up, kernel_2d, mode="reflect")


def build_gaussian_pyramid(image: np.ndarray, scaling_fn, levels: int):
    gauss = [image]
    filtered = []
    current = image
    for _ in range(levels):
        low = scaling_fn(current)
        filtered.append(low)
        if min(low.shape) < 2:
            break
        current = downsample2(low)
        gauss.append(current)
    return gauss, filtered


def build_laplacian_predictive(
    image: np.ndarray,
    scaling_fn,
    levels: int,
):
    gauss = [image]
    current = image
    for _ in range(levels):
        low = scaling_fn(current)
        if min(low.shape) < 2:
            break
        current = downsample2(low)
        gauss.append(current)

    lap = []
    for i in range(len(gauss) - 1):
        pred = upsample_by_factor(gauss[i + 1], 2)
        pred = pred[: gauss[i].shape[0], : gauss[i].shape[1]]
        lap.append(gauss[i] - pred)

    residual = gauss[-1]
    return lap, residual, gauss


def reconstruct_laplacian(
    lap,
    residual,
):
    current = residual
    for d in reversed(lap):
        pred = upsample_by_factor(current, 2)
        pred = pred[: d.shape[0], : d.shape[1]]
        current = pred + d
    return current
