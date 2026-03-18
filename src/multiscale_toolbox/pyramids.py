from __future__ import annotations

import numpy as np
from scipy import ndimage


def downsample2(x: np.ndarray) -> np.ndarray:
    return x[tuple(slice(None, None, 2) for _ in range(x.ndim))]


def upsample_by_factor(x: np.ndarray, factor) -> np.ndarray:
    if np.isscalar(factor):
        factors = (int(round(float(factor))),) * x.ndim
    else:
        factors = tuple(int(round(float(f))) for f in factor)
        if len(factors) != x.ndim:
            raise ValueError("factor sequence must match x.ndim")

    if any(f < 1 for f in factors):
        raise ValueError("upsample_by_factor only supports positive integer factors")
    if any(not np.isclose(f, fi) for f, fi in zip(np.atleast_1d(factor), factors)) and not np.isscalar(factor):
        raise ValueError("factor sequence must contain integers")
    if np.isscalar(factor) and not np.isclose(float(factor), factors[0]):
        raise ValueError("factor must be an integer")

    if all(f == 1 for f in factors):
        return x.astype(np.float32).copy()

    out_shape = tuple(size * factor_int for size, factor_int in zip(x.shape, factors))
    up = np.zeros(out_shape, dtype=np.float32)
    up[tuple(slice(None, None, factor_int) for factor_int in factors)] = x.astype(np.float32)

    y = up
    for axis, factor_int in enumerate(factors):
        if factor_int == 1:
            continue
        ramp = np.arange(1, factor_int + 1, dtype=np.float32) / float(factor_int)
        kernel_1d = np.concatenate([ramp, ramp[-2::-1]])
        y = ndimage.convolve1d(y, kernel_1d, axis=axis, mode="reflect")
    return y.astype(np.float32)


def crop_to_shape(x: np.ndarray, target_shape) -> np.ndarray:
    return x[tuple(slice(0, size) for size in target_shape)]


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
        pred = crop_to_shape(pred, gauss[i].shape)
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
        pred = crop_to_shape(pred, d.shape)
        current = pred + d
    return current
