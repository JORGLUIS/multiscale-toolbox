from __future__ import annotations

import numpy as np
from scipy import ndimage


def _normalize_ndim(ndim: int) -> int:
    ndim_int = int(ndim)
    if ndim_int < 1:
        raise ValueError("ndim must be at least 1")
    return ndim_int


def mean_filter(size: int, ndim: int = 2):
    ndim = _normalize_ndim(ndim)
    kernel_shape = (size,) * ndim
    kernel = np.ones(kernel_shape, dtype=np.float32) / float(size**ndim)

    def apply(x: np.ndarray) -> np.ndarray:
        return ndimage.convolve(x, kernel, mode="reflect")

    return apply


def gaussian_filter_fn(sigma: float):
    def apply(x: np.ndarray) -> np.ndarray:
        return ndimage.gaussian_filter(x, sigma=sigma, mode="reflect")

    return apply


def ideal_lowpass_filter(cutoff_ratio: float):
    def apply(x: np.ndarray) -> np.ndarray:
        freqs = np.meshgrid(
            *[np.fft.fftfreq(size) for size in x.shape],
            indexing="ij",
        )
        radius = np.zeros(x.shape, dtype=np.float32)
        for grid in freqs:
            radius += grid.astype(np.float32) ** 2
        radius = np.sqrt(radius)
        mask = (radius <= cutoff_ratio).astype(np.float32)
        y = np.fft.ifftn(np.fft.fftn(x) * mask)
        return np.real(y).astype(np.float32)

    return apply


def binomial_filter_1d(coeffs) -> np.ndarray:
    arr = np.asarray(coeffs, dtype=np.float32)
    return arr / np.sum(arr)


def binomial_filter_fn(coeffs, ndim: int = 2):
    ndim = _normalize_ndim(ndim)
    k = binomial_filter_1d(coeffs)

    def apply(x: np.ndarray) -> np.ndarray:
        y = x.astype(np.float32)
        for axis in range(ndim):
            y = ndimage.convolve1d(y, k, axis=axis, mode="reflect")
        return y.astype(np.float32)

    return apply


def spherical_mean_filter_fn(radius: int, ndim: int = 3):
    ndim = _normalize_ndim(ndim)
    radius_int = int(radius)
    if radius_int < 1:
        raise ValueError("radius must be at least 1")

    coords = np.ogrid[tuple(slice(-radius_int, radius_int + 1) for _ in range(ndim))]
    distance_sq = np.zeros(tuple(2 * radius_int + 1 for _ in range(ndim)), dtype=np.float32)
    for axis_coords in coords:
        distance_sq += axis_coords.astype(np.float32) ** 2
    kernel = (distance_sq <= float(radius_int**2)).astype(np.float32)
    kernel /= np.sum(kernel)

    def apply(x: np.ndarray) -> np.ndarray:
        return ndimage.convolve(x, kernel, mode="reflect")

    return apply


def create_default_filters(ndim: int = 2) -> dict[str, object]:
    ndim = _normalize_ndim(ndim)
    size3 = "x".join(["3"] * ndim)
    size5 = "x".join(["5"] * ndim)

    filters = {
        f"Promedio {size3}": mean_filter(3, ndim=ndim),
        f"Promedio {size5}": mean_filter(5, ndim=ndim),
        "Gauss sigma=1": gaussian_filter_fn(1.0),
        "Gauss sigma=2": gaussian_filter_fn(2.0),
        "Ideal cutoff=0.10": ideal_lowpass_filter(0.10),
        "Ideal cutoff=0.18": ideal_lowpass_filter(0.18),
        "Binomial [1,2,1]": binomial_filter_fn([1, 2, 1], ndim=ndim),
        "Binomial [1,4,6,4,1]": binomial_filter_fn([1, 4, 6, 4, 1], ndim=ndim),
    }
    if ndim >= 3:
        filters["Esferico r=1"] = spherical_mean_filter_fn(1, ndim=ndim)
        filters["Esferico r=2"] = spherical_mean_filter_fn(2, ndim=ndim)
    return filters
