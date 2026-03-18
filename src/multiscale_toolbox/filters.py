from __future__ import annotations

import numpy as np
from scipy import ndimage


def mean_filter(size: int):
    kernel = np.ones((size, size), dtype=np.float32) / float(size * size)

    def apply(x: np.ndarray) -> np.ndarray:
        return ndimage.convolve(x, kernel, mode="reflect")

    return apply


def gaussian_filter_fn(sigma: float):
    def apply(x: np.ndarray) -> np.ndarray:
        return ndimage.gaussian_filter(x, sigma=sigma, mode="reflect")

    return apply


def ideal_lowpass_filter(cutoff_ratio: float):
    def apply(x: np.ndarray) -> np.ndarray:
        rows, cols = x.shape
        fy = np.fft.fftfreq(rows)
        fx = np.fft.fftfreq(cols)
        gx, gy = np.meshgrid(fx, fy)
        radius = np.sqrt(gx**2 + gy**2)
        mask = (radius <= cutoff_ratio).astype(np.float32)
        y = np.fft.ifft2(np.fft.fft2(x) * mask)
        return np.real(y).astype(np.float32)

    return apply


def binomial_filter_1d(coeffs) -> np.ndarray:
    arr = np.asarray(coeffs, dtype=np.float32)
    return arr / np.sum(arr)


def binomial_filter_fn(coeffs):
    k = binomial_filter_1d(coeffs)
    kernel = np.outer(k, k).astype(np.float32)

    def apply(x: np.ndarray) -> np.ndarray:
        return ndimage.convolve(x, kernel, mode="reflect")

    return apply


def create_default_filters() -> dict[str, object]:
    return {
        "Promedio 3x3": mean_filter(3),
        "Promedio 5x5": mean_filter(5),
        "Gauss sigma=1": gaussian_filter_fn(1.0),
        "Gauss sigma=2": gaussian_filter_fn(2.0),
        "Ideal cutoff=0.10": ideal_lowpass_filter(0.10),
        "Ideal cutoff=0.18": ideal_lowpass_filter(0.18),
        "Binomial [1,2,1]": binomial_filter_fn([1, 2, 1]),
        "Binomial [1,4,6,4,1]": binomial_filter_fn([1, 4, 6, 4, 1]),
    }
