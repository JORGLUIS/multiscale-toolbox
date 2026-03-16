from __future__ import annotations

import numpy as np

from .pyramids import reconstruct_laplacian
from .thresholds import hard_threshold, soft_threshold


def reconstruct_with_multipliers(lap, residual, multipliers):
    mod_lap = [d * a for d, a in zip(lap, multipliers)]
    rec = reconstruct_laplacian(mod_lap, residual)
    return rec, mod_lap


def threshold_laplacian_global(lap, quantile=0.90, mode="hard"):
    all_abs = np.concatenate([np.abs(d).ravel() for d in lap])
    threshold = float(np.quantile(all_abs, quantile))
    fn = hard_threshold if mode == "hard" else soft_threshold
    out = [fn(d, threshold) for d in lap]
    return out, threshold


def threshold_laplacian_per_level(lap, quantile=0.90, mode="hard"):
    fn = hard_threshold if mode == "hard" else soft_threshold
    out = []
    thresholds = []
    for d in lap:
        threshold = float(np.quantile(np.abs(d), quantile))
        thresholds.append(threshold)
        out.append(fn(d, threshold))
    return out, thresholds
