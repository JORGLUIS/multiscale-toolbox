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


def reconstruct_selected_layers(lap, residual, active_levels=None, include_residual=True):
    if active_levels is None:
        active_levels = []
    active_set = set(active_levels)
    mod_lap = [d if i in active_set else np.zeros_like(d) for i, d in enumerate(lap)]
    base = residual if include_residual else np.zeros_like(residual)
    return reconstruct_laplacian(mod_lap, base)


def reconstruct_each_detail_contribution(lap, residual):
    contributions = []
    if not lap:
        raise ValueError("lap must contain at least one level")
    zero_residual = np.zeros_like(residual)
    for i in range(len(lap)):
        mod_lap = [np.zeros_like(d) for d in lap]
        mod_lap[i] = lap[i]
        contributions.append(reconstruct_laplacian(mod_lap, zero_residual))
    return contributions
