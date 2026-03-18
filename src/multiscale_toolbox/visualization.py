from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def show_pyramid(levels, title, cmap="gray", symmetric=False, max_cols=6):
    n = len(levels)
    cols = min(n, max_cols)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i < n:
            level = levels[i]
            if symmetric:
                vmax = float(np.max(np.abs(level)))
                vmax = max(vmax, 1e-6)
                ax.imshow(level, cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                ax.imshow(level, cmap=cmap)
            ax.set_title(f"L{i} - {level.shape[0]}x{level.shape[1]}")
            ax.axis("off")
        else:
            ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_volume_slices(volume, title, indices=None, cmap="gray", symmetric=False):
    if volume.ndim != 3:
        raise ValueError("show_volume_slices expects a 3D volume")

    if indices is None:
        indices = tuple(size // 2 for size in volume.shape)
    z, y, x = indices

    slices = [
        volume[z, :, :],
        volume[:, y, :],
        volume[:, :, x],
    ]
    labels = [
        f"Axial z={z}",
        f"Coronal y={y}",
        f"Sagittal x={x}",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, img, label in zip(axes, slices, labels):
        if symmetric:
            vmax = max(float(np.max(np.abs(img))), 1e-6)
            ax.imshow(img.T, cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax)
        else:
            ax.imshow(img.T, cmap=cmap, origin="lower")
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
