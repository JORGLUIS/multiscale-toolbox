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
