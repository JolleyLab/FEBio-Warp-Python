"""Matplotlib viewer: true-scale reference/deformed overlays with a time slider."""

import argparse
from pathlib import Path
import numpy as np


def render(path, output=None, label="Synthetic membrane"):
    import matplotlib

    if output is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    data = np.load(path)
    rest = data["rest"] * 1000
    faces = data["faces"]
    frames = data["snapshots"] * 1000
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.16)
    reference = Poly3DCollection(
        rest[faces],
        facecolors=(0, 0, 0, 0),
        edgecolors=(0.3, 0.3, 0.3, 0.2),
        linewidths=0.2,
    )
    ax.add_collection3d(reference)
    displacement = np.linalg.norm(frames - rest, axis=2)
    norm = plt.Normalize(0, max(float(displacement.max()), 1.0e-9))
    cmap = plt.get_cmap("viridis")
    surface = Poly3DCollection(
        frames[-1][faces], linewidths=0.1, edgecolors=(0, 0, 0, 0.1)
    )
    ax.add_collection3d(surface)
    fixed = data["boundary"]
    ax.scatter(*rest[fixed].T, c="red", s=9, label="Fixed perimeter")
    low = np.minimum(rest.min(0), frames.min(axis=(0, 1)))
    high = np.maximum(rest.max(0), frames.max(axis=(0, 1)))
    center = (low + high) / 2
    radius = (high - low).max() * 0.55
    ax.set(
        xlim=(center[0] - radius, center[0] + radius),
        ylim=(center[1] - radius, center[1] + radius),
        zlim=(center[2] - radius, center[2] + radius),
        xlabel="X [mm]",
        ylabel="Y [mm]",
        zlabel="Z [mm]",
    )
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=28, azim=35)
    ax.legend()
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.55,
        label="Displacement [mm]",
    )

    def update(index):
        i = int(index)
        surface.set_verts(frames[i][faces])
        surface.set_facecolor(cmap(norm(displacement[i][faces].mean(axis=1))))
        ax.set_title(
            f'{label} — true scale\nt = {data["times"][i]:.5f} s; gray = reference'
        )
        fig.canvas.draw_idle()

    update(len(frames) - 1)
    if output is not None:
        fig.savefig(output, dpi=160)
        plt.close(fig)
    else:
        slider = Slider(
            fig.add_axes([0.2, 0.06, 0.55, 0.03]),
            "Frame",
            0,
            len(frames) - 1,
            valinit=len(frames) - 1,
            valstep=1,
        )
        slider.on_changed(update)
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("state", type=Path)
    p.add_argument("--save", type=Path)
    p.add_argument("--label", default="Synthetic membrane")
    a = p.parse_args()
    render(a.state, a.save, label=a.label)
