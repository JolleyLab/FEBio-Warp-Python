"""Why contact points must be at most R_out/2 apart.

A probe point (radius optional) sits a gap g above a square lattice of contact
points with spacing a. The force it feels, relative to the dense-sampling
continuum value, is reported directly over a lattice point and over the
centre of a lattice cell (a "hole"). At a = R_out the force over a hole is 38
percent of the continuum value and the force over a point twice it; at
a = R_out/2 both are within 4 percent. Coarse sampling is how a thin strut
passes between contact points.

Run from the repository root:
    python examples/contact/point_spacing.py [--radius 0.0] [--gap-ratio 0.4]
"""

import argparse

import numpy as np
import warp as wp

from warp_febio.contact import (
    ContactPoints,
    ContactPotential,
    PotentialParams,
    node_points,
)

V = wp.vec3d


def probe_force(P, spacing, gap, over_hole, radius, device="cpu", extent=8.0):
    """Vertical force on a probe above a lattice, per unit lattice weight density."""
    r_out = P.r_out[0]
    n = int(np.ceil(extent * r_out / spacing))
    xs = (np.arange(-n, n + 1)) * spacing
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    lattice = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    shift = 0.5 * spacing if over_hole else 0.0
    probe = np.array([[shift, shift, gap + radius]])
    nodes = np.concatenate([lattice, probe])
    pts = ContactPoints.concatenate(
        [
            node_points(np.arange(len(lattice)), body=0, weight=spacing * spacing),  # weight = area per point
            node_points([len(lattice)], body=1, weight=1.0, radius=radius),
        ]
    )
    cp = ContactPotential(
        pts,
        P,
        np.zeros((2, 2), dtype=int),
        nodes,
        len(nodes),
        device=device,
        grid_dim=128,
    )
    x = wp.array(nodes, dtype=V, device=device)
    cp.build_pairs(x)
    f = wp.zeros(len(nodes), dtype=V, device=device)
    cp.energy_force(x, f)
    return f.numpy()[-1, 2]


def continuum_force(P, gap):
    """Dense-sampling limit of the force on a point above a plane of unit weight
    density: 2 pi g phi(g)."""
    return 2.0 * np.pi * gap * P.phi(gap)[()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.0, help="probe radius (capsule offset)")
    ap.add_argument(
        "--gap-ratio",
        type=float,
        default=0.4,
        help="gap / R_out at which the probe sits",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    P = PotentialParams(kc=1.0, p=4.0, r_in=0.4, r_out=1.0)
    gap = args.gap_ratio * P.r_out[0]
    dense = continuum_force(P, gap)
    print(f"gap = {gap:.3f} R_out, probe radius = {args.radius:.3f}; continuum force = {dense:.6e}")
    print(f"{'spacing/R_out':>14} {'over point':>12} {'over hole':>12}")
    for ratio in (0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.75, 0.85, 1.0, 1.25, 1.5):
        a = ratio * P.r_out[0]
        f_pt = probe_force(P, a, gap, False, args.radius, args.device) / dense
        f_hole = probe_force(P, a, gap, True, args.radius, args.device) / dense
        print(f"{ratio:14.2f} {f_pt:12.3f} {f_hole:12.3f}")


if __name__ == "__main__":
    main()
