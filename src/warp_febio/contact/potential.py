"""Volume-potential contact (Kamensky et al., CMAME 2018) over a pair list.

One energy for every execution tier:

    E_c = sum over kept pairs (i < j)  w_i w_j phi(r_ij),    r_ij = |x_j - x_i| - rho_i - rho_j

with the force-separation law FEBio implements in FEContactPotential:

    -phi'(r) = k_c / r^p - c2          r <  R_in
             = c1 (r - R_out)^2        R_in <= r < R_out
             = 0                       r >= R_out

c1 and c2 follow from continuity of the force and its slope at R_in, so the
force is C1 and the energy C2. The optional radius offsets rho turn a point on
a wire centerline into a capsule; with rho = 0 the law is FEBio's.

Inside the kernels distances are nondimensional, s = r / R_out, so the r^-6
terms of the Hessian stay in range in FP32 and one set of coefficients serves
models in mm and in m. Parameters are tables indexed by pair class, so
device-to-tissue and device-to-device pairs may differ.

Pair multiplicity. The sum here runs over unordered pairs (i < j). FEBio's
FEContactPotential activates element pairs from the primary surface and adds
each pair's force to both elements; when one surface is both primary and
secondary (leaflet self-contact in the CardioFEM and Aslan decks) the pair
(A, B) is also activated from B's side, so every self-contact pair enters
twice. ``PotentialParams(scale=2)`` on that pair class reproduces FEBio;
the default is 1. This is flagged for review in docs/design/contact.md.

Weights. By default the integration weights are the reference ones carried by
``ContactPoints`` and the force is exactly minus the energy gradient. With
``weight_mode="current"`` the kernels use current Jacobian-weighted weights
as FEBio does; FEBio does not differentiate the weights, and neither does this
module in that mode, so the force is then not the exact gradient of the energy.

Pair Hessian (Kamensky eq. 26), with d = x_j - x_i, n = d/|d|, r = |d| - rho:

    H = C [ phi''(r) n n^T + (phi'(r) / |d|) (I - n n^T) ]

Eigenvalues: C phi'' along n (positive under repulsion) and C phi'/|d| across
it (negative). ``project=1`` drops the negative tangential part, which gives a
positive semidefinite tangent with the same equilibrium (the fallback when the
full tangent fails to give a descent direction).
"""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp
import warp.sparse as wps

from .points import ContactPoints
from .search import PairSearch, interpolate_points, scatter_to_nodes

f64 = wp.float64
vec3d = wp.vec3d
mat33d = wp.mat33d


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
class PotentialParams:
    """Parameter table of the contact potential, one row per pair class.

    ``kc``, ``p``, ``r_in``, ``r_out`` are arrays or scalars (broadcast).
    ``r_self`` is the reference-distance self-contact exclusion radius (FEBio
    ``R0_min``); ``r_min`` the optional hard-core radius used only by the
    line-search guard, never by the energy. ``r_self`` defaults to 0, as in
    FEBio; set it when ``exclusion`` alone is not enough. ``scale`` is the pair multiplicity
    per class (2 reproduces FEBio's same-surface self-contact, see module notes).
    """

    def __init__(self, kc=1.0, p=4.0, r_in=0.2, r_out=0.5, r_self=None, r_min=0.0, scale=1.0):
        kc, p, r_in, r_out, scale = np.broadcast_arrays(
            np.atleast_1d(np.asarray(kc, dtype=np.float64)),
            np.atleast_1d(np.asarray(p, dtype=np.float64)),
            np.atleast_1d(np.asarray(r_in, dtype=np.float64)),
            np.atleast_1d(np.asarray(r_out, dtype=np.float64)),
            np.atleast_1d(np.asarray(scale, dtype=np.float64)),
        )
        if np.any(p < 4.0):
            warnings.warn(
                "Kamensky 2018: p >= 4 is needed for non-penetration of volumes",
                stacklevel=2,
            )
        if np.any(r_in <= 0.0) or np.any(r_in >= r_out):
            raise ValueError("require 0 < r_in < r_out")
        self.kc, self.p, self.r_in, self.r_out, self.scale = (
            kc.copy(),
            p.copy(),
            r_in.copy(),
            r_out.copy(),
            scale.copy(),
        )
        self.n_classes = int(kc.shape[0])
        self.r_self = float(r_self) if r_self is not None else 0.0
        self.r_min = float(r_min)
        self.s_in = self.r_in / self.r_out
        # pair multiplicity (FEBio same-surface self-contact: 2) folded into the kernel stiffness
        self.kcs = self.scale * self.kc / self.r_out**self.p
        s_in = self.s_in
        self.c1 = p / (2.0 * (1.0 - s_in) * s_in ** (p + 1.0))  # Kamensky eq. (15)
        self.c2 = s_in ** (-p) - self.c1 * (s_in - 1.0) ** 2  # Kamensky eq. (16)

    def device_tables(self, device, requires_grad: bool = False):
        """Per-class parameter arrays. With ``requires_grad`` the kernel stiffness
        table (index 0, kcs) receives gradients on a wp.Tape, which is what a
        parameter adjoint dR/dkc needs; the other tables stay fixed."""

        def mk(a, g=False):
            a = np.asarray(a, dtype=np.float64)
            return wp.array(a, dtype=f64, device=device, requires_grad=g)

        return [
            mk(self.kcs, requires_grad),
            mk(self.p),
            mk(self.s_in),
            mk(self.c1),
            mk(self.c2),
            mk(self.r_out),
        ]

    def phi(self, r, cls=0):
        """Host evaluation of phi(r) (energy per unit weight product), for checks."""
        s = np.asarray(r, dtype=np.float64) / self.r_out[cls]
        p, s_in, c1, c2 = self.p[cls], self.s_in[cls], self.c1[cls], self.c2[cls]
        blend = c1 * (1.0 - s_in) ** 3 / 3.0
        G = np.where(
            s >= 1.0,
            0.0,
            np.where(
                s >= s_in,
                c1 * (1.0 - s) ** 3 / 3.0,
                blend + (np.power(np.maximum(s, 1e-300), 1.0 - p) - s_in ** (1.0 - p)) / (p - 1.0) - c2 * (s_in - s),
            ),
        )
        return self.r_out[cls] * self.kcs[cls] * G

    def force(self, r, cls=0):
        """Host evaluation of -phi'(r), for checks."""
        s = np.asarray(r, dtype=np.float64) / self.r_out[cls]
        p, s_in, c1, c2 = self.p[cls], self.s_in[cls], self.c1[cls], self.c2[cls]
        g = np.where(
            s >= 1.0,
            0.0,
            np.where(s >= s_in, c1 * (s - 1.0) ** 2, np.power(np.maximum(s, 1e-300), -p) - c2),
        )
        return self.kcs[cls] * g


def from_febio_block(block: dict, scale: float = 1.0):
    """Translate a FEBio ``contact type="contact potential"`` block into module options.

    ``block`` holds FEBio's parameter names as keys (strings or numbers, as
    parsed from a .feb file): ``kc`` (required; FEBio's default of 0 disables
    the interface), ``p``, ``R_in``, ``R_out``, ``R0_min``, ``w_tol``,
    ``integration_rule`` (``default`` or ``higher-order``),
    ``exclude_neighbors`` (0 or 1). Missing entries take FEBio's defaults.

    Returns ``(params, contact_kwargs, point_rule)``: ``contact_kwargs`` are
    keyword arguments of ``ContactPotential`` and ``ContactPotentialTerm``
    (``w_tol``, ``exclusion``, ``weight_mode="current"``); ``point_rule`` is
    ``{"quad_n_per_edge": 2 or 4, "tri_order": "3" or "7"}`` for
    ``quad_facet_points(n_per_edge=...)`` and ``tri_facet_points(order=...)``.
    Pass ``scale=2`` when the block's primary and secondary surfaces are the
    same surface (module notes on pair multiplicity). search.py lists where
    ``w_tol`` and ``R0_min`` are applied differently from FEBio.
    """
    if "kc" not in block:
        raise ValueError("contact potential block has no kc (FEBio's default 0 disables contact)")

    def get(key, default):
        return float(block.get(key, default))

    params = PotentialParams(
        kc=get("kc", 0.0),
        p=get("p", 4.0),
        r_in=get("R_in", 1.0),
        r_out=get("R_out", 2.0),
        r_self=get("R0_min", 0.0),
        scale=scale,
    )
    rule = str(block.get("integration_rule", "default")).strip().lower()
    higher = rule in ("higher-order", "1")
    contact_kwargs = {
        "w_tol": get("w_tol", 0.0),
        "exclusion": ("neighbors" if int(get("exclude_neighbors", 0)) == 1 else "element"),
        "weight_mode": "current",
    }
    point_rule = {
        "quad_n_per_edge": 4 if higher else 2,
        "tri_order": "7" if higher else "3",
    }
    return params, contact_kwargs, point_rule


# -----------------------------------------------------------------------------
# Force-separation law in nondimensional form
#   g(s)  = -phi'(r) / kcs           dg(s) = d g / d s
#   G(s)  = integral_s^1 g(s') ds'   so that phi(r) = r_out * kcs * G(s)
# -----------------------------------------------------------------------------
@wp.func
def g_force(s: f64, p: f64, s_in: f64, c1: f64, c2: f64) -> f64:
    if s < s_in:
        return wp.pow(s, -p) - c2
    if s < f64(1.0):
        return c1 * (s - f64(1.0)) * (s - f64(1.0))
    return f64(0.0)


@wp.func
def dg_force(s: f64, p: f64, s_in: f64, c1: f64, c2: f64) -> f64:
    if s < s_in:
        return -p * wp.pow(s, -p - f64(1.0))
    if s < f64(1.0):
        return f64(2.0) * c1 * (s - f64(1.0))
    return f64(0.0)


@wp.func
def G_energy(s: f64, p: f64, s_in: f64, c1: f64, c2: f64) -> f64:
    if s >= f64(1.0):
        return f64(0.0)
    blend = c1 * (f64(1.0) - s_in) * (f64(1.0) - s_in) * (f64(1.0) - s_in) / f64(3.0)
    if s >= s_in:
        return c1 * (f64(1.0) - s) * (f64(1.0) - s) * (f64(1.0) - s) / f64(3.0)
    core = (wp.pow(s, f64(1.0) - p) - wp.pow(s_in, f64(1.0) - p)) / (p - f64(1.0)) - c2 * (s_in - s)
    return blend + core


# -----------------------------------------------------------------------------
# Pair kernels
# -----------------------------------------------------------------------------
@wp.kernel
def pair_energy_force(
    pairs: wp.array(dtype=wp.vec2i),
    n_pairs: wp.array(dtype=int),
    pts: wp.array(dtype=vec3d),
    weight: wp.array(dtype=f64),
    radius: wp.array(dtype=f64),
    body: wp.array(dtype=int),
    pair_class: wp.array2d(dtype=int),
    kcs: wp.array(dtype=f64),
    p: wp.array(dtype=f64),
    s_in: wp.array(dtype=f64),
    c1: wp.array(dtype=f64),
    c2: wp.array(dtype=f64),
    r_out: wp.array(dtype=f64),
    energy: wp.array(dtype=f64),
    f_pts: wp.array(dtype=vec3d),
):
    k = wp.tid()
    if k >= n_pairs[0]:
        return
    i = pairs[k][0]
    j = pairs[k][1]
    c = pair_class[body[i], body[j]]
    d = pts[j] - pts[i]
    dist = wp.length(d)
    r = dist - radius[i] - radius[j]
    s = r / r_out[c]
    # out of range, or overlapping capsules, where the law is undefined
    if s >= f64(1.0) or r <= f64(0.0):
        return
    C = weight[i] * weight[j]
    wp.atomic_add(energy, 0, C * r_out[c] * kcs[c] * G_energy(s, p[c], s_in[c], c1[c], c2[c]))
    fmag = C * kcs[c] * g_force(s, p[c], s_in[c], c1[c], c2[c])  # repulsive
    n = d / dist
    wp.atomic_add(f_pts, j, fmag * n)
    wp.atomic_add(f_pts, i, -fmag * n)


@wp.kernel
def pair_hessian_blocks(
    pairs: wp.array(dtype=wp.vec2i),
    n_pairs: wp.array(dtype=int),
    pts: wp.array(dtype=vec3d),
    weight: wp.array(dtype=f64),
    radius: wp.array(dtype=f64),
    body: wp.array(dtype=int),
    pair_class: wp.array2d(dtype=int),
    kcs: wp.array(dtype=f64),
    p: wp.array(dtype=f64),
    s_in: wp.array(dtype=f64),
    c1: wp.array(dtype=f64),
    c2: wp.array(dtype=f64),
    r_out: wp.array(dtype=f64),
    project: int,
    H: wp.array(dtype=mat33d),
):
    k = wp.tid()
    if k >= n_pairs[0]:
        return
    i = pairs[k][0]
    j = pairs[k][1]
    c = pair_class[body[i], body[j]]
    d = pts[j] - pts[i]
    dist = wp.length(d)
    r = dist - radius[i] - radius[j]
    s = r / r_out[c]
    if s >= f64(1.0) or r <= f64(0.0):
        H[k] = mat33d()
        return
    C = weight[i] * weight[j]
    n = d / dist
    nn = wp.outer(n, n)
    phi1 = -kcs[c] * g_force(s, p[c], s_in[c], c1[c], c2[c])  # phi'(r)
    phi2 = -kcs[c] * dg_force(s, p[c], s_in[c], c1[c], c2[c]) / r_out[c]  # phi''(r)
    Hk = C * phi2 * nn
    if project == 0:
        Hk = Hk + C * (phi1 / dist) * (wp.identity(n=3, dtype=f64) - nn)
    H[k] = Hk


@wp.kernel
def hessian_matvec_points(
    pairs: wp.array(dtype=wp.vec2i),
    n_pairs: wp.array(dtype=int),
    H: wp.array(dtype=mat33d),
    v_pts: wp.array(dtype=vec3d),
    out_pts: wp.array(dtype=vec3d),
):
    k = wp.tid()
    if k >= n_pairs[0]:
        return
    i = pairs[k][0]
    j = pairs[k][1]
    Hdv = H[k] * (v_pts[j] - v_pts[i])
    wp.atomic_add(out_pts, j, Hdv)
    wp.atomic_add(out_pts, i, -Hdv)


@wp.kernel
def pair_triplets(
    pairs: wp.array(dtype=wp.vec2i),
    n_pairs: wp.array(dtype=int),
    H: wp.array(dtype=mat33d),
    pt_nodes: wp.array2d(dtype=int),
    pt_w: wp.array2d(dtype=f64),
    rows: wp.array(dtype=int),
    cols: wp.array(dtype=int),
    vals: wp.array(dtype=mat33d),
    count: wp.array(dtype=int),
):
    """Nodal 3x3 blocks of the contact tangent as COO triplets, 4 K^2 per pair:
    (+) ii, (+) jj, (-) ij, (-) ji, each chained through the point weights.
    Unused node slots write a zero block at (0, 0), pruned by the BSR build."""
    k = wp.tid()
    K = pt_nodes.shape[1]
    per_pair = 4 * K * K
    if k == 0:
        count[0] = wp.min(n_pairs[0], pairs.shape[0]) * per_pair
    if k >= n_pairs[0]:
        return
    i = pairs[k][0]
    j = pairs[k][1]
    Hk = H[k]
    base = k * per_pair
    for q in range(4):
        pa = i
        pb = i
        sign = f64(1.0)
        if q == 1:
            pa = j
            pb = j
        elif q == 2:
            pb = j
            sign = f64(-1.0)
        elif q == 3:
            pa = j
            pb = i
            sign = f64(-1.0)
        for a in range(K):
            for b in range(K):
                idx = base + (q * K + a) * K + b
                na = pt_nodes[pa, a]
                nb = pt_nodes[pb, b]
                if na >= 0 and nb >= 0:
                    rows[idx] = na
                    cols[idx] = nb
                    vals[idx] = (sign * pt_w[pa, a] * pt_w[pb, b]) * Hk
                else:
                    rows[idx] = 0
                    cols[idx] = 0
                    vals[idx] = mat33d()


@wp.kernel
def min_pair_distance(
    pairs: wp.array(dtype=wp.vec2i),
    n_pairs: wp.array(dtype=int),
    pts: wp.array(dtype=vec3d),
    radius: wp.array(dtype=f64),
    dmin: wp.array(dtype=f64),
):
    k = wp.tid()
    if k >= n_pairs[0]:
        return
    i = pairs[k][0]
    j = pairs[k][1]
    wp.atomic_min(dmin, 0, wp.length(pts[j] - pts[i]) - radius[i] - radius[j])


# -----------------------------------------------------------------------------
# Host-side term
# -----------------------------------------------------------------------------
class ContactPotential:
    """Contact potential over a point set: energy, nodal forces, tangent action.

    Everything after ``search.build`` runs over the fixed-capacity pair buffer
    with an early exit on the device pair count, so the Newton body can be
    captured in a CUDA graph; ``search.check_overflow`` is the only host read.

    Sign convention: ``force`` returns f = -dE_c/dx (repulsive, pushes bodies
    apart). The residual convention of ``contracts.PhysicsTerm`` (internal plus
    contact minus external) wants +dE_c/dx; ``enforcement.py`` applies the sign.
    """

    def __init__(
        self,
        points: ContactPoints,
        params: PotentialParams,
        pair_class: np.ndarray,
        ref_node_x: np.ndarray,
        n_nodes: int,
        exclusion: str = "neighbors",
        w_tol: float = 0.0,
        weight_mode: str = "reference",
        device: str = "cpu",
        **search_kwargs,
    ):
        if weight_mode not in ("reference", "current"):
            raise ValueError("weight_mode must be 'reference' or 'current'")
        self.points = points
        self.params = params
        self.device = device
        self.n_nodes = int(n_nodes)
        self.weight_mode = weight_mode
        self.search = PairSearch(
            points,
            pair_class,
            params.r_out,
            ref_node_x,
            params.r_self,
            exclusion=exclusion,
            w_tol=w_tol,
            device=device,
            **search_kwargs,
        )
        self.weight = wp.array(points.weight, dtype=f64, device=device)
        # kcs carries a gradient buffer so parameter adjoints can flow to it; f_pts
        # carries one so that a vector-Jacobian product can be seeded at the forces
        self.tables = params.device_tables(device, requires_grad=True)
        self.f_pts = wp.zeros(points.count, dtype=vec3d, device=device, requires_grad=True)
        self.energy = wp.zeros(1, dtype=f64, device=device, requires_grad=True)
        self.dmin = wp.zeros(1, dtype=f64, device=device)
        self._trip_rows = self._trip_cols = self._trip_vals = None
        self._trip_count = wp.zeros(1, dtype=int, device=device)
        self.H = wp.zeros(self.search.capacity, dtype=mat33d, device=device)
        self._v_pts = wp.zeros(points.count, dtype=vec3d, device=device)
        self._o_pts = wp.zeros(points.count, dtype=vec3d, device=device)

    # --- selection ----------------------------------------------------------
    def build_pairs(self, node_x: wp.array, sync: bool = True) -> int:
        n = self.search.build(node_x, sync=sync)
        if self.H.shape[0] != self.search.capacity:
            self.H = wp.zeros(self.search.capacity, dtype=mat33d, device=self.device)
        return n

    def _pair_args(self):
        s = self.search
        w = s.cur_weight if self.weight_mode == "current" else self.weight
        return [
            s.pairs,
            s.n_pairs_dev,
            s.pts,
            w,
            s.radius,
            s.body,
            s.pair_class,
            *self.tables,
        ]

    # --- evaluation over the current pair list ------------------------------
    def energy_force(self, node_x: wp.array, f_nodes: wp.array) -> wp.array:
        """E_c (device scalar) and nodal forces f = -dE_c/dx accumulated into f_nodes.

        Record on a wp.Tape by wrapping the call; ``search.pts`` and ``energy``
        carry gradients.
        """
        s = self.search
        self.energy.zero_()
        self.f_pts.zero_()
        if self.weight_mode == "current":
            s.update_points(node_x)
        else:
            wp.launch(
                interpolate_points,
                dim=s.n_pts,
                inputs=[node_x, s.pt_nodes, s.pt_w, s.pts],
                device=self.device,
            )
        wp.launch(
            pair_energy_force,
            dim=s.capacity,
            inputs=self._pair_args() + [self.energy, self.f_pts],
            device=self.device,
        )
        wp.launch(
            scatter_to_nodes,
            dim=s.n_pts,
            inputs=[self.f_pts, s.pt_nodes, s.pt_w, f_nodes],
            device=self.device,
        )
        return self.energy

    def kc_gradient_from_tables(self) -> np.ndarray:
        """Chain the tape gradient on the kcs table back to kc per class
        (kcs = scale kc / R_out^p). Host array of length n_classes."""
        g_kcs = self.tables[0].grad.numpy()
        P = self.params
        return g_kcs * P.scale / P.r_out**P.p

    def energy_value(self) -> float:
        return float(self.energy.numpy()[0])

    def hessian_blocks(self, project: bool = False) -> wp.array:
        wp.launch(
            pair_hessian_blocks,
            dim=self.search.capacity,
            inputs=self._pair_args() + [1 if project else 0, self.H],
            device=self.device,
        )
        return self.H

    def hessian_matvec(
        self,
        v_nodes: wp.array,
        out_nodes: wp.array,
        project: bool = False,
        rebuild_blocks: bool = True,
    ):
        """out_nodes += K_c v_nodes, matrix-free, chained through the point weights."""
        s = self.search
        if rebuild_blocks:
            self.hessian_blocks(project)
        self._o_pts.zero_()
        wp.launch(
            interpolate_points,
            dim=s.n_pts,
            inputs=[v_nodes, s.pt_nodes, s.pt_w, self._v_pts],
            device=self.device,
        )
        wp.launch(
            hessian_matvec_points,
            dim=s.capacity,
            inputs=[s.pairs, s.n_pairs_dev, self.H, self._v_pts, self._o_pts],
            device=self.device,
        )
        wp.launch(
            scatter_to_nodes,
            dim=s.n_pts,
            inputs=[self._o_pts, s.pt_nodes, s.pt_w, out_nodes],
            device=self.device,
        )

    def hessian_triplets(self, project: bool = False):
        """Assembled K_c as host COO triplets (rows, cols, vals) over 3n scalar dofs.

        Host-side and numpy only, for tests and small direct solves; the GPU
        paths are the matrix-free action and ``hessian_bsr``.
        """
        s = self.search
        s.check_overflow()
        if s.n_pairs == 0:
            z = np.zeros(0)
            return z.astype(np.int64), z.astype(np.int64), z
        H = self.hessian_blocks(project).numpy()[: s.n_pairs]
        pairs = s.pairs.numpy()[: s.n_pairs]
        pn, pw, K = self.points.pt_nodes, self.points.pt_w, self.points.nodes_per_point
        rows, cols, vals = [], [], []
        for pa, pb, sign in ((0, 0, 1.0), (1, 1, 1.0), (0, 1, -1.0), (1, 0, -1.0)):
            ia, ib = pairs[:, pa], pairs[:, pb]
            for a in range(K):
                na, wa = pn[ia, a], pw[ia, a]
                for b in range(K):
                    nb, wb = pn[ib, b], pw[ib, b]
                    ok = (na >= 0) & (nb >= 0)
                    if not ok.any():
                        continue
                    blk = (sign * wa[ok] * wb[ok])[:, None, None] * H[ok]
                    r0, c0 = 3 * na[ok], 3 * nb[ok]
                    for u in range(3):
                        for v in range(3):
                            rows.append(r0 + u)
                            cols.append(c0 + v)
                            vals.append(blk[:, u, v])
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

    def hessian_dense(self, project: bool = False) -> np.ndarray:
        """Dense K_c (3n x 3n) on the host, for tests."""
        n3 = 3 * self.n_nodes
        K = np.zeros((n3, n3))
        rows, cols, vals = self.hessian_triplets(project)
        np.add.at(K, (rows, cols), vals)
        return K

    def hessian_bsr(self, project: bool = False, dest=None):
        """Assembled contact tangent as a warp.sparse BSR matrix of 3x3 blocks,
        built on the device from the pair list. Pass ``dest`` (from a previous
        call) to reuse its storage. Not yet exercised under CUDA graph capture."""
        s = self.search
        self.hessian_blocks(project)
        K = self.points.nodes_per_point
        n_trip = s.capacity * 4 * K * K
        if self._trip_rows is None or self._trip_rows.shape[0] != n_trip:
            self._trip_rows = wp.zeros(n_trip, dtype=int, device=self.device)
            self._trip_cols = wp.zeros(n_trip, dtype=int, device=self.device)
            self._trip_vals = wp.zeros(n_trip, dtype=mat33d, device=self.device)
        wp.launch(
            pair_triplets,
            dim=s.capacity,
            inputs=[s.pairs, s.n_pairs_dev, self.H, s.pt_nodes, s.pt_w],
            outputs=[
                self._trip_rows,
                self._trip_cols,
                self._trip_vals,
                self._trip_count,
            ],
            device=self.device,
        )
        if dest is None:
            dest = wps.bsr_zeros(self.n_nodes, self.n_nodes, block_type=mat33d, device=self.device)
        wps.bsr_set_from_triplets(
            dest,
            self._trip_rows,
            self._trip_cols,
            self._trip_vals,
            count=self._trip_count,
        )
        return dest

    def min_distance(self, node_x: wp.array) -> float:
        """Smallest effective pair distance over the current pair list (r_min guard)."""
        s = self.search
        wp.launch(
            interpolate_points,
            dim=s.n_pts,
            inputs=[node_x, s.pt_nodes, s.pt_w, s.pts],
            device=self.device,
        )
        self.dmin.fill_(np.inf)
        wp.launch(
            min_pair_distance,
            dim=s.capacity,
            inputs=[s.pairs, s.n_pairs_dev, s.pts, s.radius, self.dmin],
            device=self.device,
        )
        return float(self.dmin.numpy()[0])

    # --- metrics ------------------------------------------------------------
    def point_forces(self) -> np.ndarray:
        return self.f_pts.numpy()

    def contact_area(self, body: int | None = None) -> float:
        """Wu 2022 rule: area of the points with nonzero contact force."""
        f = np.linalg.norm(self.f_pts.numpy(), axis=1)
        mask = f > 0.0
        if body is not None:
            mask &= self.points.body == body
        return float(np.sum(self.points.area[mask]))

    def contact_fraction(self, body: int, threshold: float, other_body: int, node_x: np.ndarray) -> float:
        """Zelonis rule: fraction of a body's weight within ``threshold`` of another body
        (host evaluation; brute force over the other body's points)."""
        x = self.points.positions(np.asarray(node_x))
        a = self.points.body == body
        b = self.points.body == other_body
        xa, xb = x[a], x[b]
        close = np.zeros(xa.shape[0], dtype=bool)
        chunk = 2048
        for k in range(0, xa.shape[0], chunk):
            d = np.linalg.norm(xa[k : k + chunk, None, :] - xb[None, :, :], axis=2)
            d = d - self.points.radius[a][k : k + chunk, None] - self.points.radius[b][None, :]
            close[k : k + chunk] = d.min(axis=1) < threshold
        w = self.points.weight[a]
        return float(w[close].sum() / w.sum())
