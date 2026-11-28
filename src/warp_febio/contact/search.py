"""Candidate pair generation for point-based contact.

Selection, not physics: which pairs of contact points can interact in the
current configuration. The result is an explicit pair list that the energy,
force and tangent kernels loop over. Keeping selection separate from
evaluation is what makes the evaluation differentiable (the pair set is
piecewise constant in the nodal positions) and what lets the whole Newton body
run without host synchronization.

Mechanics of the search (Kamensky 2018, Section 3.1; FEBio's uniform grid,
whose cell size is at least R_out):

* ``wp.HashGrid`` in float64 on the points themselves, cell size equal to the
  search radius.
* Two passes, count then fill, with a device-side exclusive scan in between, so
  the pair buffer has fixed capacity and overflow is detected by a single
  host read at a convenient time.
* A pair (i, j) with i < j is kept when its effective distance
  ``|x_j - x_i| - radius_i - radius_j`` is below the outer radius of its pair
  class and it is not excluded.

Exclusion (self-contact):

* different bodies: never excluded;
* same body and reference distance below ``r_self``: excluded (FEBio ``R0_min``);
* same body and, depending on ``exclusion``: ``"element"`` drops points of the
  same source element (FEBio with ``exclude_neighbors`` set to 0: only an
  element against itself is skipped, neighbors are left to ``w_tol`` and
  ``R0_min``); ``"neighbors"`` drops points whose source elements share a
  node (FEBio's default, ``exclude_neighbors`` 1); ``"none"`` applies the
  reference distance only.
  Element ids are per builder call, so give each builder call its own body
  when using ``"element"``.

``w_tol``: a pair is dropped when the cosine between the pair direction and
the current facet normal is below ``w_tol``. Two deliberate differences from
FEBio's FEContactPotential, which matter for a facet-level comparison: FEBio
applies ``w_tol`` and ``R0_min`` as element-pair activation gates (one passing
Gauss-point pair activates the whole element pair, which is then summed
without rechecking) and tests only the primary point's normal; here both are
applied per point pair, and a pair is dropped only when every point of the
pair that has a normal fails the test. FEBio's edge-to-surface variant
declares ``w_tol`` but does not use it; here a wire-to-surface pair is tested
through the surface point's normal. With ``w_tol > 0`` the energy is not
continuous in the positions (a pair in range can be switched by the normal
test); with ``w_tol = 0`` it is C2.

Not yet implemented from FEBio: ``check_intersections``.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .points import ContactPoints

f64 = wp.float64
vec3d = wp.vec3d


@wp.kernel
def interpolate_points(
    node_x: wp.array(dtype=vec3d),
    pt_nodes: wp.array2d(dtype=int),
    pt_w: wp.array2d(dtype=f64),
    pts: wp.array(dtype=vec3d),
):
    i = wp.tid()
    x = vec3d()
    for a in range(pt_nodes.shape[1]):
        n = pt_nodes[i, a]
        if n >= 0:
            x = x + pt_w[i, a] * node_x[n]
    pts[i] = x


@wp.kernel
def point_frames(
    node_x: wp.array(dtype=vec3d),
    pt_nodes: wp.array2d(dtype=int),
    pt_w: wp.array2d(dtype=f64),
    pt_dw1: wp.array2d(dtype=f64),
    pt_dw2: wp.array2d(dtype=f64),
    rule_w: wp.array(dtype=f64),
    pts: wp.array(dtype=vec3d),
    normals: wp.array(dtype=vec3d),
    cur_weight: wp.array(dtype=f64),
):
    """Position, current unit normal (zero for lines and bare nodes) and current
    Jacobian-weighted integration weight of each point, as FEBio's
    FEContactPotential::UpdateSurface computes them."""
    i = wp.tid()
    x = vec3d()
    t1 = vec3d()
    t2 = vec3d()
    for a in range(pt_nodes.shape[1]):
        n = pt_nodes[i, a]
        if n >= 0:
            xn = node_x[n]
            x = x + pt_w[i, a] * xn
            t1 = t1 + pt_dw1[i, a] * xn
            t2 = t2 + pt_dw2[i, a] * xn
    pts[i] = x
    l1 = wp.length(t1)
    l2 = wp.length(t2)
    if l2 > f64(0.0):
        nrm = wp.cross(t1, t2)
        J = wp.length(nrm)
        if J > f64(0.0):
            normals[i] = nrm / J
        else:  # degenerate facet: no usable normal, zero current weight
            normals[i] = vec3d()
        cur_weight[i] = rule_w[i] * J
    elif l1 > f64(0.0):
        normals[i] = vec3d()
        cur_weight[i] = rule_w[i] * l1
    else:
        normals[i] = vec3d()
        cur_weight[i] = rule_w[i]


@wp.kernel
def scatter_to_nodes(
    f_pts: wp.array(dtype=vec3d),
    pt_nodes: wp.array2d(dtype=int),
    pt_w: wp.array2d(dtype=f64),
    f_nodes: wp.array(dtype=vec3d),
):
    """Transpose of interpolate_points: accumulate point vectors onto nodes."""
    i = wp.tid()
    for a in range(pt_nodes.shape[1]):
        n = pt_nodes[i, a]
        if n >= 0:
            wp.atomic_add(f_nodes, n, pt_w[i, a] * f_pts[i])


@wp.func
def pair_excluded(
    i: int,
    j: int,
    body: wp.array(dtype=int),
    element: wp.array(dtype=int),
    ref_pts: wp.array(dtype=vec3d),
    r_self: f64,
    pt_nodes: wp.array2d(dtype=int),
    exclusion: int,
) -> bool:
    """exclusion: 0 none, 1 same source element, 2 source elements sharing a node."""
    if body[i] != body[j]:
        return False
    if wp.length(ref_pts[j] - ref_pts[i]) < r_self:
        return True
    if exclusion == 1:
        return element[i] == element[j]
    if exclusion == 2:
        for a in range(pt_nodes.shape[1]):
            na = pt_nodes[i, a]
            if na >= 0:
                for b in range(pt_nodes.shape[1]):
                    if pt_nodes[j, b] == na:
                        return True
    return False


@wp.func
def pair_active(
    i: int,
    j: int,
    pts: wp.array(dtype=vec3d),
    normals: wp.array(dtype=vec3d),
    radius: wp.array(dtype=f64),
    body: wp.array(dtype=int),
    pair_class: wp.array2d(dtype=int),
    r_out: wp.array(dtype=f64),
    w_tol: f64,
) -> bool:
    c = pair_class[body[i], body[j]]
    if c < 0:
        return False
    d = pts[j] - pts[i]
    dist = wp.length(d)
    if dist <= f64(0.0):  # coincident points have no direction; never pair them
        return False
    if dist - radius[i] - radius[j] >= r_out[c]:
        return False
    if w_tol > f64(0.0):
        e = d / dist
        ni = normals[i]
        nj = normals[j]
        has_i = wp.length_sq(ni) > f64(0.0)
        has_j = wp.length_sq(nj) > f64(0.0)
        pass_i = has_i and (wp.abs(wp.dot(e, ni)) >= w_tol)
        pass_j = has_j and (wp.abs(wp.dot(e, nj)) >= w_tol)
        if (has_i or has_j) and not (pass_i or pass_j):
            return False
    return True


@wp.kernel
def count_pairs(
    grid: wp.uint64,
    pts: wp.array(dtype=vec3d),
    normals: wp.array(dtype=vec3d),
    ref_pts: wp.array(dtype=vec3d),
    radius: wp.array(dtype=f64),
    body: wp.array(dtype=int),
    element: wp.array(dtype=int),
    pair_class: wp.array2d(dtype=int),
    r_out: wp.array(dtype=f64),
    w_tol: f64,
    pt_nodes: wp.array2d(dtype=int),
    exclusion: int,
    search_radius: f64,
    r_self: f64,
    counts: wp.array(dtype=int),
):
    i = wp.tid()
    n = int(0)
    q = wp.hash_grid_query(grid, pts[i], search_radius)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j > i:
            if pair_active(i, j, pts, normals, radius, body, pair_class, r_out, w_tol):
                if not pair_excluded(i, j, body, element, ref_pts, r_self, pt_nodes, exclusion):
                    n += 1
    counts[i] = n


@wp.kernel
def fill_pairs(
    grid: wp.uint64,
    pts: wp.array(dtype=vec3d),
    normals: wp.array(dtype=vec3d),
    ref_pts: wp.array(dtype=vec3d),
    radius: wp.array(dtype=f64),
    body: wp.array(dtype=int),
    element: wp.array(dtype=int),
    pair_class: wp.array2d(dtype=int),
    r_out: wp.array(dtype=f64),
    w_tol: f64,
    pt_nodes: wp.array2d(dtype=int),
    exclusion: int,
    search_radius: f64,
    r_self: f64,
    offsets: wp.array(dtype=int),
    pairs: wp.array(dtype=wp.vec2i),
):
    i = wp.tid()
    k = offsets[i]
    cap = pairs.shape[0]
    q = wp.hash_grid_query(grid, pts[i], search_radius)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j > i:
            if pair_active(i, j, pts, normals, radius, body, pair_class, r_out, w_tol):
                if not pair_excluded(i, j, body, element, ref_pts, r_self, pt_nodes, exclusion):
                    if k < cap:
                        pairs[k] = wp.vec2i(i, j)
                    k += 1


@wp.kernel
def total_from_scan(
    counts: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    n_pairs: wp.array(dtype=int),
):
    last = counts.shape[0] - 1
    n_pairs[0] = offsets[last] + counts[last]


class PairSearch:
    """Device-resident contact points and their current pair list.

    ``points`` defines the point set; ``pair_class`` is a symmetric
    [n_bodies, n_bodies] integer table (-1 means the two bodies never
    interact); ``r_out`` holds the outer radius of each class. ``ref_node_x``
    is the reference configuration used for the reference-distance exclusion.
    ``exclusion`` is ``"element"``, ``"neighbors"`` or ``"none"`` (module
    notes). ``grid_dim`` is the hash grid resolution per axis and
    ``pairs_per_point`` the initial pair-buffer capacity per point; both are
    performance tunables, and the buffer grows on overflow.
    """

    EXCLUSION_MODES = {"none": 0, "element": 1, "neighbors": 2}

    def __init__(
        self,
        points: ContactPoints,
        pair_class: np.ndarray,
        r_out: np.ndarray,
        ref_node_x: np.ndarray,
        r_self: float,
        exclusion: str = "neighbors",
        w_tol: float = 0.0,
        device: str = "cpu",
        grid_dim: int = 64,
        pairs_per_point: int = 48,
    ):
        self.points = points
        self.device = device
        self.n_pts = points.count
        if exclusion not in self.EXCLUSION_MODES:
            raise ValueError("exclusion must be 'none', 'element' or 'neighbors'")
        self.exclusion = self.EXCLUSION_MODES[exclusion]
        self.r_self = float(r_self)
        self.w_tol = float(w_tol)
        pair_class = np.asarray(pair_class, dtype=np.int32)
        r_out = np.asarray(r_out, dtype=np.float64)
        if pair_class.ndim != 2 or pair_class.shape[0] != pair_class.shape[1]:
            raise ValueError("pair_class must be a square [n_bodies, n_bodies] table")
        if not np.array_equal(pair_class, pair_class.T):
            raise ValueError("pair_class must be symmetric: pairs are stored unordered")
        if points.body.max() >= pair_class.shape[0]:
            raise ValueError("a body id exceeds the pair_class table")
        if pair_class.max() >= r_out.shape[0]:
            raise ValueError("a pair class exceeds the parameter table")

        self.pt_nodes = wp.array(points.pt_nodes, dtype=int, device=device)
        self.pt_w = wp.array(points.pt_w, dtype=f64, device=device)
        self.pt_dw1 = wp.array(points.pt_dw1, dtype=f64, device=device)
        self.pt_dw2 = wp.array(points.pt_dw2, dtype=f64, device=device)
        self.rule_w = wp.array(points.rule_weight, dtype=f64, device=device)
        self.radius = wp.array(points.radius, dtype=f64, device=device)
        self.body = wp.array(points.body, dtype=int, device=device)
        self.element = wp.array(points.element, dtype=int, device=device)
        self.pair_class = wp.array(pair_class, dtype=int, device=device)
        self.r_out = wp.array(r_out, dtype=f64, device=device)

        # pts carries a gradient buffer so the energy can be recorded on a tape
        self.pts = wp.zeros(self.n_pts, dtype=vec3d, device=device, requires_grad=True)
        self.normals = wp.zeros(self.n_pts, dtype=vec3d, device=device)
        self.cur_weight = wp.zeros(self.n_pts, dtype=f64, device=device)
        self.counts = wp.zeros(self.n_pts, dtype=int, device=device)
        self.offsets = wp.zeros(self.n_pts, dtype=int, device=device)
        self.n_pairs_dev = wp.zeros(1, dtype=int, device=device)
        self.capacity = int(pairs_per_point * self.n_pts)
        self.pairs = wp.zeros(self.capacity, dtype=wp.vec2i, device=device)
        self.grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device=device, dtype=f64)
        # every pair whose effective distance can be below r_out must be in reach
        self.search_radius = float(r_out.max() + 2.0 * points.radius.max())
        self.n_pairs = 0

        ref_x = wp.array(np.asarray(ref_node_x, dtype=np.float64), dtype=vec3d, device=device)
        self.ref_pts = wp.zeros(self.n_pts, dtype=vec3d, device=device)
        wp.launch(
            interpolate_points,
            dim=self.n_pts,
            inputs=[ref_x, self.pt_nodes, self.pt_w],
            outputs=[self.ref_pts],
            device=device,
        )

    def update_points(self, node_x: wp.array):
        """Positions, current normals and current weights for ``node_x``."""
        wp.launch(
            point_frames,
            dim=self.n_pts,
            inputs=[
                node_x,
                self.pt_nodes,
                self.pt_w,
                self.pt_dw1,
                self.pt_dw2,
                self.rule_w,
            ],
            outputs=[self.pts, self.normals, self.cur_weight],
            device=self.device,
        )

    def _selection_args(self):
        return [
            self.grid.id,
            self.pts,
            self.normals,
            self.ref_pts,
            self.radius,
            self.body,
            self.element,
            self.pair_class,
            self.r_out,
            f64(self.w_tol),
            self.pt_nodes,
            self.exclusion,
            f64(self.search_radius),
            f64(self.r_self),
        ]

    def build(self, node_x: wp.array, sync: bool = True) -> int:
        """Enumerate the pairs of configuration ``node_x``.

        With ``sync=True`` the pair count is read back and the buffer regrown
        and refilled on overflow. With ``sync=False`` nothing touches the host;
        call ``check_overflow`` later (once per step is enough) and rebuild if
        it returns True, since an overflowed list is incomplete.
        """
        self.update_points(node_x)
        self.grid.build(self.pts, self.search_radius)
        sel = self._selection_args()
        wp.launch(
            count_pairs,
            dim=self.n_pts,
            inputs=sel,
            outputs=[self.counts],
            device=self.device,
        )
        wp.utils.array_scan(self.counts, self.offsets, inclusive=False)
        wp.launch(
            total_from_scan,
            dim=1,
            inputs=[self.counts, self.offsets],
            outputs=[self.n_pairs_dev],
            device=self.device,
        )
        wp.launch(
            fill_pairs,
            dim=self.n_pts,
            inputs=sel + [self.offsets],
            outputs=[self.pairs],
            device=self.device,
        )
        if sync and self.check_overflow():
            return self.build(node_x, sync=True)
        return self.n_pairs

    def check_overflow(self) -> bool:
        """Host read of the pair count. On overflow the buffer is regrown, the
        device count is reset to zero so that no kernel consumes the incomplete
        list, and True is returned: the caller must build again."""
        self.n_pairs = int(self.n_pairs_dev.numpy()[0])
        if self.n_pairs > self.capacity:
            self.capacity = int(1.5 * self.n_pairs) + 16
            self.pairs = wp.zeros(self.capacity, dtype=wp.vec2i, device=self.device)
            self.n_pairs_dev.zero_()
            self.n_pairs = 0
            return True
        return False

    def pair_array(self) -> np.ndarray:
        """Host copy of the current pairs, [n_pairs, 2]."""
        self.check_overflow()
        return self.pairs.numpy()[: self.n_pairs].copy()
