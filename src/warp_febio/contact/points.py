"""Contact point sets: positions interpolated from nodes through fixed weights.

A contact point is a node-weighted location on a body. The pair kernels in
``contact/potential.py`` never see an element type; they see positions, an
integration weight, a body id, an optional radius offset and the node list the
point interpolates from. That is what lets one module serve shell Gauss points
(leaflets), facet points of a solid surface (clip arms, a tetrahedral device
frame, a hex vessel wall), points along a cable (chordae) and points along a
beam centerline (a nitinol strut or wire).

Conventions, following FEBio's FEContactPotential and
FEEdgeToSurfaceContactPotential:

* ``weight`` is the reference integration weight of the point: reference-facet
  Jacobian times rule weight for a surface (an area), segment length times
  rule weight for a line. FEBio uses the current Jacobian; the derivative
  weights carried here let ``weight_mode="current"`` reproduce that.
* ``area`` is the surface area carried by the point, used for the contact area
  metric of Wu 2022 (area over points with nonzero contact force). It is zero
  for line points.
* ``radius`` is a per-point offset subtracted from the pair distance, so a wire
  of radius r_w on its centerline is treated as a capsule. FEBio has no such
  term; set the radius to zero and put the wire radius into R_in for a
  FEBio-equivalent run.
* ``body`` groups points for self-contact exclusion; pairs on different bodies
  always interact.

Lengths follow the caller's unit convention (SI inside this package).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContactPoints:
    """Node-weighted contact points of one or more bodies.

    Arrays (host, numpy): ``pt_nodes`` [n, K] node ids with -1 in unused slots,
    ``pt_w`` [n, K] shape-function weights, ``weight`` [n] (reference
    integration weight), ``area`` [n], ``radius`` [n], ``body`` [n] (int32),
    ``element`` [n] (int32, the source facet or segment; informational).

    Optional, for the FEBio-equivalent reference tier: ``pt_dw1`` and
    ``pt_dw2`` [n, K] are the derivative weights of the two parametric
    tangents (one tangent for a line, none for a bare node) and
    ``rule_weight`` [n] the rule weight without the Jacobian, so that a
    kernel can evaluate the current normal and the current Jacobian-weighted
    integration weight the way FEBio does (``FEContactPotential::UpdateSurface``).
    """

    pt_nodes: np.ndarray
    pt_w: np.ndarray
    weight: np.ndarray
    area: np.ndarray
    radius: np.ndarray
    body: np.ndarray
    element: np.ndarray
    pt_dw1: np.ndarray | None = None
    pt_dw2: np.ndarray | None = None
    rule_weight: np.ndarray | None = None

    def __post_init__(self):
        for name in ("pt_nodes", "pt_w", "weight", "area", "radius", "body", "element"):
            setattr(self, name, np.asarray(getattr(self, name)))
        n, K = self.pt_nodes.shape
        for name in ("pt_w", "weight", "area", "radius", "body", "element"):
            if getattr(self, name).shape[0] != n:
                raise ValueError(f"{name} has {getattr(self, name).shape[0]} rows, expected {n}")
        self.pt_nodes = np.ascontiguousarray(self.pt_nodes, dtype=np.int32)
        self.pt_w = np.ascontiguousarray(self.pt_w, dtype=np.float64)
        self.weight = np.ascontiguousarray(self.weight, dtype=np.float64)
        self.area = np.ascontiguousarray(self.area, dtype=np.float64)
        self.radius = np.ascontiguousarray(self.radius, dtype=np.float64)
        self.body = np.ascontiguousarray(self.body, dtype=np.int32)
        self.element = np.ascontiguousarray(self.element, dtype=np.int32)
        if self.pt_dw1 is None:
            self.pt_dw1 = np.zeros((n, K))
        if self.pt_dw2 is None:
            self.pt_dw2 = np.zeros((n, K))
        if self.rule_weight is None:
            self.rule_weight = self.weight.copy()
        self.pt_dw1 = np.ascontiguousarray(self.pt_dw1, dtype=np.float64)
        self.pt_dw2 = np.ascontiguousarray(self.pt_dw2, dtype=np.float64)
        self.rule_weight = np.ascontiguousarray(self.rule_weight, dtype=np.float64)

    @property
    def count(self) -> int:
        return int(self.pt_nodes.shape[0])

    @property
    def nodes_per_point(self) -> int:
        return int(self.pt_nodes.shape[1])

    def positions(self, node_x: np.ndarray) -> np.ndarray:
        """Host evaluation of the point positions for a nodal configuration."""
        x = np.zeros((self.count, 3))
        for a in range(self.nodes_per_point):
            ok = self.pt_nodes[:, a] >= 0
            x[ok] += self.pt_w[ok, a, None] * node_x[self.pt_nodes[ok, a]]
        return x

    @staticmethod
    def concatenate(sets: list[ContactPoints]) -> ContactPoints:
        """Merge point sets that share one node numbering (bodies keep their ids)."""
        K = max(s.nodes_per_point for s in sets)

        def pad(a, fill):
            out = np.full((a.shape[0], K), fill, dtype=a.dtype)
            out[:, : a.shape[1]] = a
            return out

        return ContactPoints(
            np.concatenate([pad(s.pt_nodes, -1) for s in sets]),
            np.concatenate([pad(s.pt_w, 0.0) for s in sets]),
            np.concatenate([s.weight for s in sets]),
            np.concatenate([s.area for s in sets]),
            np.concatenate([s.radius for s in sets]),
            np.concatenate([s.body for s in sets]),
            np.concatenate([s.element for s in sets]),
            np.concatenate([pad(s.pt_dw1, 0.0) for s in sets]),
            np.concatenate([pad(s.pt_dw2, 0.0) for s in sets]),
            np.concatenate([s.rule_weight for s in sets]),
        )


# -----------------------------------------------------------------------------
# Quadrilateral facets (quad shells, hex faces)
# -----------------------------------------------------------------------------
def _quad_rule(n_per_edge: int, rule: str):
    """Parametric points and weights on [-1, 1]^2 for a quad facet."""
    if rule == "gauss":
        g, w = np.polynomial.legendre.leggauss(n_per_edge)
    elif rule == "lattice":
        # midpoint sub-lattice: n_per_edge^2 equal cells, points at cell centers
        g = -1.0 + (2.0 * np.arange(n_per_edge) + 1.0) / n_per_edge
        w = np.full(n_per_edge, 2.0 / n_per_edge)
    else:
        raise ValueError("rule must be 'gauss' or 'lattice'")
    r, s = np.meshgrid(g, g, indexing="ij")
    wr, ws = np.meshgrid(w, w, indexing="ij")
    return r.ravel(), s.ravel(), (wr * ws).ravel()


def quad_facet_points(
    quads: np.ndarray,
    node_x: np.ndarray,
    body: int,
    n_per_edge: int = 2,
    rule: str = "gauss",
    radius: float = 0.0,
) -> ContactPoints:
    """Points on bilinear Quad4 facets.

    ``n_per_edge=2, rule='gauss'`` is FEBio's default integration rule;
    ``n_per_edge=4`` is its ``higher-order`` rule (16 per quad). ``rule='lattice'``
    gives a regular sub-lattice for densification independent of the mechanics
    quadrature (Kamensky 2018, Section 4.3). Weights are reference-facet
    Jacobian times rule weight; ``area`` equals ``weight``.
    """
    quads = np.asarray(quads, dtype=np.int32)
    node_x = np.asarray(node_x, dtype=np.float64)
    r_all, s_all, w_all = _quad_rule(n_per_edge, rule)
    X = node_x[quads]  # [nq, 4, 3]
    pt_nodes, pt_w, weight, element, dw1, dw2, rule_w = [], [], [], [], [], [], []
    for r, s, w in zip(r_all, s_all, w_all):
        N = 0.25 * np.array([(1 - r) * (1 - s), (1 + r) * (1 - s), (1 + r) * (1 + s), (1 - r) * (1 + s)])
        dNr = 0.25 * np.array([-(1 - s), (1 - s), (1 + s), -(1 + s)])
        dNs = 0.25 * np.array([-(1 - r), -(1 + r), (1 + r), (1 - r)])
        tr = np.einsum("a,qac->qc", dNr, X)
        ts = np.einsum("a,qac->qc", dNs, X)
        J = np.linalg.norm(np.cross(tr, ts), axis=1)
        pt_nodes.append(quads)
        pt_w.append(np.tile(N, (len(quads), 1)))
        dw1.append(np.tile(dNr, (len(quads), 1)))
        dw2.append(np.tile(dNs, (len(quads), 1)))
        weight.append(w * J)
        rule_w.append(np.full(len(quads), w))
        element.append(np.arange(len(quads), dtype=np.int32))
    weight = np.concatenate(weight)
    n = weight.shape[0]
    return ContactPoints(
        np.concatenate(pt_nodes),
        np.concatenate(pt_w),
        weight,
        weight.copy(),
        np.full(n, radius),
        np.full(n, body, dtype=np.int32),
        np.concatenate(element),
        np.concatenate(dw1),
        np.concatenate(dw2),
        np.concatenate(rule_w),
    )


# -----------------------------------------------------------------------------
# Triangular facets (triangle shells, tet surfaces)
# -----------------------------------------------------------------------------
def _tri_rule(order: str):
    """Barycentric points and weights (summing to 1) on the reference triangle."""
    if order == "1":
        return np.array([[1 / 3, 1 / 3, 1 / 3]]), np.array([1.0])
    if order == "3":
        return np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]), np.full(3, 1 / 3)
    if order == "7":
        a, b = 0.059715871789770, 0.470142064105115
        c, d = 0.797426985353087, 0.101286507323456
        pts = np.array(
            [
                [1 / 3, 1 / 3, 1 / 3],
                [a, b, b],
                [b, a, b],
                [b, b, a],
                [c, d, d],
                [d, c, d],
                [d, d, c],
            ]
        )
        w = np.array(
            [
                0.225,
                0.132394152788506,
                0.132394152788506,
                0.132394152788506,
                0.125939180544827,
                0.125939180544827,
                0.125939180544827,
            ]
        )
        return pts, w
    raise ValueError("order must be '1', '3' or '7'")


def tri_facet_points(
    tris: np.ndarray,
    node_x: np.ndarray,
    body: int,
    order: str = "3",
    radius: float = 0.0,
) -> ContactPoints:
    """Points on linear Tri3 facets. ``order='7'`` is FEBio's higher-order rule."""
    tris = np.asarray(tris, dtype=np.int32)
    node_x = np.asarray(node_x, dtype=np.float64)
    bary, w_all = _tri_rule(order)
    X = node_x[tris]
    area = 0.5 * np.linalg.norm(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), axis=1)
    pt_nodes, pt_w, weight, element, rule_w = [], [], [], [], []
    for N, w in zip(bary, w_all):
        pt_nodes.append(tris)
        pt_w.append(np.tile(N, (len(tris), 1)))
        weight.append(w * area)
        rule_w.append(np.full(len(tris), 0.5 * w))  # area = 0.5 |t1 x t2|
        element.append(np.arange(len(tris), dtype=np.int32))
    weight = np.concatenate(weight)
    n = weight.shape[0]
    return ContactPoints(
        np.concatenate(pt_nodes),
        np.concatenate(pt_w),
        weight,
        weight.copy(),
        np.full(n, radius),
        np.full(n, body, dtype=np.int32),
        np.concatenate(element),
        np.tile([-1.0, 1.0, 0.0], (n, 1)),
        np.tile([-1.0, 0.0, 1.0], (n, 1)),
        np.concatenate(rule_w),
    )


def boundary_triangles(tets: np.ndarray) -> np.ndarray:
    """Outward-oriented boundary faces of a Tet4 mesh (faces that appear once)."""
    tets = np.asarray(tets, dtype=np.int64)
    # faces opposite each vertex, oriented outward for positive-volume tets
    faces = np.concatenate(
        [
            tets[:, [1, 2, 3]],
            tets[:, [0, 3, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 1]],
        ]
    )
    key = np.sort(faces, axis=1)
    _, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    return faces[idx[counts == 1]].astype(np.int32)


# -----------------------------------------------------------------------------
# Line segments (cables, beam centerlines)
# -----------------------------------------------------------------------------
def segment_points(
    segments: np.ndarray,
    node_x: np.ndarray,
    body: int,
    spacing: float | None = None,
    n_per_segment: int | None = None,
    radius: float = 0.0,
    rule: str = "lattice",
) -> ContactPoints:
    """Points along 2-node segments: chordae (radius 0) or beam centerlines.

    Either ``spacing`` (target arc-length spacing, at most R_out/2 is the rule)
    or ``n_per_segment``. ``rule='nodal'`` puts one point at each end of the
    segment with half its length as weight, which is FEBio's line contact
    element (FE_LINE2NI, nodal integration); ``'gauss'`` uses Gauss points;
    ``'lattice'`` places points at equal sub-interval midpoints. Weight is
    segment length times rule weight; area is zero. The radius is subtracted
    from pair distances (capsule).
    """
    segments = np.asarray(segments, dtype=np.int32)
    node_x = np.asarray(node_x, dtype=np.float64)
    length = np.linalg.norm(node_x[segments[:, 1]] - node_x[segments[:, 0]], axis=1)
    if rule == "nodal":
        if spacing is not None or n_per_segment not in (None, 2):
            raise ValueError("rule='nodal' fixes two points per segment")
        n_per_segment = 2
    if (spacing is None) == (n_per_segment is None):
        raise ValueError("give exactly one of spacing or n_per_segment")
    pt_nodes, pt_w, weight, element, rule_w = [], [], [], [], []
    for e, (seg, L) in enumerate(zip(segments, length)):
        n = n_per_segment if n_per_segment is not None else max(1, int(np.ceil(L / spacing)))
        if rule == "nodal":
            n = 2
            t = np.array([0.0, 1.0])
            w = np.array([0.5, 0.5])
        elif rule == "gauss":
            g, w = np.polynomial.legendre.leggauss(n)
            t = 0.5 * (g + 1.0)
            w = 0.5 * w
        elif rule == "lattice":
            t = (np.arange(n) + 0.5) / n
            w = np.full(n, 1.0 / n)
        else:
            raise ValueError("rule must be 'nodal', 'gauss' or 'lattice'")
        pt_nodes.append(np.tile(seg, (n, 1)))
        pt_w.append(np.stack([1.0 - t, t], axis=1))
        weight.append(w * L)
        rule_w.append(w)
        element.append(np.full(n, e, dtype=np.int32))
    weight = np.concatenate(weight)
    n = weight.shape[0]
    return ContactPoints(
        np.concatenate(pt_nodes),
        np.concatenate(pt_w),
        weight,
        np.zeros(n),
        np.full(n, radius),
        np.full(n, body, dtype=np.int32),
        np.concatenate(element),
        np.tile([-1.0, 1.0], (n, 1)),
        np.zeros((n, 2)),
        np.concatenate(rule_w),
    )


def node_points(node_ids: np.ndarray, body: int, weight: float = 1.0, radius: float = 0.0) -> ContactPoints:
    """Nodes used directly as contact points (tests, rigid point clouds)."""
    node_ids = np.asarray(node_ids, dtype=np.int32)
    n = node_ids.shape[0]
    return ContactPoints(
        node_ids[:, None],
        np.ones((n, 1)),
        np.full(n, weight),
        np.zeros(n),
        np.full(n, radius),
        np.full(n, body, dtype=np.int32),
        np.arange(n, dtype=np.int32),
    )
