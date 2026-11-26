"""Pressure on a surface, as a term.

TEMPLATE (run path draft, November 2026). Structure for discussion, not final.

FEBio counterpart
    FEPressureLoad (surface_load type "pressure"), integrated by
    FESurface::LoadVector and FESurface::LoadStiffness.
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEPressureLoad.cpp
    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FESurface.cpp

Same as FEBio
    The four parameters and their defaults (pressure, symmetric_stiffness = 1,
    linear = 0, shell_bottom = 0). The load vector on node a,
        f_a = sum_n w_n H_a (-p) (dx/dr x dx/ds),
    on the current facet geometry (a follower load), or on the reference
    geometry when linear = 1. quad4 facets with FEBio's 2x2 Gauss rule
    (FEQuad4G4). A positive pressure pushes against the facet normal. The
    pressure is scaled by its load curve at the time being solved. With
    shell_bottom the facet is the back face of the shell and the sign flips.
    Both of FEBio's stiffness forms, selected by symmetric_stiffness.

Different, and why
    It is a Term like any other, so the solver does not know it is a load.
    The pressure value is a device array, so it can carry a gradient.
    FEBio's default stiffness for this load is a symmetrized form, which is
    not the derivative of the load vector element by element. Newton may use
    it; the adjoint may not. `linearize(exact=True)` returns the true
    derivative (FEBio's non-symmetric form) whatever the input asks for.

Not implemented
    tri3 and higher-order facets; a pressure given as a map over the surface.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import warp as wp

from ..assembly.system import BlockTangent, State, Term
from ..coverage import IMPLEMENTED, NOT_SUPPORTED

vec3d = wp.vec3d
mat33d = wp.mat33d

TYPE = "pressure"
_SETTINGS = ("pressure", "symmetric_stiffness", "linear", "shell_bottom")

# FEQuad4G4: integration points and weights in FEBio's order, and the quad4
# shape functions and their derivatives at those points.
_A = 1.0 / np.sqrt(3.0)
_GAUSS = np.array([[-_A, -_A], [_A, -_A], [_A, _A], [-_A, _A]])
_NODE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def _shape_tables():
    r, s = _GAUSS[:, :1], _GAUSS[:, 1:]
    ri, si = _NODE[:, 0], _NODE[:, 1]
    H = 0.25 * (1 + r * ri) * (1 + s * si)  # (point, node)
    Gr = 0.25 * ri * (1 + s * si)
    Gs = 0.25 * si * (1 + r * ri)
    return H, Gr, Gs


@wp.func
def _tangents(
    f: int,
    n: int,
    emap: wp.array2d(dtype=wp.int32),
    X: wp.array2d(dtype=vec3d),
    u: wp.array(dtype=vec3d),
    follow: wp.float64,
    Gr: wp.array2d(dtype=wp.float64),
    Gs: wp.array2d(dtype=wp.float64),
):
    """Covariant tangents dx/dr, dx/ds of facet f at integration point n."""
    dxr = vec3d()
    dxs = vec3d()
    for j in range(4):
        x = X[f, j] + follow * u[emap[f, j]]
        dxr = dxr + Gr[n, j] * x
        dxs = dxs + Gs[n, j] * x
    return dxr, dxs


@wp.kernel
def _load_vector(
    emap: wp.array2d(dtype=wp.int32),  # (F, 4) entry of each facet node
    X: wp.array2d(dtype=vec3d),  # (F, 4) reference positions
    u: wp.array(dtype=vec3d),
    follow: wp.float64,  # 1: current geometry; 0: reference (linear = 1)
    H: wp.array2d(dtype=wp.float64),
    Gr: wp.array2d(dtype=wp.float64),
    Gs: wp.array2d(dtype=wp.float64),
    pressure: wp.array(dtype=wp.float64),  # (1,) the parameter
    factor: wp.float64,  # load curve value and the shell_bottom sign
    residual: wp.array(dtype=vec3d),
):
    """residual += f_a (an external force). Written for reverse mode."""
    f, a = wp.tid()
    acc = vec3d()
    for n in range(4):
        dxr, dxs = _tangents(f, n, emap, X, u, follow, Gr, Gs)
        acc = acc + H[n, a] * wp.cross(dxr, dxs)  # weights are 1
    wp.atomic_add(residual, emap[f, a], -pressure[0] * factor * acc)


@wp.kernel
def _load_stiffness(
    emap: wp.array2d(dtype=wp.int32),
    X: wp.array2d(dtype=vec3d),
    u: wp.array(dtype=vec3d),
    H: wp.array2d(dtype=wp.float64),
    Gr: wp.array2d(dtype=wp.float64),
    Gs: wp.array2d(dtype=wp.float64),
    pressure: wp.array(dtype=wp.float64),
    factor: wp.float64,
    symmetric: int,
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    vals: wp.array(dtype=mat33d),
):
    """Contribution to K = -dR/du: one 3x3 block per facet node pair, FEBio's
    two forms. The block is the skew matrix of a vector (FEBio's mat3da)."""
    f, i, j = wp.tid()
    P = -pressure[0] * factor
    one = wp.float64(1.0)
    k = mat33d()
    for n in range(4):
        dxr, dxs = _tangents(f, n, emap, X, u, one, Gr, Gs)
        if symmetric != 0:
            v = (
                dxr * (H[n, j] * Gs[n, i] - H[n, i] * Gs[n, j])
                - dxs * (H[n, j] * Gr[n, i] - H[n, i] * Gr[n, j])
            ) * (wp.float64(0.5) * P)
        else:
            v = (dxs * Gr[n, j] - dxr * Gs[n, j]) * (P * H[n, i])
        k = k + wp.skew(v)
    t = (f * 4 + i) * 4 + j
    rows[t] = emap[f, i]
    cols[t] = emap[f, j]
    vals[t] = k


def inspect_load(record, model):
    """Coverage of one <surface_load>: (status, note), or None if it is not
    a pressure load. Used by the coverage report."""
    if record.tag != "surface_load" or record.type != TYPE:
        return None
    surface = model.mesh.surfaces.get(record.attrs.get("surface", ""))
    if surface is None:
        return NOT_SUPPORTED, f"surface {record.attrs.get('surface')!r} not resolved"
    if any(t != "quad4" for t in surface.facet_types):
        return NOT_SUPPORTED, "quad4 facets only in this draft"
    for c in record.children:
        if c.tag not in _SETTINGS:
            return NOT_SUPPORTED, f"setting {c.tag!r}"
    p = record.child("pressure")
    if p is None or not np.isscalar(p.value) or p.attrs.get("type"):
        return NOT_SUPPORTED, "a pressure that is not one number with a load curve"
    return IMPLEMENTED, ""


class PressureLoad(Term):
    """FEBio surface_load "pressure" on quad4 facets."""

    febio_type = TYPE

    def __init__(self, record, model, fields, directors, device):
        self.name = record.name or record.attrs["surface"]
        self.model = model
        self.device = device
        self.n = fields.count
        surface = model.mesh.surfaces[record.attrs["surface"]]
        facets = np.stack(surface.facets).astype(np.int64)  # (F, 4) node indices
        self.count = F = len(facets)

        p = record.child("pressure")
        lc = p.attrs.get("lc")
        self.load_controller = int(lc) if lc else None
        self.symmetric = int(record.get("symmetric_stiffness", 1))
        self.linear = int(record.get("linear", 0))
        self.shell_bottom = int(record.get("shell_bottom", 0))

        X = model.mesh.nodes[facets]
        if self.shell_bottom:
            # The back face of the shell: s = r - d, shell displacement.
            emap = fields.entries_of("shell displacement", facets)
            X = X - directors[facets]
        else:
            emap = fields.entries_of("displacement", facets)
        if (emap < 0).any():
            raise ValueError(f"pressure load {self.name!r}: facet nodes without the variable")

        H, Gr, Gs = _shape_tables()
        dev = dict(device=device)
        self.emap = wp.array(emap.astype(np.int32), dtype=wp.int32, **dev)
        self.X = wp.array(X, dtype=vec3d, **dev)
        self.H = wp.array(H, dtype=wp.float64, **dev)
        self.Gr = wp.array(Gr, dtype=wp.float64, **dev)
        self.Gs = wp.array(Gs, dtype=wp.float64, **dev)
        # The parameter, on the device so it can carry a gradient.
        self.pressure = wp.array([float(p.value)], dtype=wp.float64, requires_grad=True, **dev)
        self.rows = wp.zeros(F * 16, dtype=wp.int32, **dev)
        self.cols = wp.zeros(F * 16, dtype=wp.int32, **dev)
        self.vals = wp.zeros(F * 16, dtype=mat33d, **dev)

    def factor(self, time: float) -> float:
        f = 1.0
        if self.load_controller is not None:
            f = self.model.load_controller(self.load_controller).value(time)
        return -f if self.shell_bottom else f

    # -- Term ----------------------------------------------------------------
    # No `update`: the load keeps no data of its own between passes.

    def residual(self, state: State, history: Any, out: wp.array) -> None:
        wp.launch(
            _load_vector,
            dim=(self.count, 4),
            inputs=[self.emap, self.X, state.u, 0.0 if self.linear else 1.0,
                    self.H, self.Gr, self.Gs, self.pressure, self.factor(state.time)],
            outputs=[out],
            device=self.device,
        )

    def linearize(self, state: State, history: Any, exact: bool = False) -> Optional[BlockTangent]:
        if self.linear:
            return None  # the load does not depend on the solution
        symmetric = 0 if exact else self.symmetric
        wp.launch(
            _load_stiffness,
            dim=(self.count, 4, 4),
            inputs=[self.emap, self.X, state.u, self.H, self.Gr, self.Gs,
                    self.pressure, self.factor(state.time), symmetric],
            outputs=[self.rows, self.cols, self.vals],
            device=self.device,
        )
        return BlockTangent(self.n, self.rows, self.cols, self.vals)

    def parameters(self) -> dict:
        return {"pressure": self.pressure}
