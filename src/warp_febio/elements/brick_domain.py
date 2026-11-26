"""Brick domain: hex8 solids and quad4 shells as one term.

TEMPLATE (run path draft, November 2026). Structure for discussion, not final.
The element mathematics is that of the fixture in elements/brick8.py, which
documents how a FEBio quad4 "elastic-shell" is the same trilinear brick as a
hex8. What changes here is the structure around it.

FEBio counterpart
    FEElasticSolidDomain and FEElasticShellDomain: Update (stresses at the
    integration points), InternalForces, StiffnessMatrix.
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticSolidDomain.cpp
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticShellDomain.cpp

Same as FEBio
    The three calls and what each computes; updated Lagrangian; FEBio's
    integration points in FEBio's order; Cauchy stress and the spatial tangent
    from the material.

Different, and why
    A domain is a batch: one element type, one material tree, three kernels
    with no branch on type. FEBio stores stress on material point objects;
    here `update` stores point data in flat device arrays (stress, 6x6
    tangent, spatial shape gradients, weight) and the force and stiffness
    kernels read them, so the material is called once per point per iterate
    (lesson 1 of the assembly note). The force path (`update`, `residual`) is
    written for reverse mode, so the adjoint gets d(residual)/d(parameters)
    from a tape without a second implementation of the element.
"""

# No `from __future__ import annotations` here: the kernels below are built in a
# closure and Warp evaluates their annotations (pvec, gvec) when they are made.
from typing import Any

import numpy as np
import warp as wp

from ..assembly.system import BlockTangent, State, Term
from ..materials.status import NEGATIVE_JACOBIAN, NON_FINITE, OK
from .brick8 import GAUSS_WEIGHTS, shape_derivatives, voigt

vec3d = wp.vec3d
mat33d = wp.mat33d
mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)

_KERNELS = {}


def _make_kernels(bound):
    """The three kernels of a brick domain for one material tree."""
    key = bound.model.name
    if key in _KERNELS:
        return _KERNELS[key]
    stress_and_tangent = bound.stress_and_tangent()
    pvec = bound.model.param_vector(wp.float64)
    gvec = bound.model.geometry_vector(wp.float64)

    def update(
        X: wp.array2d(dtype=vec3d),  # (E, 8) reference positions
        emap: wp.array2d(dtype=wp.int32),  # (E, 8) entry of each brick node
        u: wp.array(dtype=vec3d),  # solution, one 3-vector per entry
        dN: wp.array3d(dtype=wp.float64),  # (Q, 8, 3) shape derivatives
        gw: wp.array(dtype=wp.float64),  # (Q,) weights
        params: wp.array(dtype=pvec),  # material parameters of the batch
        geom: wp.array(dtype=gvec),  # (E,) material axes and fibers
        sigma: wp.array2d(dtype=mat33d),  # (E, Q) Cauchy stress
        c: wp.array2d(dtype=mat66d),  # (E, Q) spatial tangent, tens4ds order
        grad: wp.array3d(dtype=vec3d),  # (E, Q, 8) spatial gradient of N_i
        w: wp.array2d(dtype=wp.float64),  # (E, Q) current volume weight
        status: wp.array2d(dtype=wp.int32),  # (E, Q)
    ):
        """Point data at one integration point. The only call to the material."""
        e, q = wp.tid()
        J0 = mat33d()
        Jt = mat33d()
        for i in range(8):
            dn = vec3d(dN[q, i, 0], dN[q, i, 1], dN[q, i, 2])
            Xi = X[e, i]
            J0 = J0 + wp.outer(Xi, dn)
            Jt = Jt + wp.outer(Xi + u[emap[e, i]], dn)
        detJ0 = wp.determinant(J0)
        detJt = wp.determinant(Jt)
        # A kernel cannot throw. FEBio throws NegativeJacobian here; this
        # writes a status code that the host reads after the launch.
        if detJ0 > wp.float64(0.0) and detJt > wp.float64(0.0):
            F = Jt * wp.inverse(J0)
            JtiT = wp.transpose(wp.inverse(Jt))
            s, cc = stress_and_tangent(F, geom[e], params[0])
            if wp.isfinite(s) and wp.isfinite(cc):
                sigma[e, q] = s
                c[e, q] = cc
                w[e, q] = detJt * gw[q]
                for i in range(8):
                    grad[e, q, i] = JtiT * vec3d(dN[q, i, 0], dN[q, i, 1], dN[q, i, 2])
                status[e, q] = wp.static(OK)
            else:
                status[e, q] = wp.static(NON_FINITE)
        else:
            status[e, q] = wp.static(NEGATIVE_JACOBIAN)

    def internal_force(
        emap: wp.array2d(dtype=wp.int32),
        sigma: wp.array2d(dtype=mat33d),
        grad: wp.array3d(dtype=vec3d),
        w: wp.array2d(dtype=wp.float64),
        residual: wp.array(dtype=vec3d),
    ):
        """residual -= f_i, f_i = sum_q sigma grad N_i dv (FEBio's sign)."""
        e, i = wp.tid()
        f = vec3d()
        for q in range(8):
            f = f + sigma[e, q] * grad[e, q, i] * w[e, q]
        wp.atomic_sub(residual, emap[e, i], f)

    def stiffness(
        emap: wp.array2d(dtype=wp.int32),
        sigma: wp.array2d(dtype=mat33d),
        c: wp.array2d(dtype=mat66d),
        grad: wp.array3d(dtype=vec3d),
        w: wp.array2d(dtype=wp.float64),
        rows: wp.array(dtype=wp.int32),
        cols: wp.array(dtype=wp.int32),
        vals: wp.array(dtype=mat33d),
    ):
        """One 3x3 block per element node pair: material part
        (grad N_i . c . grad N_j, FEBio's vdotTdotv) plus geometric part
        (grad N_i . sigma grad N_j) I. Not taped."""
        e, i, j = wp.tid()
        nq = w.shape[1]
        k = mat33d()
        for q in range(nq):
            gi = grad[e, q, i]
            gj = grad[e, q, j]
            cc = c[e, q]
            ww = w[e, q]
            kg = wp.dot(gi, sigma[e, q] * gj) * ww
            for a in range(3):
                for b in range(3):
                    val = wp.float64(0.0)
                    for m in range(3):
                        for n in range(3):
                            val = val + gi[m] * cc[voigt(a, m), voigt(b, n)] * gj[n]
                    k[a, b] = k[a, b] + val * ww
                k[a, a] = k[a, a] + kg
        t = (e * 8 + i) * 8 + j
        rows[t] = emap[e, i]
        cols[t] = emap[e, j]
        vals[t] = k

    kernels = tuple(
        wp.kernel(f, module="unique") for f in (update, internal_force, stiffness)
    )
    _KERNELS[key] = kernels
    return kernels


class BrickDomain(Term):
    """One element block with one material: FEBio "elastic-solid" on hex8, or
    "elastic-shell" on quad4 (front and back face as the brick's two faces)."""

    internal = True

    def __init__(self, name, febio_type, bound, X, emap, n_entries, model, device):
        self.name = name
        self.febio_type = febio_type
        self.element_type = "hex8" if febio_type == "elastic-solid" else "quad4"
        self.material = bound
        self.model = model
        self.device = device
        self.n = n_entries
        self.count = E = len(X)
        Q = len(GAUSS_WEIGHTS)
        self._update, self._force, self._stiffness = _make_kernels(bound)
        self.pvec = bound.model.param_vector(wp.float64)
        gvec = bound.model.geometry_vector(wp.float64)

        dev = dict(device=device)
        self.X = wp.array(X, dtype=vec3d, **dev)
        self.emap = wp.array(emap.astype(np.int32), dtype=wp.int32, **dev)
        self.dN = wp.array(shape_derivatives(), dtype=wp.float64, **dev)
        self.gw = wp.array(GAUSS_WEIGHTS, dtype=wp.float64, **dev)
        self.geom = wp.array(bound.geometry_array(E), dtype=gvec, **dev)
        # Parameters live in a device array, so they can follow a load curve
        # (one write per step) and carry gradients.
        self.params = wp.array(
            [self.pvec(*bound.param_values(0.0))], dtype=self.pvec, requires_grad=True, **dev
        )
        # Point data. requires_grad lets the adjoint tape pass through it.
        grad_on = dict(requires_grad=True, **dev)
        self.sigma = wp.zeros((E, Q), dtype=mat33d, **grad_on)
        self.c = wp.zeros((E, Q), dtype=mat66d, **grad_on)
        self.grad = wp.zeros((E, Q, 8), dtype=vec3d, **grad_on)
        self.w = wp.zeros((E, Q), dtype=wp.float64, **grad_on)
        self.status = wp.zeros((E, Q), dtype=wp.int32, **dev)
        # Stiffness blocks: fixed count and fixed (row, column) for the mesh.
        # The system hands the term its slice of the block buffer.
        self.block_count = E * 64

    # -- Term ----------------------------------------------------------------

    def update(self, state: State) -> int:
        if self.material.time_dependent:
            self.set_parameters(self.material.param_values(state.time))
        wp.launch(
            self._update,
            dim=(self.count, len(GAUSS_WEIGHTS)),
            inputs=[self.X, self.emap, state.u, self.dN, self.gw, self.params, self.geom],
            outputs=[self.sigma, self.c, self.grad, self.w, self.status],
            device=self.device,
        )
        # One small device-to-host read per update, where FEBio would throw.
        return int(self.status.numpy().max()) if self.count else OK

    def residual(self, state: State, history: Any, out: wp.array) -> None:
        wp.launch(
            self._force,
            dim=(self.count, 8),
            inputs=[self.emap, self.sigma, self.grad, self.w],
            outputs=[out],
            device=self.device,
        )

    def linearize(self, state: State, history: Any, exact: bool = False) -> BlockTangent:
        # Always the exact derivative of the internal force.
        wp.launch(
            self._stiffness,
            dim=(self.count, 8, 8),
            inputs=[self.emap, self.sigma, self.c, self.grad, self.w],
            outputs=[self.rows, self.cols, self.vals],
            device=self.device,
        )
        return BlockTangent(self.n, self.rows, self.cols, self.vals)

    def parameters(self) -> dict:
        return {"material": self.params}

    # -- beyond Term ---------------------------------------------------------

    @property
    def parameter_names(self) -> tuple:
        return tuple(self.material.model.param_names)

    def set_parameters(self, values) -> None:
        self.params.assign([self.pvec(*[float(v) for v in values])])

    def plot(self, variable: str) -> np.ndarray:
        """Host copy of a plot variable at the last update, FEBio's name."""
        if variable != "stress":
            raise KeyError(variable)
        return self.sigma.numpy()  # (E, Q, 3, 3)
