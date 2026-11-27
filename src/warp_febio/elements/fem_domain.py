"""Solid domain built on warp.fem: hex8 and tet4.

TEMPLATE (run path draft, November 2026). Structure for discussion, not final.

Why this exists
    warp.fem already provides what a displacement-based solid element needs:
    the geometry (Hexmesh, Tetmesh), the trilinear and linear spaces, Gauss
    quadrature, integration of a linear form to a nodal vector and of a
    bilinear form to a block sparse matrix, in double precision, and it is
    differentiable under wp.Tape. This domain uses those pieces and adds only
    the two integrands. It gives tet4 without a hand-written element, and for
    hex8 it is an independent second implementation next to BrickDomain.

FEBio counterpart
    FEElasticSolidDomain. FEBio's node order for hex8 and tet4 is the order
    warp.fem expects. Integration: hex8 2x2x2 Gauss (FE_HEX8G8); tet4 one
    point (FE_TET4G1, FEBio's default).

Same as FEBio
    What is computed: internal force and stiffness of an updated Lagrangian
    solid, from the Cauchy stress and spatial tangent of the material layer.

Different, and why
    The integrals are written over the reference configuration (first
    Piola-Kirchhoff stress against the reference gradient of the test
    function), which is the form warp.fem integrates. It is the same
    integral as FEBio's over the current configuration.
    The material is called inside the integrands, so it is evaluated in the
    force pass and again in the stiffness pass. BrickDomain stores point
    data and evaluates once; doing the same here (warp.fem can read stored
    data by quadrature point index) is the known next step for speed.

Limits
    One element type per domain. No per-point history. Shells are not
    covered: warp.fem has no thick shell element.
"""

import re
from typing import Any

import numpy as np
import warp as wp
import warp.fem as fem

from ..assembly.system import BlockTangent, State, Term
from ..materials.status import NEGATIVE_JACOBIAN, NON_FINITE, OK

vec3d = wp.vec3d
mat33d = wp.mat33d

# FEBio element type -> (warp.fem geometry, quadrature order, points per element)
ELEMENTS = {"hex8": (fem.Hexmesh, 2, 8), "tet4": (fem.Tetmesh, 0, 1)}

_FORMS = {}


def _make_forms(bound):
    """Integrands for one material tree."""
    key = bound.model.name
    if key in _FORMS:
        return _FORMS[key]
    stress_and_tangent = bound.stress_and_tangent()
    pvec = bound.model.param_vector(wp.float64)
    gvec = bound.model.geometry_vector(wp.float64)

    @wp.func
    def deformation_gradient(grad_u: mat33d):
        return grad_u + wp.identity(n=3, dtype=wp.float64)

    def internal_force(
        s: fem.Sample,
        u: fem.Field,
        v: fem.Field,
        params: wp.array(dtype=pvec),
        geom: wp.array(dtype=gvec),
    ):
        """P : Grad v, with P = J sigma F^-T."""
        F = deformation_gradient(fem.grad(u, s))
        sigma, c = stress_and_tangent(F, geom[s.element_index], params[0])
        P = wp.determinant(F) * sigma * wp.transpose(wp.inverse(F))
        return wp.ddot(P, fem.grad(v, s))

    def stiffness(
        s: fem.Sample,
        u: fem.Field,
        du: fem.Field,
        v: fem.Field,
        params: wp.array(dtype=pvec),
        geom: wp.array(dtype=gvec),
    ):
        """(grad v : c : grad du + sigma : (grad du^T grad v)) J, with spatial
        gradients grad w = Grad w F^-1: material and geometric parts."""
        F = deformation_gradient(fem.grad(u, s))
        Fi = wp.inverse(F)
        sigma, c = stress_and_tangent(F, geom[s.element_index], params[0])
        gv = fem.grad(v, s) * Fi
        gu = fem.grad(du, s) * Fi
        two = wp.float64(2.0)  # noqa: F841 (unused; left as written in the draft)
        # symmetric parts in FEBio's tens4ds order xx, yy, zz, xy, yz, xz
        a = wp.vector(
            gv[0, 0],
            gv[1, 1],
            gv[2, 2],
            gv[0, 1] + gv[1, 0],
            gv[1, 2] + gv[2, 1],
            gv[0, 2] + gv[2, 0],
        )
        b = wp.vector(
            gu[0, 0],
            gu[1, 1],
            gu[2, 2],
            gu[0, 1] + gu[1, 0],
            gu[1, 2] + gu[2, 1],
            gu[0, 2] + gu[2, 0],
        )
        return (wp.dot(a, c * b) + wp.ddot(sigma, wp.transpose(gu) * gv)) * wp.determinant(F)

    def point_jacobian(s: fem.Sample, u: fem.Field):
        return wp.determinant(deformation_gradient(fem.grad(u, s)))

    def point_stress(
        s: fem.Sample,
        u: fem.Field,
        params: wp.array(dtype=pvec),
        geom: wp.array(dtype=gvec),
    ):
        F = deformation_gradient(fem.grad(u, s))
        sigma, c = stress_and_tangent(F, geom[s.element_index], params[0])
        return sigma

    # warp.fem caches the kernel it generates for an integrand under the
    # integrand's name. Two materials would otherwise both define
    # "internal_force", and the later definition would silently replace the
    # earlier one, so a domain would integrate another domain's material.
    # Each material tree therefore gets integrands with its own names.
    tag = re.sub(r"\W", "_", key)
    forms = []
    for function in (internal_force, stiffness, point_jacobian, point_stress):
        function.__name__ = function.__qualname__ = f"{function.__name__}__{tag}"
        forms.append(fem.integrand(function))
    _FORMS[key] = tuple(forms)
    return _FORMS[key]


@wp.kernel
def _gather(
    u: wp.array(dtype=vec3d),
    entry: wp.array(dtype=wp.int32),
    out: wp.array(dtype=vec3d),
):
    """Nodal displacements from the entries of the field table."""
    i = wp.tid()
    out[i] = u[entry[i]]


@wp.kernel
def _scatter_subtract(
    f: wp.array(dtype=vec3d),
    entry: wp.array(dtype=wp.int32),
    residual: wp.array(dtype=vec3d),
):
    """residual -= internal force (FEBio's sign)."""
    i = wp.tid()
    wp.atomic_sub(residual, entry[i], f[i])


class FemSolidDomain(Term):
    """FEBio "elastic-solid" on hex8 or tet4, integrated by warp.fem."""

    febio_type = "elastic-solid"

    internal = True

    def __init__(self, name, element_type, bound, nodes, connectivity, entry, n_entries, device):
        geometry, order, points = ELEMENTS[element_type]
        self.name = name
        self.element_type = element_type
        self.material = bound
        self.device = device
        self.n = n_entries
        self.count = E = len(connectivity)
        self.points = points
        self._force, self._stiffness, self._jacobian, self._stress = _make_forms(bound)
        self.pvec = bound.model.param_vector(wp.float64)
        gvec = bound.model.geometry_vector(wp.float64)
        dev = dict(device=device)

        # warp.fem objects: geometry over the model's nodes, a vector space
        # with one node per vertex, the cells and their Gauss rule.
        self.geometry = geometry(
            wp.array(connectivity.astype(np.int32), dtype=int, **dev),
            wp.array(nodes, dtype=vec3d, **dev),
        )
        self.space = fem.make_polynomial_space(self.geometry, degree=1, dtype=vec3d)
        self.cells = fem.Cells(self.geometry)
        self.quadrature = fem.RegularQuadrature(self.cells, order=order, family=fem.Polynomial.GAUSS_LEGENDRE)
        assert self.quadrature.total_point_count() == E * points
        self.test = fem.make_test(self.space, domain=self.cells)
        self.trial = fem.make_trial(self.space, domain=self.cells)
        self.u = self.space.make_field()
        self.u.dof_values = wp.zeros(len(nodes), dtype=vec3d, requires_grad=True, **dev)

        self.entry = wp.array(entry.astype(np.int32), dtype=wp.int32, **dev)  # node -> entry
        self.geom = wp.array(bound.geometry_array(E), dtype=gvec, **dev)
        self.params = wp.array(
            [self.pvec(*bound.param_values(0.0))],
            dtype=self.pvec,
            requires_grad=True,
            **dev,
        )
        self.f = wp.zeros(len(nodes), dtype=vec3d, requires_grad=True, **dev)
        self.detF = wp.zeros(E * points, dtype=wp.float64, **dev)

        # The stiffness pattern is fixed by the mesh. Integrate once to learn
        # it; later passes refill the same matrix.
        self.K = self._integrate_stiffness(None)
        self.block_count = self.K.nnz_sync()

    # -- warp.fem calls ------------------------------------------------------

    def _values(self):
        return {"params": self.params, "geom": self.geom}

    def _integrate_stiffness(self, output):
        return fem.integrate(
            self._stiffness,
            quadrature=self.quadrature,
            fields={"u": self.u, "du": self.trial, "v": self.test},
            values=self._values(),
            output_dtype=wp.float64,
            output=output,
            bsr_options={"prune_numerical_zeros": False},
        )

    # -- Term ----------------------------------------------------------------

    def bind_blocks(self, rows, cols, vals) -> None:
        super().bind_blocks(rows, cols, vals)
        # (row, column) of every block, once: matrix rows and columns are
        # nodes, the system's are entries.
        nnz = self.block_count
        offsets = self.K.offsets.numpy()[: self.K.nrow + 1]
        node_rows = np.repeat(np.arange(self.K.nrow), np.diff(offsets))
        entry = self.entry.numpy()
        rows.assign(entry[node_rows])
        cols.assign(entry[self.K.columns.numpy()[:nnz]])

    def update(self, state: State) -> int:
        if self.material.time_dependent:
            self.set_parameters(self.material.param_values(state.time))
        wp.launch(
            _gather,
            dim=self.entry.shape[0],
            inputs=[state.u, self.entry],
            outputs=[self.u.dof_values],
            device=self.device,
        )
        # A kernel cannot throw; the worst Jacobian is read on the host.
        fem.interpolate(self._jacobian, at=self.quadrature, fields={"u": self.u}, dest=self.detF)
        det = self.detF.numpy()
        if not np.isfinite(det).all():
            return NON_FINITE
        return OK if (det > 0.0).all() else NEGATIVE_JACOBIAN

    def residual(self, state: State, history: Any, out: wp.array) -> None:
        fem.integrate(
            self._force,
            quadrature=self.quadrature,
            fields={"u": self.u, "v": self.test},
            values=self._values(),
            output=self.f,
        )
        wp.launch(
            _scatter_subtract,
            dim=self.f.shape[0],
            inputs=[self.f, self.entry],
            outputs=[out],
            device=self.device,
        )

    def linearize(self, state: State, history: Any, exact: bool = False) -> BlockTangent:
        # Always the exact derivative of the internal force.
        self._integrate_stiffness(self.K)
        wp.copy(self.vals, self.K.values, count=self.block_count)
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
        sigma = wp.zeros(self.count * self.points, dtype=mat33d, device=self.device)
        fem.interpolate(
            self._stress,
            at=self.quadrature,
            fields={"u": self.u},
            values=self._values(),
            dest=sigma,
        )
        return sigma.numpy().reshape(self.count, self.points, 3, 3)
