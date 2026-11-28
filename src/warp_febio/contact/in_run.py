"""The contact potential as a term of the run path.

TEMPLATE (contact in a run, November 2026). Structure for discussion, not final.
Design note: docs/design/contact_in_run.md.

FEBio counterpart
    FEContactPotential as FESolidSolver2 drives it: Update (surface points,
    search grid, active pairs), ContactForces and ContactStiffness after the
    domains and loads, and BuildMatrixProfile, which adds the active pairs to
    the matrix profile.
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEContactPotential.cpp

Same as FEBio
    The interface owns its two surfaces, its search and its law. The pairs
    are found again at every iterate. The contact force enters the residual
    as an external force (FEBio's sign, as every other term here).

Different, and why
    The contact blocks do not go into the system's block buffer. Their
    pattern follows the pairs, and the buffer's pattern is fixed for the life
    of the system. They form a matrix of their own (`contact_matrix`), which
    the system adds to the structural matrix before the solve
    (System.constrained_matrix). Only that small matrix is rebuilt.

What is connected here, of the four connections the design note lists
    Points from the model's SurfacePair; the interface as a Term; its blocks
    in a matrix of their own; the rule that no node may move farther than
    R_out in one iterate (the step is then cut).

Not yet
    `w_tol` above zero is applied per point pair here and per element pair
    in FEBio (contact/search.py), so results with it are not FEBio's; the
    coverage report says so. Activation by step. The projected tangent as a
    fallback. Contact under the adjoint: the pair search is not taped, and
    `System.matrix`, which the adjoint uses, leaves the contact blocks out.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..assembly.system import State, Term
from ..coverage import IMPLEMENTED, NOT_SUPPORTED
from ..materials.status import OK
from .points import ContactPoints, quad_facet_points, tri_facet_points
from .potential import ContactPotential, from_febio_block

TYPE = "contact potential"
_FACETS = ("quad4", "tri3")


def _surface_pair(record, model):
    name = record.attrs.get("surface_pair")
    pair = next((r for r in model.mesh.other if r.tag == "SurfacePair" and r.attrs.get("name") == name), None)
    if pair is None:
        return None
    return str(pair.get("primary")), str(pair.get("secondary"))


def inspect_contact(record, model):
    """Coverage of one contact record: (status, note)."""
    if record.type != TYPE:
        return NOT_SUPPORTED, ""
    names = _surface_pair(record, model)
    if names is None or any(n not in model.mesh.surfaces for n in names):
        return NOT_SUPPORTED, "surface pair not resolved"
    for n in names:
        if any(t not in _FACETS for t in model.mesh.surfaces[n].facet_types):
            return NOT_SUPPORTED, f"surface {n!r} has facets other than quad4 and tri3"
    if record.get("kc") is None:
        return NOT_SUPPORTED, "no kc"
    if str(record.get("laugon", "PENALTY")).upper() not in ("PENALTY", "0"):
        return NOT_SUPPORTED, "augmented Lagrangian"
    if int(float(record.get("check_intersections", 0))) != 0:
        return NOT_SUPPORTED, "check_intersections"
    note = "pairs found at every iterate; tangent in a matrix of its own"
    if float(record.get("w_tol", 0.0)) > 0.0:
        note += "; w_tol is applied per point pair here and per element pair in FEBio, so results differ from FEBio's"
    return IMPLEMENTED, note


@wp.kernel
def _positions(X: wp.array(dtype=wp.vec3d), u: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    x[i] = X[i] + u[i]


@wp.kernel
def _mask_rows_and_columns(
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    free: wp.array(dtype=wp.vec3d),
    vals: wp.array(dtype=wp.mat33d),
):
    k = wp.tid()
    vals[k] = wp.diag(free[rows[k]]) * vals[k] * wp.diag(free[cols[k]])


class ContactPotentialInRun(Term):
    """One `contact potential` interface of a model, as a Term."""

    internal = False
    block_count = 0  # its blocks are in `contact_matrix`, not in the block buffer
    element_type = "contact"

    def __init__(self, record, model, fields, device):
        self.name = record.name or record.attrs.get("surface_pair", "contact")
        self.device = device
        self.n = fields.count
        primary, secondary = _surface_pair(record, model)
        same = primary == secondary
        params, kwargs, rule = from_febio_block(record, scale=2.0 if same else 1.0)
        self.r_out = float(np.max(params.r_out))
        # Contact points are addressed by entries of the field table: a point
        # moves with the `displacement` entries of its facet's nodes (for a
        # shell, its front face).
        entry = fields.entries_of("displacement")
        X = np.zeros((self.n, 3))
        X[entry] = model.mesh.nodes
        sets = []
        for body, name in enumerate([primary] if same else [primary, secondary]):
            surface = model.mesh.surfaces[name]
            for kind, build, option in (
                ("quad4", quad_facet_points, dict(n_per_edge=rule["quad_n_per_edge"])),
                ("tri3", tri_facet_points, dict(order=rule["tri_order"])),
            ):
                facets = [entry[f] for f, t in zip(surface.facets, surface.facet_types) if t == kind]
                if facets:
                    sets.append(build(np.array(facets, dtype=np.int32), X, body, **option))
        points = ContactPoints.concatenate(sets) if len(sets) > 1 else sets[0]
        # One class of pairs. Two surfaces interact with each other only; one
        # surface named twice interacts with itself.
        pair_class = np.array([[0]]) if same else np.array([[-1, 0], [0, -1]])
        self.potential = ContactPotential(points, params, pair_class, X, self.n, device=device, **kwargs)
        self.X = wp.array(X, dtype=wp.vec3d, device=device)
        self.x = wp.zeros(self.n, dtype=wp.vec3d, device=device)
        self.pairs = 0
        self._K = None

    # -- Term ----------------------------------------------------------------

    def update(self, state: State) -> int:
        """Current positions, then the pair search (FEContactPotential::Update)."""
        wp.launch(_positions, dim=self.n, inputs=[self.X, state.u], outputs=[self.x], device=self.device)
        self.pairs = self.potential.build_pairs(self.x)
        return OK

    def residual(self, state: State, history, out: wp.array) -> None:
        """out += f, the repulsive nodal forces -dE_c/dx: an external force."""
        self.potential.energy_force(self.x, out)

    def linearize(self, state: State, history, exact: bool = False):
        return None  # see contact_matrix

    # -- beyond Term ---------------------------------------------------------

    #: The largest move of a node in one iterate that the interface accepts.
    #: Beyond R_out a point can pass through a surface unseen (contact note).
    @property
    def max_move(self) -> float:
        return self.r_out

    def contact_matrix(self, free: wp.array):
        """K_c = d2E_c/dx2 over the current pairs as a block matrix with the
        constrained rows and columns zeroed, or None without pairs. Call
        after `update`."""
        if self.pairs == 0:
            return None
        self.potential.search.update_points(self.x)
        self._K = K = self.potential.hessian_bsr(dest=self._K)
        nnz = K.nnz_sync()
        if nnz == 0:
            return None
        wp.launch(
            _mask_rows_and_columns,
            dim=nnz,
            inputs=[K.uncompress_rows()[:nnz], K.columns[:nnz], free, K.values[:nnz]],
            device=self.device,
        )
        return K

    def results(self) -> dict:
        """Outputs at the last residual. "contact area" is FEBio's surface
        data of that name (FEContactPotential: the area of the integration
        points with a nonzero traction; the rule of Wu et al. 2022). It is a
        sum of step functions of the state, so it is an output and not an
        objective: its gradient is zero almost everywhere."""
        return {
            "contact area": self.potential.contact_area() if self.pairs else 0.0,
            "pairs": self.pairs,
            "energy": self.potential.energy_value() if self.pairs else 0.0,
        }
