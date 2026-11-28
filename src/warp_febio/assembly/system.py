"""Terms, and the system that sums them into a residual and a block matrix.

TEMPLATE (run path draft, November 2026). Structure for discussion, not final.

FEBio counterpart
    FESolidSolver2::Residual loops over the domains (internal forces), the
    model loads, and the contact interfaces, and FESolidSolver2::StiffnessMatrix
    makes the same passes into FEGlobalMatrix.
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FESolidSolver2.cpp

Same as FEBio
    The residual is F_external - F_internal. The tangent K is the derivative
    of the internal force minus that of the external force, K = -dR/du.

Different, and why
    The solver does not know what kinds of things contribute. Everything is a
    `Term` (the calls of `contracts.PhysicsTerm`, plus `update`), so a domain,
    a pressure load, a set of springs and a contact interface are summed by
    the same loop. Terms write into one device residual of 3-vectors and
    append 3x3 blocks to one triplet list; the matrix is built once from
    those on the device (full assembly).

Built for the device
    The block list has a fixed size and fixed (row, column) pairs for a given
    mesh, so the system owns one buffer and every term writes its blocks
    into its own slice of it; nothing is gathered on the host. The sparsity
    is analysed once: warp.sparse builds a map from triplets to matrix
    blocks (bsr_set_from_triplets, build_mapping) and each later iteration
    only refills values through it (bsr_set_from_mapped_values).

What a term must keep to for differentiation
    `update` and `residual` are launched under a `wp.Tape` by the adjoint
    (sensitivity/adjoint.py). Kernels they launch must therefore be written
    for reverse mode: no array element written twice, fixed loop bounds.
    `linearize` is never taped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp
import warp.sparse

from ..discretization.fields import FieldTable
from ..materials.status import NEGATIVE_JACOBIAN, OK

vec3d = wp.vec3d
mat33d = wp.mat33d


#: Default of System.damping_tangent. See there, and decision D4 of
#: docs/design/dynamics.md for what was measured.
DAMPING_TANGENT = "exact"
DAMPING_TANGENTS = ("febio", "consistent", "exact")


class SolveError(RuntimeError):
    """A step failed where FEBio would throw (negative Jacobian, no convergence)."""


@dataclass
class State:
    """The solution at one time, on the device.

    FEBio keeps this on nodes and material points. Here it is separate from
    the model, so several states can exist for one compiled system (output
    history; forward and adjoint solves). `u` holds one 3-vector per entry of
    the field table.
    """

    time: float
    u: wp.array  # (n_entries,) vec3d
    # Dynamic steps only (analysis/dynamics.py); None in a static step.
    v: wp.array | None = None  # velocity
    a: wp.array | None = None  # acceleration
    a_mass: wp.array | None = None  # alpha_m a + (1 - alpha_m) a_n, what the inertial force acts on
    dt: float = 0.0  # the step that led to this state


class BlockTangent:
    """A term's tangent as 3x3 blocks: the TangentOperator of contracts.py.
    The arrays are the term's slice of the system's block buffer, so the
    blocks are already where the global matrix is built from."""

    def __init__(self, n: int, rows: wp.array, cols: wp.array, vals: wp.array):
        self.n, self.rows, self.cols, self.vals = n, rows, cols, vals
        self._matrix = None

    def matrix(self):
        if self._matrix is None:
            self._matrix = warp.sparse.bsr_from_triplets(
                self.n,
                self.n,
                self.rows,
                self.cols,
                self.vals,
                prune_numerical_zeros=False,
            )
        return self._matrix

    def apply(self, vector: wp.array, out: wp.array) -> None:
        warp.sparse.bsr_mv(self.matrix(), vector, out, alpha=1.0, beta=0.0)

    def apply_transpose(self, vector: wp.array, out: wp.array) -> None:
        warp.sparse.bsr_mv(warp.sparse.bsr_transposed(self.matrix()), vector, out, alpha=1.0, beta=0.0)


class Term:
    """Anything that adds to the residual and the tangent.

    The calls are those of `contracts.PhysicsTerm` (residual, linearize,
    commit, rollback) with `update` in front. Sign: FEBio's, residual =
    external - internal, so a domain subtracts its internal force and a load
    adds its load vector. `linearize` returns the term's contribution to
    K = -dR/du.
    """

    name: str = ""
    #: True for domains: the term's residual is (minus) an internal force.
    #: FEBio reports nodal reaction forces from the internal forces alone.
    internal: bool = False
    #: Sign with which FEBio's domain of this kind adds the stiffness of a
    #: body force to K (BodyForceStiffness). +1 for solids and trusses. The
    #: shell domain adds it with the opposite sign (see BrickDomain).
    body_force_stiffness_sign: float = 1.0
    #: Number of 3x3 blocks the term contributes to the tangent. Fixed for
    #: the life of the system, with fixed (row, column) pairs.
    block_count: int = 0

    def bind_blocks(self, rows: wp.array, cols: wp.array, vals: wp.array) -> None:
        """Receive the term's slice of the system's block buffer. `linearize`
        writes its blocks there."""
        self.rows, self.cols, self.vals = rows, cols, vals

    def update(self, state: State) -> int:
        """Bring the term's own data to `state` (point data for a domain,
        pairing for contact, current facet geometry for a follower load).
        Returns a status code from materials/status.py."""
        return OK

    def residual(self, state: State, history: Any, out: wp.array) -> None:
        raise NotImplementedError

    def linearize(self, state: State, history: Any, exact: bool = False) -> BlockTangent | None:
        """The term's part of K. FEBio lets some terms give Newton a tangent
        that is not the derivative of their residual (the symmetrized
        pressure stiffness; `tangent_scale` on a material). With exact=True a
        term must return the true derivative, which is what the adjoint
        needs. A term whose tangent is always exact ignores the flag."""
        return None

    def mass_coefficients(self) -> np.ndarray | None:
        """Consistent mass of the term, one number per stiffness block in the
        order of its blocks: the block of a node pair is that number times
        the identity (FEBio: MassMatrix of the domain). None for a term
        without mass (a load, springs: FEDiscreteElasticDomain adds none)."""
        return None

    def commit(self, trial_history: Any) -> None:
        """Accept trial history on a converged step. No history yet."""

    def rollback(self) -> None:
        """Discard trial history on a rejected step. No history yet."""

    def parameters(self) -> dict:
        """Device arrays that hold differentiable parameters, by name."""
        return {}


# ---------------------------------------------------------------------------
# constraints: kept in the system and masked


@wp.kernel
def _mask_vector(x: wp.array(dtype=vec3d), free: wp.array(dtype=vec3d)):
    i = wp.tid()
    x[i] = wp.cw_mul(x[i], free[i])


@wp.kernel
def _constrain_blocks(
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    vals: wp.array(dtype=mat33d),
    free: wp.array(dtype=vec3d),
    count: int,
    out_vals: wp.array(dtype=mat33d),
):
    """Values of the constrained system: each term block with its constrained
    rows and columns zeroed, followed by one diagonal block per entry with a
    unit entry for each constrained component. Constrained unknowns solve to
    zero and a symmetric matrix stays symmetric."""
    k = wp.tid()
    if k < count:
        out_vals[k] = wp.diag(free[rows[k]]) * vals[k] * wp.diag(free[cols[k]])
    else:
        out_vals[k] = wp.diag(vec3d(1.0, 1.0, 1.0) - free[k - count])


@wp.kernel
def _apply_scalar_blocks(
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    coefficient: wp.array(dtype=wp.float64),
    x: wp.array(dtype=vec3d),
    alpha: wp.float64,
    y: wp.array(dtype=vec3d),
):
    """y += alpha M x for blocks that are a number times the identity (mass,
    mass damping). Reads x and adds to y only, so a tape passes through it."""
    k = wp.tid()
    c = coefficient[k]
    if c != wp.float64(0.0):
        wp.atomic_add(y, rows[k], (alpha * c) * x[cols[k]])


@wp.kernel
def _add_scalar_blocks(
    mass: wp.array(dtype=wp.float64),
    damping: wp.array(dtype=wp.float64),
    mass_scale: wp.float64,
    damping_scale: wp.float64,
    vals: wp.array(dtype=mat33d),
):
    """vals += (mass_scale mass + damping_scale damping) I, block by block."""
    k = wp.tid()
    c = mass_scale * mass[k] + damping_scale * damping[k]
    if c != wp.float64(0.0):
        vals[k] = vals[k] + wp.diag(vec3d(c, c, c))


@wp.kernel
def _scalar_to_blocks(coefficient: wp.array(dtype=wp.float64), vals: wp.array(dtype=mat33d)):
    k = wp.tid()
    c = coefficient[k]
    vals[k] = wp.diag(vec3d(c, c, c))


@wp.kernel
def _apply_blocks(
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    vals: wp.array(dtype=mat33d),
    x: wp.array(dtype=vec3d),
    alpha: wp.float64,
    y: wp.array(dtype=vec3d),
):
    """y += alpha K x, straight from the blocks."""
    k = wp.tid()
    wp.atomic_add(y, rows[k], alpha * (vals[k] * x[cols[k]]))


# ---------------------------------------------------------------------------


class System:
    """A compiled model as the solver sees it: a field table and a term list.

    FEBio counterpart: FEModel after Init, as FESolidSolver2 uses it. Built by
    analysis/steps.py; holds nothing about valves, patches or devices.
    """

    def __init__(self, model, fields: FieldTable, terms: list, device):
        self.model = model
        self.fields = fields
        self.terms = terms
        self.device = device
        self.n = n = fields.count
        self.free = wp.array(fields.free, dtype=vec3d, device=device)

        # One block buffer for the whole system: a slice per term, then one
        # diagonal block per entry for the constraints.
        self.block_count = total = sum(t.block_count for t in terms)
        rows = np.zeros(total + n, dtype=np.int32)
        rows[total:] = np.arange(n)
        self.rows = wp.array(rows, dtype=wp.int32, device=device)
        self.cols = wp.array(rows, dtype=wp.int32, device=device)
        self.vals = wp.zeros(total + n, dtype=mat33d, device=device)
        self.constrained_vals = wp.zeros(total + n, dtype=mat33d, device=device)
        offset = 0
        for term in terms:
            end = offset + term.block_count
            term.bind_blocks(self.rows[offset:end], self.cols[offset:end], self.vals[offset:end])
            offset = end
        self._A = warp.sparse.bsr_zeros(n, n, mat33d, device=device)
        self._map = None  # triplet-to-block map, built at the first assembly
        self._blocks_placed = False  # the terms have written their (row, column) pairs
        # Dynamics (analysis/dynamics.py): None for a static model.
        self.time_integration = None
        #: The stiffness Newton uses for mass damping (decision D4):
        #:   "febio"       C / dt per unit mass, with FEBio's sign per domain
        #:                 (added for solids and trusses, subtracted for shells)
        #:   "consistent"  C / dt per unit mass, added for every domain
        #:   "exact"       the derivative of the damping force, C gamma / (beta dt)
        #: The adjoint always uses the derivative (exact=True).
        self.damping_tangent = DAMPING_TANGENT
        self.mass = None  # (total,) consistent mass per block
        self.damping = None  # (total,) C times the mass, on damped parts
        self.has_damping = False

    def set_dynamics(self, time_integration, damping_by_term: dict | None = None) -> None:
        """Make the system dynamic: collect every term's mass, one number per
        block of the buffer, and the mass damping coefficients.

        The mass couples the node pairs the stiffness blocks couple, so it
        needs no pattern of its own and is computed once (decision D1:
        consistent mass, as FEBio). damping_by_term: {term name: C} for the
        parts under a `mass damping` load; its force is C M v on those parts.
        """
        damping_by_term = damping_by_term or {}
        mass = np.zeros(self.block_count)
        damping = np.zeros(self.block_count)
        newton = np.zeros(self.block_count)
        offset = 0
        for term in self.terms:
            end = offset + term.block_count
            m = term.mass_coefficients()
            if m is not None:
                mass[offset:end] = m
                damping[offset:end] = damping_by_term.get(term.name, 0.0) * np.asarray(m)
                newton[offset:end] = term.body_force_stiffness_sign * damping[offset:end]
            offset = end
        self.time_integration = time_integration
        self.mass = wp.array(mass, dtype=wp.float64, device=self.device)
        self.damping = wp.array(damping, dtype=wp.float64, device=self.device)
        # The same with FEBio's sign per domain, for the Newton tangent.
        self.damping_newton = wp.array(newton, dtype=wp.float64, device=self.device)
        self.has_damping = bool(np.any(damping != 0.0))

    def state(self, requires_grad: bool = False) -> State:
        """A zero state. The call mirrors Newton's `Model.state(requires_grad=...)`."""
        u = wp.zeros(self.n, dtype=vec3d, device=self.device, requires_grad=requires_grad)
        if self.time_integration is None:
            return State(time=0.0, u=u)
        zeros = lambda: wp.zeros(self.n, dtype=vec3d, device=self.device, requires_grad=requires_grad)
        return State(time=0.0, u=u, v=zeros(), a=zeros(), a_mass=zeros())

    # -- the three passes of an iteration -----------------------------------

    def update(self, state: State) -> None:
        """Every term brings its data to `state`. The material is called here,
        once per integration point, and nowhere else in an iteration."""
        for term in self.terms:
            status = term.update(state)
            if status != OK:
                kind = "negative Jacobian" if status == NEGATIVE_JACOBIAN else "non-finite stress"
                raise SolveError(f"{kind} in {term.name!r}")

    def residual(self, state: State, requires_grad: bool = False) -> wp.array:
        """R = F_external - F_internal at the last `update`."""
        R = wp.zeros(self.n, dtype=vec3d, device=self.device, requires_grad=requires_grad)
        for term in self.terms:
            term.residual(state, None, R)
        if self.time_integration is not None and state.a_mass is not None:
            self._dynamic_residual(state, R)
        return R

    def _dynamic_residual(self, state: State, R: wp.array) -> None:
        """R -= M (alpha_m a + (1 - alpha_m) a_n) + C M v: FEBio's inertial
        forces (InertialForces of the elastic domains) and the `mass damping`
        body force, both through the mass blocks."""
        if not self._blocks_placed:
            # The mass goes through the block positions, which the terms
            # write when they first linearize. A run does that before any
            # residual; a caller that starts with a residual gets it here.
            self.tangent(state)
        total = self.block_count
        args = dict(dim=total, outputs=[R], device=self.device)
        rows, cols = self.rows[:total], self.cols[:total]
        wp.launch(_apply_scalar_blocks, inputs=[rows, cols, self.mass, state.a_mass, -1.0], **args)
        if self.has_damping:
            wp.launch(_apply_scalar_blocks, inputs=[rows, cols, self.damping, state.v, -1.0], **args)

    def tangent(self, state: State, exact: bool = False) -> None:
        """Every term writes its blocks of K at the last `update` into the
        system's buffer. With exact=True, K = -dR/du exactly (see
        Term.linearize)."""
        for term in self.terms:
            if term.linearize(state, None, exact) is None and term.block_count:
                term.vals.zero_()
        self._blocks_placed = True
        if self.time_integration is not None and state.dt > 0.0:
            # FESolidSolver2::StiffnessMatrix: the mass scaled by
            # alpha_m / (beta dt^2). Mass damping: FEBio's stiffness is C / dt
            # per unit mass (FEMassDamping::stiffness), which is not the
            # derivative of its force, C gamma / (beta dt), and its shell
            # domain adds it with the opposite sign. Newton follows FEBio in
            # both; exact=True gives the derivative (decision D4).
            ti = self.time_integration
            mass_scale = ti.alpha_m / (ti.beta * state.dt**2)
            choice = "exact" if exact else self.damping_tangent
            damping = self.damping_newton if choice == "febio" else self.damping
            damping_scale = ti.gamma / (ti.beta * state.dt) if choice == "exact" else 1.0 / state.dt
            wp.launch(
                _add_scalar_blocks,
                dim=self.block_count,
                inputs=[self.mass, damping, mass_scale, damping_scale],
                outputs=[self.vals],
                device=self.device,
            )

    # -- matrices ------------------------------------------------------------

    def constrained_matrix(self):
        """The matrix the linear solve uses, from the last `tangent`. The
        sparsity is analysed at the first call; later calls refill values."""
        total = self.block_count
        wp.launch(
            _constrain_blocks,
            dim=total + self.n,
            inputs=[self.rows, self.cols, self.vals, self.free, total],
            outputs=[self.constrained_vals],
            device=self.device,
        )
        if self._map is None:
            self._map = warp.sparse.bsr_set_from_triplets(
                self._A,
                self.rows,
                self.cols,
                self.constrained_vals,
                prune_numerical_zeros=False,
                build_mapping=True,
            )
            self._A.nnz_sync()
        else:
            warp.sparse.bsr_set_from_mapped_values(self._A, self._map, self.constrained_vals)
        return self._A

    def mass_matrix(self):
        """The constrained mass matrix, for the initial accelerations. It
        goes through the block buffer, so a `tangent` must follow before the
        buffer is used as a stiffness again; every Newton iteration starts
        with one."""
        if not self._blocks_placed:
            raise RuntimeError("mass_matrix needs the block positions: call update and tangent once first")
        wp.launch(
            _scalar_to_blocks,
            dim=self.block_count,
            inputs=[self.mass],
            outputs=[self.vals],
            device=self.device,
        )
        return self.constrained_matrix()

    def apply_tangent(self, x: wp.array, y: wp.array, alpha: float = 1.0) -> None:
        """y += alpha K x with the unconstrained K of the last `tangent`."""
        total = self.block_count
        wp.launch(
            _apply_blocks,
            dim=total,
            inputs=[self.rows, self.cols, self.vals, x, alpha],
            outputs=[y],
            device=self.device,
        )

    def matrix(self):
        """The unconstrained K of the last `tangent` as a matrix of its own.
        For checks; the solve does not need it."""
        total = self.block_count
        return warp.sparse.bsr_from_triplets(
            self.n,
            self.n,
            self.rows[:total],
            self.cols[:total],
            self.vals[:total],
            prune_numerical_zeros=False,
        )

    def mask(self, x: wp.array) -> None:
        wp.launch(_mask_vector, dim=self.n, inputs=[x, self.free], device=self.device)

    def parameters(self) -> dict:
        """All differentiable parameter arrays, keyed "term name.parameter set".

        A term that holds several named groups gives full names itself (the
        springs of a model: "discrete set name.material"); see
        `qualified_parameters` on the term."""
        out = {}
        for term in self.terms:
            qualified = getattr(term, "qualified_parameters", False)
            for key, array in term.parameters().items():
                name = key if qualified else f"{term.name}.{key}"
                if name in out:
                    raise ValueError(
                        f"two parameter arrays are named {name!r}; part and discrete set names must differ"
                    )
                out[name] = array
        return out
