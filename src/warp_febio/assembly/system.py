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

What a term must keep to for differentiation
    `update` and `residual` are launched under a `wp.Tape` by the adjoint
    (sensitivity/adjoint.py). Kernels they launch must therefore be written
    for reverse mode: no array element written twice, fixed loop bounds.
    `linearize` is never taped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import warp as wp
import warp.sparse

from ..discretization.fields import FieldTable
from ..materials.status import NEGATIVE_JACOBIAN, OK

vec3d = wp.vec3d
mat33d = wp.mat33d


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


class Triplets:
    """Blocks of the global matrix as (row entry, column entry, 3x3 value)."""

    def __init__(self):
        self.rows, self.cols, self.vals = [], [], []

    def add(self, rows: wp.array, cols: wp.array, vals: wp.array):
        self.rows.append(rows)
        self.cols.append(cols)
        self.vals.append(vals)

    def arrays(self, device):
        if not self.rows:
            z = lambda t: wp.zeros(0, dtype=t, device=device)
            return z(wp.int32), z(wp.int32), z(mat33d)
        if len(self.rows) == 1:
            return self.rows[0], self.cols[0], self.vals[0]
        cat = lambda parts, t: wp.array(
            np.concatenate([p.numpy() for p in parts]), dtype=t, device=device
        )
        # Template: concatenation through the host. A device-side copy into
        # one preallocated buffer per term replaces this.
        return cat(self.rows, wp.int32), cat(self.cols, wp.int32), cat(self.vals, mat33d)


class BlockTangent:
    """A term's tangent as 3x3 blocks: the TangentOperator of contracts.py,
    with `assemble`. Contact's ContactTangent has the same three calls."""

    def __init__(self, n: int, rows: wp.array, cols: wp.array, vals: wp.array):
        self.n, self.rows, self.cols, self.vals = n, rows, cols, vals
        self._matrix = None

    def assemble(self, dest: Triplets) -> None:
        dest.add(self.rows, self.cols, self.vals)

    def matrix(self):
        if self._matrix is None:
            self._matrix = warp.sparse.bsr_from_triplets(
                self.n, self.n, self.rows, self.cols, self.vals,
                prune_numerical_zeros=False,
            )
        return self._matrix

    def apply(self, vector: wp.array, out: wp.array) -> None:
        warp.sparse.bsr_mv(self.matrix(), vector, out, alpha=1.0, beta=0.0)

    def apply_transpose(self, vector: wp.array, out: wp.array) -> None:
        warp.sparse.bsr_mv(
            warp.sparse.bsr_transposed(self.matrix()), vector, out, alpha=1.0, beta=0.0
        )


class Term:
    """Anything that adds to the residual and the tangent.

    The calls are those of `contracts.PhysicsTerm` (residual, linearize,
    commit, rollback) with `update` in front. Sign: FEBio's, residual =
    external - internal, so a domain subtracts its internal force and a load
    adds its load vector. `linearize` returns the term's contribution to
    K = -dR/du.
    """

    name: str = ""

    def update(self, state: State) -> int:
        """Bring the term's own data to `state` (point data for a domain,
        pairing for contact, current facet geometry for a follower load).
        Returns a status code from materials/status.py."""
        return OK

    def residual(self, state: State, history: Any, out: wp.array) -> None:
        raise NotImplementedError

    def linearize(self, state: State, history: Any) -> Optional[BlockTangent]:
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
    out_rows: wp.array(dtype=wp.int32),
    out_cols: wp.array(dtype=wp.int32),
    out_vals: wp.array(dtype=mat33d),
):
    """Blocks of the constrained system: each block with its constrained rows
    and columns zeroed, followed by one diagonal block per entry with a unit
    entry for each constrained component. Constrained unknowns solve to zero
    and a symmetric matrix stays symmetric."""
    k = wp.tid()
    if k < count:
        out_rows[k] = rows[k]
        out_cols[k] = cols[k]
        out_vals[k] = wp.diag(free[rows[k]]) * vals[k] * wp.diag(free[cols[k]])
    else:
        i = k - count
        out_rows[k] = i
        out_cols[k] = i
        out_vals[k] = wp.diag(vec3d(1.0, 1.0, 1.0) - free[i])


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
        self.n = fields.count
        self.free = wp.array(fields.free, dtype=vec3d, device=device)

    def state(self, requires_grad: bool = False) -> State:
        """A zero state. The call mirrors Newton's `Model.state(requires_grad=...)`."""
        u = wp.zeros(self.n, dtype=vec3d, device=self.device, requires_grad=requires_grad)
        return State(time=0.0, u=u)

    # -- the three passes of an iteration -----------------------------------

    def update(self, state: State) -> None:
        """Every term brings its data to `state`. The material is called here,
        once per integration point, and nowhere else in an iteration."""
        for term in self.terms:
            status = term.update(state)
            if status != OK:
                kind = (
                    "negative Jacobian" if status == NEGATIVE_JACOBIAN else "non-finite stress"
                )
                raise SolveError(f"{kind} in {term.name!r}")

    def residual(self, state: State, requires_grad: bool = False) -> wp.array:
        """R = F_external - F_internal at the last `update`."""
        R = wp.zeros(self.n, dtype=vec3d, device=self.device, requires_grad=requires_grad)
        for term in self.terms:
            term.residual(state, None, R)
        return R

    def tangent(self, state: State):
        """K = -dR/du at the last `update`, unconstrained, as triplets."""
        triplets = Triplets()
        for term in self.terms:
            op = term.linearize(state, None)
            if op is not None:
                op.assemble(triplets)
        return triplets.arrays(self.device)

    # -- matrices ------------------------------------------------------------

    def matrix(self, triplets):
        rows, cols, vals = triplets
        return warp.sparse.bsr_from_triplets(
            self.n, self.n, rows, cols, vals, prune_numerical_zeros=False
        )

    def constrained_matrix(self, triplets):
        rows, cols, vals = triplets
        m = rows.shape[0]
        out = [
            wp.zeros(m + self.n, dtype=t, device=self.device)
            for t in (wp.int32, wp.int32, mat33d)
        ]
        wp.launch(
            _constrain_blocks,
            dim=m + self.n,
            inputs=[rows, cols, vals, self.free, m],
            outputs=out,
            device=self.device,
        )
        return warp.sparse.bsr_from_triplets(
            self.n, self.n, *out, prune_numerical_zeros=False
        )

    def mask(self, x: wp.array) -> None:
        wp.launch(_mask_vector, dim=self.n, inputs=[x, self.free], device=self.device)

    def parameters(self) -> dict:
        """All differentiable parameter arrays, keyed "term name.parameter set"."""
        out = {}
        for term in self.terms:
            for key, array in term.parameters().items():
                out[f"{term.name}.{key}"] = array
        return out
