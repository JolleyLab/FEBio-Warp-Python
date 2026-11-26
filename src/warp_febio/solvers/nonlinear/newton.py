"""Newton iteration for one time step.

TEMPLATE (run path draft, November 2026). Structure for discussion, not final.

FEBio counterpart
    FENewtonSolver::SolveStep and FESolidSolver2::Quasin: the iteration, the
    handling of prescribed degrees of freedom, and the convergence tests.
    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FENewtonSolver.cpp
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FESolidSolver2.cpp

Same as FEBio
    R = F_external - F_internal. In the first iteration the increment of the
    prescribed components enters the right-hand side through the stiffness
    and is then applied. Convergence on dtol, etol, rtol with min_residual,
    within max_refs reformations.

Different, and why
    Full Newton: the tangent is reformed every iteration (FEBio defaults to
    BFGS updates between reformations), and there is no line search yet. The
    converged solution of a step agrees within the tolerances; iteration
    counts can differ.
    The loop holds the solution, the residual and the matrix on the device.
    What crosses to the host each iteration is a few scalars: the worst
    status code and the norms for the convergence tests.
    Each iterate is evaluated once: `update` and `residual` after the
    increment give the convergence norms, and the next iteration takes its
    tangent from the same point data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
import warp.sparse

from ...assembly.system import SolveError, State, System
from ...discretization.fields import PRESCRIBED, UPDATE_RULE
from ..linear.warp_linear import solve

vec3d = wp.vec3d


@dataclass
class NewtonSettings:
    """Defaults from the FENewtonSolver and FESolidSolver2 constructors."""

    dtol: float = 0.001
    etol: float = 0.01
    rtol: float = 0.0
    min_residual: float = 1e-20
    max_refs: int = 15

    @classmethod
    def from_record(cls, solver):
        s = cls()
        if solver is not None:
            s.dtol = float(solver.get("dtol", s.dtol))
            s.etol = float(solver.get("etol", s.etol))
            s.rtol = float(solver.get("rtol", s.rtol))
            s.min_residual = float(solver.get("min_residual", s.min_residual))
            s.max_refs = int(solver.get("max_refs", s.max_refs))
        return s


@wp.kernel
def _prescribed_increment(
    u: wp.array(dtype=vec3d),
    target: wp.array(dtype=vec3d),
    prescribed: wp.array(dtype=vec3d),
    du: wp.array(dtype=vec3d),
):
    """du = target - u on prescribed components, zero elsewhere."""
    i = wp.tid()
    du[i] = wp.cw_mul(prescribed[i], target[i] - u[i])


@wp.kernel
def _add(u: wp.array(dtype=vec3d), du: wp.array(dtype=vec3d)):
    """The "add" update rule of the field table (displacements)."""
    i = wp.tid()
    u[i] = u[i] + du[i]


def _apply_increment(system: System, u: wp.array, du: wp.array) -> None:
    # Every variable in the table so far adds. A variable with another rule
    # ("rotation": exponential map) gets its own kernel over its entries here.
    assert all(UPDATE_RULE[v] == "add" for v in system.fields.variables)
    wp.launch(_add, dim=system.n, inputs=[u, du], device=system.device)


def solve_step(system: System, state0: State, time: float, settings: NewtonSettings,
               linear_solver: str = "direct", say=None):
    """One time step from the converged `state0` to `time`.

    Returns (state, iterations). Raises SolveError if the step fails; the
    caller keeps `state0`, so nothing has to be undone.
    """
    say = say or (lambda *_: None)
    device, n, fields = system.device, system.n, system.fields
    inner = wp.utils.array_inner
    u = wp.clone(state0.u)
    state = State(time, u)

    # Prescribed components: target values at this time, and their increment.
    target = wp.array(fields.prescribed_values(system.model, time), dtype=vec3d, device=device)
    pmask = wp.array((fields.bc == PRESCRIBED).astype(np.float64), dtype=vec3d, device=device)
    du_p = wp.zeros(n, dtype=vec3d, device=device)
    wp.launch(_prescribed_increment, dim=n, inputs=[u, target, pmask], outputs=[du_p], device=device)
    any_prescribed = bool(fields.prescribed)

    Ui = wp.zeros(n, dtype=vec3d, device=device)  # total increment of this step
    system.update(state)
    R = system.residual(state)
    normRi = normEi = 0.0
    for it in range(settings.max_refs + 1):
        triplets = system.tangent(state)
        rhs = R
        if it == 0 and any_prescribed:
            # FEBio moves K * (prescribed increment) to the right-hand side.
            warp.sparse.bsr_mv(system.matrix(triplets), du_p, rhs, alpha=-1.0, beta=1.0)
        system.mask(rhs)
        A = system.constrained_matrix(triplets)
        du, linear_iterations = solve(A, rhs, linear_solver)
        system.mask(du)
        if it == 0:
            normRi, normEi = inner(rhs, rhs), abs(inner(du, rhs))
            _apply_increment(system, u, du_p)
        _apply_increment(system, u, du)
        _apply_increment(system, Ui, du)

        system.update(state)
        R = system.residual(state)
        R1 = wp.clone(R)
        system.mask(R1)
        normR1, normu, normU = inner(R1, R1), inner(du, du), inner(Ui, Ui)
        normE1 = abs(inner(du, R1))
        if not (np.isfinite(normR1) and np.isfinite(normu)):
            raise SolveError("non-finite residual or solution")
        ok = True
        if settings.rtol > 0 and normR1 > settings.rtol * normRi:
            ok = False
        if settings.dtol > 0 and normu > settings.dtol**2 * normU:
            ok = False
        if settings.etol > 0 and normE1 > settings.etol * normEi:
            ok = False
        if not ok and normR1 < settings.min_residual:
            ok = True
        say(
            f"  time {time:.6g} iteration {it + 1}: R {normR1:.3e} E {normE1:.3e} "
            f"u {normu:.3e} linear {linear_iterations}"
        )
        if ok:
            for term in system.terms:
                term.commit(None)
            return state, it + 1
    for term in system.terms:
        term.rollback()
    raise SolveError(f"no convergence at time {time:g} in {settings.max_refs} reformations")
