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
    # Line search (FELineSearch). FEBio's defaults; lstol = 0 switches it off.
    lstol: float = 0.9
    lsmin: float = 0.01
    lsiter: int = 5
    ls_check_jacobians: bool = False

    @classmethod
    def from_record(cls, solver):
        s = cls()
        if solver is not None:
            s.dtol = float(solver.get("dtol", s.dtol))
            s.etol = float(solver.get("etol", s.etol))
            s.rtol = float(solver.get("rtol", s.rtol))
            s.min_residual = float(solver.get("min_residual", s.min_residual))
            s.max_refs = int(solver.get("max_refs", s.max_refs))
            s.lstol = float(solver.get("lstol", s.lstol))
            s.lsmin = float(solver.get("lsmin", s.lsmin))
            s.lsiter = int(solver.get("lsiter", s.lsiter))
            s.ls_check_jacobians = bool(int(solver.get("ls_check_jacobians", s.ls_check_jacobians)))
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
def _largest_move(du: wp.array(dtype=vec3d), largest: wp.array(dtype=wp.float64)):
    """largest[0] = max over entries of |du|^2, on the device."""
    i = wp.tid()
    wp.atomic_max(largest, 0, wp.length_sq(du[i]))


@wp.kernel
def _set_scaled_sum(base: wp.array(dtype=vec3d), du: wp.array(dtype=vec3d), s: wp.float64, u: wp.array(dtype=vec3d)):
    """u = base + s du: the trial displacement of a line search."""
    i = wp.tid()
    u[i] = base[i] + s * du[i]


@wp.kernel
def _scale(x: wp.array(dtype=vec3d), s: wp.float64):
    i = wp.tid()
    x[i] = s * x[i]


def line_search(settings: NewtonSettings, r0: float, evaluate):
    """FEBio's line search (FELineSearch::DoLineSearch, after Bonet and Wood,
    "Nonlinear Continuum Mechanics for Finite Element Analysis").

    evaluate(s) moves the model to u + s du, updates it and returns
    r(s) = du . R(u + s du); it may raise SolveError where FEBio would throw
    (a negative Jacobian). r0 = du . R(u). A step is accepted when
    |r(s) / r0| <= lstol. Otherwise s is updated from a quadratic through
    r0 and r(s), at most lsiter times; after that the step with the smallest
    |r| seen is taken. A step that falls below lsmin is replaced by 0.5, as
    FEBio does (and NIKE3D before it). With ls_check_jacobians, a step whose
    update fails is halved, at most lsiter times, and lsmin is not applied.

    Returns the step taken; the model is left at it.
    """

    def update(s):
        if not settings.ls_check_jacobians:
            return s, evaluate(s)
        for attempt in range(settings.lsiter + 1):
            try:
                return s, evaluate(s)
            except SolveError:
                if attempt == settings.lsiter:
                    raise
                s *= 0.5

    s, smin, rmin, n = 1.0, 1.0, abs(r0), 0
    while True:
        s, r1 = update(s)
        if s < settings.lsmin and not settings.ls_check_jacobians:
            s = 0.5
            evaluate(s)
            return s
        if n == 0 or abs(r1) < rmin:
            smin, rmin = s, abs(r1)
        r = 0.0 if abs(r1) < 1e-17 else abs(r1 / r0)
        if r <= settings.lstol:
            return s
        a = r0 / r1
        A = 1.0 + a * (s - 1.0)
        B = a * s * s
        D = B * B - 4.0 * A * B
        if D >= 0.0:
            s = (B + np.sqrt(D)) / (2.0 * A)
            if s < 0.0:
                s = (B - np.sqrt(D)) / (2.0 * A)
            s = max(float(s), 0.0)
        else:
            s = 0.5 * B / A
        n += 1
        if n >= settings.lsiter:
            evaluate(smin)
            return smin


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


def solve_step(
    system: System,
    state0: State,
    time: float,
    settings: NewtonSettings,
    linear_solver: str = "direct",
    say=None,
):
    """One time step from the converged `state0` to `time`.

    Returns (state, iterations). Raises SolveError if the step fails; the
    caller keeps `state0`, so nothing has to be undone.
    """
    say = say or (lambda *_: None)
    device, n, fields = system.device, system.n, system.fields
    inner = wp.utils.array_inner
    dynamic = system.time_integration is not None and state0.v is not None
    if dynamic:
        # Velocity and acceleration follow the displacement after every
        # change of it (FESolidSolver2::UpdateKinematics), starting with the
        # displacement unchanged (PrepStep).
        from ...analysis.dynamics import new_dynamic_state, update_kinematics

        state = new_dynamic_state(system, state0, time)
        u = state.u
        kinematics = lambda: update_kinematics(system, state, state0)
    else:
        u = wp.clone(state0.u)
        state = State(time, u)
        kinematics = lambda: None

    # Prescribed components: target values at this time, and their increment.
    target = wp.array(fields.prescribed_values(system.model, time), dtype=vec3d, device=device)
    pmask = wp.array((fields.bc == PRESCRIBED).astype(np.float64), dtype=vec3d, device=device)
    du_p = wp.zeros(n, dtype=vec3d, device=device)
    wp.launch(
        _prescribed_increment,
        dim=n,
        inputs=[u, target, pmask],
        outputs=[du_p],
        device=device,
    )
    any_prescribed = bool(fields.prescribed)

    Ui = wp.zeros(n, dtype=vec3d, device=device)  # total increment of this step
    moves = [t.max_move for t in system.terms if hasattr(t, "max_move")]
    max_move = min(moves) if moves else None
    kinematics()
    system.update(state)
    R = system.residual(state)
    normRi = normEi = 0.0
    for it in range(settings.max_refs):
        system.tangent(state)
        rhs = R
        if it == 0 and any_prescribed:
            # FEBio moves K * (prescribed increment) to the right-hand side.
            system.apply_tangent(du_p, rhs, alpha=-1.0)
        system.mask(rhs)
        A = system.constrained_matrix()
        du, linear_iterations = solve(A, rhs, linear_solver)
        system.mask(du)
        if max_move is not None:
            # A rule of the contact potential made a rule of the step: no
            # node moves farther than R_out in one iterate, or a point could
            # pass through a surface unseen. The step is rejected and cut.
            # One number comes back to the host, not the increment.
            moved = wp.zeros(1, dtype=wp.float64, device=device)
            wp.launch(_largest_move, dim=n, inputs=[du], outputs=[moved], device=device)
            largest = float(np.sqrt(moved.numpy()[0]))
            if largest > max_move:
                raise SolveError(f"a node moved {largest:.3g} in one iterate, more than R_out {max_move:g}")
        if it == 0:
            normRi, normEi = inner(rhs, rhs), abs(inner(du, rhs))
            _apply_increment(system, u, du_p)

        # The step along du: FEBio's line search, or the full step with
        # lstol = 0. Each trial moves the model to u + s du and evaluates
        # the residual there, so the model is left at the step taken.
        u_base = wp.clone(u)
        trial = {}

        def evaluate(s, u_base=u_base, du=du, trial=trial):
            wp.launch(_set_scaled_sum, dim=n, inputs=[u_base, du, s], outputs=[u], device=device)
            kinematics()
            system.update(state)
            trial["R"] = system.residual(state)
            trial["R1"] = wp.clone(trial["R"])
            system.mask(trial["R1"])
            return inner(du, trial["R1"])

        if settings.lstol > 0:
            step = line_search(settings, inner(du, rhs), evaluate)
        else:
            step = 1.0
            evaluate(step)
        R, R1 = trial["R"], trial["R1"]
        if step != 1.0:
            wp.launch(_scale, dim=n, inputs=[du, step], device=device)  # du is now the step taken
        _apply_increment(system, Ui, du)
        # FENewtonSolver::CheckConvergence: the solution norm of the scaled
        # increment, the energy norm |ls (ui . R1)|.
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
            f"u {normu:.3e} linear {linear_iterations} step {step:.3g}"
        )
        if ok:
            for term in system.terms:
                term.commit(None)
            return state, it + 1
    for term in system.terms:
        term.rollback()
    raise SolveError(f"no convergence at time {time:g} in {settings.max_refs} reformations")
