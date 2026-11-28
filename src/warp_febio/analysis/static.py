"""Static analysis: time stepping with Newton iterations.

FEBio counterparts
    FEAnalysis::Solve             the step loop
    FETimeStepController          automatic time stepping and must points
    FESolidSolver2::Quasin        the iteration and its convergence tests
(https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEAnalysis.cpp,
 https://github.com/febiosoftware/FEBio/blob/develop/FECore/FETimeStepController.cpp,
 https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FESolidSolver2.cpp)

Same as FEBio
    The residual R = F_external - F_internal; the handling of prescribed
    degrees of freedom in the first iteration (their increment enters the
    right-hand side through the stiffness); the convergence tests with dtol,
    etol, rtol, min_residual and max_refs; the time step controller (growth
    toward dtmax from the iteration count, must points, retries).

Different in this draft, and listed by the coverage report
    Full Newton every iteration (FEBio defaults to BFGS updates between
    reformations) and no line search. The converged solution of a step is the
    same within the tolerances, but the iteration count can differ, and the
    time step controller uses that count. Where the two codes take different
    time steps, their output times differ.

Linear solve
    Element forces and stiffness are evaluated on the device and assembled into
    a warp.sparse BSR matrix. The system is then solved either with one of
    Warp's iterative solvers on the device (warp.optim.linear: cg, bicgstab,
    gmres, with a diagonal preconditioner), or, for checking, with a dense
    direct solve on the host. FEBio uses sparse direct solvers (Pardiso,
    skyline). The choice of solver for FEBio-Warp is open; this is a first test
    of what Warp provides.
"""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.optim.linear
import warp.sparse

from ..compiled import CompiledModel, State
from ..elements.brick8 import (
    NEGATIVE_JACOBIAN,
    OK,
    scatter_internal_force,
    stiffness_triplets,
)
from ..results import Results, StateResult

LINEAR_SOLVERS = ("cg", "bicgstab", "gmres", "direct")


class SolveError(RuntimeError):
    """A time step failed (FEBio: negative Jacobian, no convergence, ...)."""


# ---------------------------------------------------------------------------
# residual and tangent on the device


def evaluate(cm: CompiledModel, state: State, tangent: bool = True):
    """Residual, and the unconstrained tangent as (rows, cols, values) triplets.

    All arrays are on the device. The tangent of every element is computed in
    the same kernel as its force; `tangent=False` only skips gathering it.
    """
    device = cm.device
    residual = wp.zeros(cm.ndof, dtype=wp.float64, device=device)
    total = sum(d.count * 24 * 24 for d in cm.domains) if tangent else 0
    rows = wp.zeros(total, dtype=wp.int32, device=device)
    cols = wp.zeros(total, dtype=wp.int32, device=device)
    vals = wp.zeros(total, dtype=wp.float64, device=device)
    offset = 0
    for d in cm.domains:
        d.update_parameters(state.time)
        wp.launch(
            d.kernel,
            dim=d.count,
            inputs=[d.X, d.dofmap, state.u, cm.dN, cm.gw, d.params, d.geom],
            outputs=[d.fe, d.ke, d.sigma, d.status],
            device=device,
        )
        # A kernel cannot throw: read the worst status and raise here, where
        # FEBio would throw NegativeJacobian.
        worst = int(d.status.numpy().max()) if d.count else OK
        if worst != OK:
            kind = "negative Jacobian" if worst == NEGATIVE_JACOBIAN else "non-finite stress"
            raise SolveError(f"{kind} in part {d.name!r}")
        wp.launch(
            scatter_internal_force,
            dim=(d.count, 24),
            inputs=[d.fe, d.dofmap],
            outputs=[residual],
            device=device,
        )
        if tangent:
            wp.launch(
                stiffness_triplets,
                dim=(d.count, 24, 24),
                inputs=[d.ke, d.dofmap, offset],
                outputs=[rows, cols, vals],
                device=device,
            )
            offset += d.count * 24 * 24
    return residual, (rows, cols, vals)


def tangent_matrix(cm: CompiledModel, triplets):
    """The unconstrained tangent as a warp.sparse BSR matrix (scalar blocks)."""
    rows, cols, vals = triplets
    return warp.sparse.bsr_from_triplets(cm.ndof, cm.ndof, rows, cols, vals, prune_numerical_zeros=False)


@wp.kernel
def _constrain_triplets(
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    vals: wp.array(dtype=wp.float64),
    constrained: wp.array(dtype=wp.int32),
    count: int,
    out_rows: wp.array(dtype=wp.int32),
    out_cols: wp.array(dtype=wp.int32),
    out_vals: wp.array(dtype=wp.float64),
):
    """Entries of the constrained system: element entries with a constrained
    row or column set to zero, followed by a unit diagonal for each constrained
    degree of freedom. The matrix stays symmetric and the constrained unknowns
    solve to zero."""
    k = wp.tid()
    if k < count:
        out_rows[k] = rows[k]
        out_cols[k] = cols[k]
        if constrained[rows[k]] == 0 and constrained[cols[k]] == 0:
            out_vals[k] = vals[k]
        else:
            out_vals[k] = wp.float64(0.0)
    else:
        i = k - count
        out_rows[k] = i
        out_cols[k] = i
        if constrained[i] != 0:
            out_vals[k] = wp.float64(1.0)
        else:
            out_vals[k] = wp.float64(0.0)


@wp.kernel
def _mask(x: wp.array(dtype=wp.float64), constrained: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if constrained[i] != 0:
        x[i] = wp.float64(0.0)


def constrained_matrix(cm: CompiledModel, triplets, constrained):
    rows, cols, vals = triplets
    n = rows.shape[0]
    out = [wp.zeros(n + cm.ndof, dtype=t, device=cm.device) for t in (wp.int32, wp.int32, wp.float64)]
    wp.launch(
        _constrain_triplets,
        dim=n + cm.ndof,
        inputs=[rows, cols, vals, constrained, n],
        outputs=out,
        device=cm.device,
    )
    return warp.sparse.bsr_from_triplets(cm.ndof, cm.ndof, *out, prune_numerical_zeros=False)


def dense(A) -> np.ndarray:
    """Host dense copy of a scalar BSR matrix. For checks and small models."""
    n = A.nrow
    nnz = A.nnz_sync()
    offsets = A.offsets.numpy()[: n + 1]
    cols = A.columns.numpy()[:nnz]
    vals = A.values.numpy()[:nnz]
    K = np.zeros((n, A.ncol))
    np.add.at(K, (np.repeat(np.arange(n), np.diff(offsets)), cols), vals)
    return K


def solve_linear(A, b, method: str, tol: float = 1e-13):
    """Solve A x = b on the device with a Warp solver, or densely on the host.

    Returns (x, iterations). The iterative solvers stop at a relative residual
    of `tol`; SolveError is raised if they do not reach it.
    """
    n = b.shape[0]
    if method == "direct":
        x = np.linalg.solve(dense(A), b.numpy())
        return wp.array(x, dtype=wp.float64, device=b.device), 1
    solver = {
        "cg": warp.optim.linear.cg,
        "bicgstab": warp.optim.linear.bicgstab,
        "gmres": warp.optim.linear.gmres,
    }[method]
    x = wp.zeros(n, dtype=wp.float64, device=b.device)
    bnorm = float(np.linalg.norm(b.numpy()))
    if bnorm == 0.0:
        return x, 0
    M = warp.optim.linear.preconditioner(A, "diag")
    maxiter = max(200, 20 * n)
    iterations, error, atol = solver(A, b, x, tol=tol, maxiter=maxiter, M=M, check_every=1, use_cuda_graph=False)
    if not np.isfinite(error) or error > max(atol, 10 * tol * bnorm):
        raise SolveError(f"linear solver {method} stopped at residual {error:.3e} after {iterations} iterations")
    return x, int(iterations)


# ---------------------------------------------------------------------------
# output


def collect(cm: CompiledModel, state: State, iterations=0) -> StateResult:
    """Copy one state to the host in the form the writers use."""
    u = state.u.numpy()
    N = len(cm.model.mesh.nodes)
    disp = u[cm.dof_index[:, :3]]
    shell = None
    if (cm.dof_index[:, 3:] >= 0).any():
        shell = np.zeros((N, 3))
        has = cm.dof_index[:, 3] >= 0
        shell[has] = u[cm.dof_index[has, 3:]]
    out = StateResult(
        time=state.time,
        displacement=disp,
        shell_displacement=shell,
        iterations=iterations,
    )
    for d in cm.domains:
        s = d.sigma.numpy()  # (E, Q, 3, 3)
        mean = s.mean(axis=1)
        out.stress_points[d.name] = s
        out.stress[d.name] = np.stack(
            [
                mean[:, 0, 0],
                mean[:, 1, 1],
                mean[:, 2, 2],
                mean[:, 0, 1],
                mean[:, 1, 2],
                mean[:, 0, 2],
            ],
            axis=1,
        )
    return out


# ---------------------------------------------------------------------------
# time step controller


class TimeStepController:
    """FETimeStepController, for one step. See the FEBio source linked above."""

    def __init__(self, record, model, dt0, tend):
        get = record.get
        self.max_retries = int(get("max_retries", 5))
        self.opt_iter = int(get("opt_iter", 11))
        self.dtmin = float(get("dtmin", 0.0))
        self.dtmax = float(get("dtmax", 0.1))
        self.aggressiveness = int(get("aggressiveness", 0))
        self.cutback = float(get("cutback", 0.5))
        self.dtforce = bool(get("dtforce", 0))
        self.tend = tend
        self.dtp = dt0
        self.retries = 0
        self.ddt = 0.0
        self.must_curve = None
        self.must_points = []
        self.next_must = -1
        self.at_must = False
        dtmax = record.child("dtmax")
        if dtmax is not None and "lc" in dtmax.attrs:
            # A load curve on dtmax gives the must points and the step limit.
            self.must_curve = model.load_controller(int(dtmax.attrs["lc"]))
            self.must_points = [float(t) for t in self.must_curve.points[:, 0]]

    def auto(self, niter, time, dt):
        dtn = max(self.dtp, self.dtmin)
        dtmax = self.must_curve.value(time) if self.must_curve is not None else self.dtmax
        if self.dtforce:
            dtn = dtmax
        elif niter > 0:
            scale = np.sqrt(self.opt_iter / niter)
            if scale >= 1:
                dtn = dtn + (dtmax - dtn) * min(0.20, scale - 1)
                dtn = min(dtn, 5.0 * self.dtp)
                if dtmax > 0:
                    dtn = min(dtn, dtmax)
            else:
                dtn = dtn - (dtn - self.dtmin) * (1 - scale)
                if self.dtmin > 0:
                    dtn = max(dtn, self.dtmin)
                if dtmax > 0:
                    dtn = min(dtn, dtmax)
        else:
            if self.dtmin > 0:
                dtn = max(dtn, self.dtmin)
            if dtmax > 0:
                dtn = min(dtn, dtmax)
        self.dtp = dtn
        if self.must_points:
            dtn = self._check_must_points(time, dtn)
        if time + dtn > self.tend:
            dtn = self.tend - time
        return dtn

    def _check_must_points(self, t, dt):
        eps = self.tend * 1e-12
        self.at_must = False
        points = self.must_points
        if self.next_must >= len(points):
            return dt
        if self.next_must < 0:
            self.next_must = 0
            while self.next_must < len(points) and points[self.next_must] < t + eps:
                self.next_must += 1
                if self.next_must >= len(points):
                    return dt
        tmust = points[self.next_must]
        if tmust < t + dt + eps:
            dt = tmust - t
            self.at_must = True
            self.next_must += 1
        return dt

    def retry(self, dt):
        if self.retries == 0:
            self.ddt = dt / (self.max_retries + 1)
        dtn = dt - self.ddt if self.aggressiveness == 0 else dt * self.cutback
        self.retries += 1
        if self.at_must:
            self.next_must -= 1
            self.at_must = False
        self.dtp = dtn
        return dtn


# ---------------------------------------------------------------------------
# the analysis


def solve_step(cm, u0, time, settings, linear_solver, say):
    """One time step from the converged vector u0 to `time`.

    Returns (u, iterations). Raises SolveError if the step fails.
    """
    dtol, etol, rtol, rmin, max_refs = settings
    device = cm.device
    constrained = cm.constrained_device
    u = u0.copy()
    pdofs, pvals = cm.prescribed_values(time)
    # Increment of the prescribed degrees of freedom over this step.
    du_p = np.zeros(cm.ndof)
    du_p[pdofs] = pvals - u[pdofs]
    Ui = np.zeros(cm.ndof)  # total increment of the free equations in this step
    free = cm.free
    for it in range(max_refs):
        residual, triplets = evaluate(cm, State(time, wp.array(u, dtype=wp.float64, device=device)))
        rhs = residual
        if it == 0 and len(pdofs):
            # FEBio moves K * (prescribed increment) to the right-hand side in
            # the first iteration, then applies the increment.
            K = tangent_matrix(cm, triplets)
            warp.sparse.bsr_mv(
                K,
                wp.array(du_p, dtype=wp.float64, device=device),
                rhs,
                alpha=-1.0,
                beta=1.0,
            )
        wp.launch(_mask, dim=cm.ndof, inputs=[rhs, constrained], device=device)
        A = constrained_matrix(cm, triplets, constrained)
        x, linear_iterations = solve_linear(A, rhs, linear_solver)
        ui = x.numpy()
        r0 = rhs.numpy()
        if it == 0:
            normRi, normEi = abs(r0 @ r0), abs(ui @ r0)
            u[pdofs] = pvals
        u[free] += ui[free]
        Ui[free] += ui[free]
        R1 = evaluate(cm, State(time, wp.array(u, dtype=wp.float64, device=device)), tangent=False)[0].numpy()[free]
        uf = ui[free]
        normR1, normu, normU, normE1 = R1 @ R1, uf @ uf, Ui @ Ui, abs(uf @ R1)
        if not (np.isfinite(normR1) and np.isfinite(normu)):
            raise SolveError("non-finite residual or solution")
        ok = True
        if rtol > 0 and normR1 > rtol * normRi:
            ok = False
        if dtol > 0 and normu > dtol * dtol * normU:
            ok = False
        if etol > 0 and normE1 > etol * normEi:
            ok = False
        if not ok and normR1 < rmin:
            ok = True
        say(
            f"  time {time:.6g} iteration {it + 1}: R {normR1:.3e} E {normE1:.3e} u {normu:.3e} linear {linear_iterations}"
        )
        if ok:
            return u, it + 1
    raise SolveError(f"no convergence at time {time:g} in {max_refs} reformations")


def run(cm: CompiledModel, log=None, linear_solver: str = "cg") -> Results:
    """Run the model's static step. Returns the states FEBio would plot.

    linear_solver: "cg", "bicgstab" or "gmres" (Warp, on the device), or
    "direct" (dense, on the host; for checks and small models).
    """
    if linear_solver not in LINEAR_SOLVERS:
        raise ValueError(f"linear_solver must be one of {LINEAR_SOLVERS}")
    say = log or (lambda *_: None)
    model = cm.model
    control = model.steps[0].control
    solver = control.child("solver")
    get = (lambda tag, default: solver.get(tag, default)) if solver is not None else (lambda tag, default: default)
    # Defaults from the FENewtonSolver and FESolidSolver2 constructors.
    settings = (
        float(get("dtol", 0.001)),
        float(get("etol", 0.01)),
        float(get("rtol", 0.0)),
        float(get("min_residual", 1e-20)),
        int(get("max_refs", 15)),
    )
    nsteps = int(control.get("time_steps", 10))
    dt0 = float(control.get("step_size", 0.1))
    tend = nsteps * dt0
    eps = tend * 1e-12
    must_only = str(control.get("plot_level", "PLOT_MAJOR_ITRS")) == "PLOT_MUST_POINTS"
    stepper = control.child("time_stepper")
    controller = TimeStepController(stepper, model, dt0, tend) if stepper is not None else None

    state = cm.state()
    results = Results()
    # The reference state: evaluate once so the stresses (zero) are defined.
    # FEBio writes the initial state at time 0 (seen in its output for the
    # reference inputs of this draft).
    evaluate(cm, state, tangent=False)
    results.states.append(collect(cm, state))

    u = state.u.numpy()
    time, dt = 0.0, dt0
    if controller is not None:
        dt = controller.auto(0, time, dt)
    while tend - time > eps:
        new_time = min(time + dt, tend)
        try:
            u_new, iterations = solve_step(cm, u, new_time, settings, linear_solver, say)
        except SolveError as err:
            if controller is not None and controller.retries < controller.max_retries:
                dt = controller.retry(dt)
                say(f"step to time {new_time:g} failed ({err}); retrying with dt {dt:g}")
                continue
            raise
        at_must = controller.at_must if controller is not None else True
        u, time = u_new, new_time
        say(f"converged at time {time:.6g} in {iterations} iterations")
        if not must_only or at_must:
            state = State(time, wp.array(u, dtype=wp.float64, device=cm.device))
            evaluate(cm, state, tangent=False)  # stresses at the converged state
            results.states.append(collect(cm, state, iterations=iterations))
        if controller is not None:
            controller.retries = 0
            if time + eps < tend:
                # FEBio passes m_niter, the number of Newton iterations of
                # the converged step (FEAnalysis.cpp, AutoTimeStep call).
                dt = controller.auto(iterations, time, dt)
    return results
