"""Static analysis: fixed time steps with Newton iterations.

FEBio counterparts: FEAnalysis::Solve for the step loop and
FESolidSolver2::Quasin for the iteration and its convergence tests
(https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEAnalysis.cpp,
 https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FESolidSolver2.cpp).

What is the same as FEBio
    The residual R = F_external - F_internal, the handling of prescribed
    degrees of freedom in the first iteration (their increment enters the
    right-hand side through the stiffness), and the three convergence tests
    with FEBio's settings dtol, etol, rtol, min_residual and max_refs.

What this draft does differently, on purpose and reported by the coverage report
    Full Newton every iteration (no BFGS updates), no line search, no automatic
    time stepping, and a dense direct linear solve on the host. Element forces
    and stiffness are evaluated on the device. The linear solver and the sparse
    assembly belong to the assembly and solver design and are placeholders here.
"""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.sparse

from ..compiled import CompiledModel, State
from ..elements.brick8 import (
    NEGATIVE_JACOBIAN,
    OK,
    scatter_internal_force,
    stiffness_triplets,
)
from ..results import Results, StateResult


class SolveError(RuntimeError):
    pass


def evaluate(cm: CompiledModel, state: State):
    """Residual and tangent at a state, both on the device.

    Returns (residual, tangent): a wp.array of length ndof and a
    warp.sparse BSR matrix with scalar blocks.
    """
    device = cm.device
    residual = wp.zeros(cm.ndof, dtype=wp.float64, device=device)
    total = sum(d.count * 24 * 24 for d in cm.domains)
    rows = wp.zeros(total, dtype=wp.int32, device=device)
    cols = wp.zeros(total, dtype=wp.int32, device=device)
    vals = wp.zeros(total, dtype=wp.float64, device=device)
    offset = 0
    for d in cm.domains:
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
            kind = (
                "negative Jacobian"
                if worst == NEGATIVE_JACOBIAN
                else "non-finite stress"
            )
            raise SolveError(f"{kind} in part {d.name!r}")
        wp.launch(
            scatter_internal_force,
            dim=(d.count, 24),
            inputs=[d.fe, d.dofmap],
            outputs=[residual],
            device=device,
        )
        wp.launch(
            stiffness_triplets,
            dim=(d.count, 24, 24),
            inputs=[d.ke, d.dofmap, offset],
            outputs=[rows, cols, vals],
            device=device,
        )
        offset += d.count * 24 * 24
    tangent = warp.sparse.bsr_from_triplets(
        cm.ndof, cm.ndof, rows, cols, vals, prune_numerical_zeros=False
    )
    return residual, tangent


def _dense(A) -> np.ndarray:
    """Host dense copy of a scalar BSR matrix. Draft only: small models."""
    n = A.nrow
    nnz = A.nnz_sync()
    offsets = A.offsets.numpy()[: n + 1]
    cols = A.columns.numpy()[:nnz]
    vals = A.values.numpy()[:nnz]
    K = np.zeros((n, A.ncol))
    rows = np.repeat(np.arange(n), np.diff(offsets))
    np.add.at(K, (rows, cols), vals)
    return K


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


def run(cm: CompiledModel, log=None) -> Results:
    """Run the model's single static step. Returns the states FEBio would plot:
    the reference state at time 0 and one state per converged time step."""
    say = log or (lambda *_: None)
    control = cm.model.steps[0].control
    solver = control.child("solver")
    get = (
        (lambda tag, default: solver.get(tag, default))
        if solver is not None
        else (lambda tag, default: default)
    )
    # Defaults from FENewtonSolver / FESolidSolver2 constructors.
    dtol, etol, rtol = (
        float(get("dtol", 0.001)),
        float(get("etol", 0.01)),
        float(get("rtol", 0.0)),
    )
    rmin = float(get("min_residual", 1e-20))
    max_refs = int(get("max_refs", 15))
    nsteps = int(control.get("time_steps", 10))
    dt = float(control.get("step_size", 0.1))

    state = cm.state()
    results = Results()
    # The reference state: evaluate once so the stresses (zero) are defined.
    evaluate(cm, state)
    # FEBio writes the initial state at time 0 (checked against its output for
    # the reference inputs of this draft).
    results.states.append(collect(cm, state))

    free = cm.free
    for n in range(1, nsteps + 1):
        time = n * dt
        u = state.u.numpy()
        pdofs, pvals = cm.prescribed_values(time)
        # Increment of the prescribed degrees of freedom over this step.
        du_p = np.zeros(cm.ndof)
        du_p[pdofs] = pvals - u[pdofs]
        Ui = np.zeros(len(free))  # total increment of the free equations
        converged = False
        for it in range(max_refs + 1):
            residual, tangent = evaluate(
                cm, State(time, wp.array(u, dtype=wp.float64, device=cm.device))
            )
            R = residual.numpy()
            K = _dense(tangent)
            rhs = R[free].copy()
            if it == 0:
                # FEBio moves K * (prescribed increment) to the right-hand side
                # in the first iteration, then applies the increment.
                rhs -= K[np.ix_(free, np.arange(cm.ndof))] @ du_p
            ui = np.linalg.solve(K[np.ix_(free, free)], rhs)
            if it == 0:
                normRi, normEi = abs(rhs @ rhs), abs(ui @ rhs)
                u[pdofs] = pvals
            u[free] += ui
            Ui += ui
            R1 = evaluate(
                cm, State(time, wp.array(u, dtype=wp.float64, device=cm.device))
            )[0].numpy()[free]
            normR1, normu, normU, normE1 = R1 @ R1, ui @ ui, Ui @ Ui, abs(ui @ R1)
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
                f"step {n} time {time:g} iteration {it + 1}: R {normR1:.3e} E {normE1:.3e} u {normu:.3e}"
            )
            if ok:
                converged = True
                break
        if not converged:
            raise SolveError(
                f"step {n} (time {time:g}) did not converge in {max_refs} reformations"
            )
        state = State(time, wp.array(u, dtype=wp.float64, device=cm.device))
        evaluate(cm, state)  # stresses at the converged state
        results.states.append(collect(cm, state, iterations=it + 1))
    return results
