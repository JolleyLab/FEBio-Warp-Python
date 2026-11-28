"""Implicit dynamics: FEBio's time integration, on the device.

TEMPLATE (dynamics draft, November 2026). Structure for discussion, not final.
Design note: docs/design/dynamics.md.

FEBio counterpart
    FESolidSolver2: Init (the parameter rules), InitAccelerations, PrepStep
    and UpdateKinematics (velocity and acceleration from the displacement),
    ExternalForces (inertial forces) and StiffnessMatrix (the mass matrix
    scaled by alpha_m / (beta dt^2)).
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FESolidSolver2.cpp

Same as FEBio
    The parameters and their rules, the update of velocity and acceleration,
    the consistent mass, the inertial force on alpha_m a + (1 - alpha_m) a_n,
    the scaling of the mass in the stiffness, and the initial accelerations
    from M a = F when a load acts at time zero.

Different, and why
    Velocity and acceleration are arrays of the State, one 3-vector per entry
    of the field table, beside the displacement (decision D2). FEBio keeps
    them on the nodes. A step is then a function from one State to the next,
    which is what an adjoint over steps replays.
    The mass is not assembled as a matrix of its own. A domain's mass couples
    the node pairs its stiffness blocks couple, so the system keeps one
    number per block of its buffer (assembly/system.py) and the mass enters
    the residual and the tangent through the blocks that are already there.

Built for the device
    One kernel over entries updates v and a from u after every increment; no
    host round trip. It reads the previous state and writes each element of
    its outputs once, so a tape can pass through it.

Reserved
    alpha_f below 1 (stress at an interpolated deformation, decision D3):
    refused by the coverage report. Rigid body inertia. Initial velocities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..assembly.system import State, System
from ..solvers.linear.warp_linear import solve

vec3d = wp.vec3d


@dataclass(frozen=True)
class TimeIntegration:
    """alpha_f, alpha_m, beta, gamma by FEBio's rules (FESolidSolver2::Init).

    rhoi = -1          alpha_f = alpha_m = 1, beta = 1/4, gamma = 1/2
    rhoi in [0, 1]     alpha_f = 1/(1 + rhoi), alpha_m = (2 - rhoi)/(1 + rhoi),
                       beta = (1 + alpha_m - alpha_f)^2 / 4,
                       gamma = 1/2 + alpha_m - alpha_f
    any other value    alpha_f = alpha_m = alpha, with beta and gamma as given
                       (the default: rhoi = -2, alpha = 1, beta = 1/4, gamma = 1/2)
    """

    alpha_f: float = 1.0
    alpha_m: float = 1.0
    beta: float = 0.25
    gamma: float = 0.5

    @classmethod
    def from_solver(cls, solver) -> TimeIntegration:
        get = (lambda tag, default: float(solver.get(tag, default))) if solver is not None else (lambda _, d: d)
        rhoi, alpha, beta, gamma = get("rhoi", -2.0), get("alpha", 1.0), get("beta", 0.25), get("gamma", 0.5)
        if rhoi == -1.0:
            return cls(1.0, 1.0, 0.25, 0.5)
        if 0.0 <= rhoi <= 1.0:
            alpha_f = 1.0 / (1.0 + rhoi)
            alpha_m = (2.0 - rhoi) / (1.0 + rhoi)
            return cls(alpha_f, alpha_m, (1.0 + alpha_m - alpha_f) ** 2 / 4.0, 0.5 + alpha_m - alpha_f)
        return cls(alpha, alpha, beta, gamma)

    @classmethod
    def from_control(cls, control) -> TimeIntegration | None:
        """The settings of a step, or None for a static step."""
        if str(control.get("analysis", "STATIC")).upper() != "DYNAMIC":
            return None
        return cls.from_solver(control.child("solver"))


def is_dynamic(model) -> bool:
    return any(str(s.control.get("analysis", "STATIC")).upper() == "DYNAMIC" for s in model.steps)


@wp.kernel
def _kinematics(
    u: wp.array(dtype=vec3d),
    u_n: wp.array(dtype=vec3d),
    v_n: wp.array(dtype=vec3d),
    a_n: wp.array(dtype=vec3d),
    dt: wp.float64,
    beta: wp.float64,
    gamma: wp.float64,
    alpha_m: wp.float64,
    v: wp.array(dtype=vec3d),
    a: wp.array(dtype=vec3d),
    a_mass: wp.array(dtype=vec3d),
):
    """FESolidSolver2::UpdateKinematics for every entry (displacement and
    shell displacement alike):

        a = (u - u_n) / (beta dt^2) - v_n / (beta dt) + (1 - 1/(2 beta)) a_n
        v = v_n + ((1 - gamma) a_n + gamma a) dt

    and the acceleration the inertial force acts on,
    alpha_m a + (1 - alpha_m) a_n (FEElasticSolidDomain::UpdateElementStress).
    With u = u_n it is the start of a step (PrepStep).
    """
    i = wp.tid()
    one = wp.float64(1.0)
    at = (u[i] - u_n[i]) / (beta * dt * dt) - v_n[i] / (beta * dt) + (one - wp.float64(0.5) / beta) * a_n[i]
    a[i] = at
    v[i] = v_n[i] + ((one - gamma) * a_n[i] + gamma * at) * dt
    a_mass[i] = alpha_m * at + (one - alpha_m) * a_n[i]


def new_dynamic_state(system: System, state0: State, time: float) -> State:
    """The trial State of a step from `state0` to `time`: u copied, v and a
    to be filled by `update_kinematics`."""
    zeros = lambda: wp.zeros(system.n, dtype=vec3d, device=system.device)
    return State(time, wp.clone(state0.u), v=zeros(), a=zeros(), a_mass=zeros(), dt=time - state0.time)


def update_kinematics(system: System, state: State, state0: State) -> None:
    """v and a of `state` from its u and the previous state. One launch."""
    ti = system.time_integration
    wp.launch(
        _kinematics,
        dim=system.n,
        inputs=[state.u, state0.u, state0.v, state0.a, state.dt, ti.beta, ti.gamma, ti.alpha_m],
        outputs=[state.v, state.a, state.a_mass],
        device=system.device,
    )


def initial_state(system: System, linear_solver: str = "direct") -> State:
    """The State at time zero of a dynamic step: at rest, with FEBio's
    initial accelerations.

    FESolidSolver2::InitAccelerations: the applied loads at time zero (no
    internal force) are balanced by the mass alone, M a = F on the free
    components, and only if F is not zero. A load that is stepped on at time
    zero therefore starts the motion with the right acceleration.
    """
    n, device = system.n, system.device
    zeros = lambda: wp.zeros(n, dtype=vec3d, device=device)
    state = State(0.0, zeros(), v=zeros(), a=zeros(), a_mass=zeros(), dt=0.0)
    system.update(state)
    # The terms write the (row, column) of their blocks when they linearize.
    # The mass uses those positions, so one tangent comes first.
    system.tangent(state)
    F = zeros()
    for term in system.terms:
        if not term.internal:
            term.residual(state, None, F)
    system.mask(F)
    if float(wp.utils.array_inner(F, F)) > 0.0:
        A = system.mass_matrix()
        a0, _ = solve(A, F, linear_solver)
        system.mask(a0)
        state.a = a0
        state.a_mass = wp.clone(a0)
    return state


def lumped_mass(system: System) -> np.ndarray:
    """Row sums of the mass blocks per entry, on the host. For checks: the
    total is the mass of the model (once per face for a shell)."""
    out = np.zeros(system.n)
    np.add.at(out, system.rows.numpy()[: system.block_count], system.mass.numpy())
    return out
