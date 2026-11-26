"""The run path: build a system from a Model, then drive it through its steps.

TEMPLATE (run path draft, November 2026). Structure for discussion, not final.
It runs the same inputs as the fixture of pull request 5 (analysis/static.py)
and is checked against it and against the stored FEBio results.

FEBio counterpart
    FEModel::Init builds the domains and numbers the equations; FEAnalysis
    activates a step's features and loops over time; FETimeStepController
    adapts the step size.
    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEModel.cpp
    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEAnalysis.cpp

Same as FEBio
    Steps, control settings and their names; load controllers evaluated at
    the time being solved; the time step controller with retries and must
    points (the class of analysis/static.py is reused unchanged).

Different, and why
    `build_system` returns a field table and a list of terms; `run` knows
    nothing else about the model. A step moves from one State to the next
    and keeps the previous State until the new one converges, so a failed
    attempt needs no undo and a converged State can be handed to the adjoint.

Where the next features attach
    build_system    one branch per domain type (select_domain), one per load,
                    one per contact interface; each returns a Term
    run             per-step activation of terms; the analysis type (static
                    here; dynamic adds inertia through the same terms)
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..assembly.system import SolveError, State, System
from ..compiled import inspect, shell_directors
from ..coverage import CoverageError
from ..discretization.fields import build_field_table
from ..elements.brick_domain import BrickDomain
from ..elements.fem_domain import ELEMENTS as FEM_ELEMENTS
from ..elements.fem_domain import FemSolidDomain
from ..loads import pressure
from ..materials.febio import bind_material
from ..results import Results, StateResult
from ..solvers.nonlinear.newton import NewtonSettings, solve_step
from .static import TimeStepController


def domain_support(kind, element_type, domain_type) -> bool:
    """Element types this run path has beyond the fixture's hex8 and quad4."""
    return (
        kind == "SolidDomain"
        and element_type in FEM_ELEMENTS
        and domain_type in (None, "elastic-solid")
    )


def select_domain(model, block, fields, directors, device, solids="dedicated"):
    """The domain term for one element block.

    FEBio counterpart: FESolidDomainFactory, which picks the domain class
    from the element shape, the material class and the domain settings.

        quad4 shell   BrickDomain (warp.fem has no thick shell)
        hex8 solid    BrickDomain, or FemSolidDomain with solids="warp.fem"
        tet4 solid    FemSolidDomain

    penta6, tri3, three-field and discrete domains are further branches that
    return other Terms.
    """
    dom = model.domain_of(block.name)
    bound, _ = bind_material(model.material(dom.material), model)
    conn = block.connectivity
    nodes = model.mesh.nodes
    if dom.kind == "SolidDomain":
        if block.febio_type != "hex8" or solids == "warp.fem":
            return FemSolidDomain(
                block.name, block.febio_type, bound, nodes, conn,
                fields.entries_of("displacement"), fields.count, device,
            )
        X = nodes[conn]  # (E, 8, 3)
        emap = fields.entries_of("displacement", conn)
        febio_type = "elastic-solid"
    else:
        # Shell as a brick: nodes 0..3 are the back face (X - d0, shell
        # displacement), nodes 4..7 the front face (X, displacement).
        front = nodes[conn]
        X = np.concatenate([front - directors[conn], front], axis=1)
        emap = np.concatenate(
            [fields.entries_of("shell displacement", conn), fields.entries_of("displacement", conn)],
            axis=1,
        )
        febio_type = "elastic-shell"
    return BrickDomain(block.name, febio_type, bound, X, emap, fields.count, model, device)


# Loads this run path can build: FEBio type -> module with inspect_load and a Term.
LOADS = {pressure.TYPE: pressure}


def load_support(record, model):
    """Coverage of one load record, for the coverage report."""
    module = LOADS.get(record.type)
    return module.inspect_load(record, model) if module is not None else None


def select_load(record, model, fields, directors, device):
    """The term for one load record (FEBio: the FEModelLoad class by type)."""
    return pressure.PressureLoad(record, model, fields, directors, device)


def build_system(model, device=None, solids: str = "dedicated") -> System:
    """Check coverage, then build the field table and the term list.

    Raises CoverageError, carrying the report, if the model needs a feature
    that is not supported. The coverage rules are those of pull request 5,
    extended by what this run path adds (loads, tet4).
    solids: "dedicated" (hex8 by the hand-written brick kernel) or
    "warp.fem" (hex8 through warp.fem). tet4 always goes through warp.fem.
    """
    report = inspect(model, load_support=load_support, domain_support=domain_support)
    if not report.runnable:
        raise CoverageError(report)
    device = wp.get_device(device)
    fields = build_field_table(model)
    directors = shell_directors(model)
    terms = [
        select_domain(model, b, fields, directors, device, solids) for b in model.mesh.blocks
    ]
    for record in model.loads:
        terms.append(select_load(record, model, fields, directors, device))
    # Discrete elements and contact interfaces are appended here as further
    # terms; neither is implemented in this template.
    system = System(model, fields, terms, device)
    system.report = report
    return system


def collect(system: System, state: State, iterations: int = 0) -> StateResult:
    """Copy one state to the host in the form the writers use."""
    fields = system.fields
    u = state.u.numpy()
    out = StateResult(
        time=state.time,
        displacement=u[fields.entries_of("displacement")],
        iterations=iterations,
    )
    shell = fields.entries_of("shell displacement")
    if (shell >= 0).any():
        out.shell_displacement = np.zeros((len(shell), 3))
        out.shell_displacement[shell >= 0] = u[shell[shell >= 0]]
    for term in system.terms:
        if not hasattr(term, "plot"):
            continue
        s = term.plot("stress")  # (E, Q, 3, 3)
        mean = s.mean(axis=1)
        out.stress_points[term.name] = s
        # FEBio's order xx, yy, zz, xy, yz, xz; plain mean over points, as
        # FEBio's plot variable "stress".
        out.stress[term.name] = np.stack(
            [mean[:, i, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))], axis=1
        )
    return out


def run(system: System, log=None, linear_solver: str = "direct", keep_states: bool = False):
    """Run the model's step. Returns Results (the states FEBio would plot);
    with keep_states=True, also the list of converged device States."""
    say = log or (lambda *_: None)
    model = system.model
    # One step, as the coverage report enforces today. With several, this
    # body becomes a loop that activates each step's terms and conditions.
    control = model.steps[0].control
    settings = NewtonSettings.from_record(control.child("solver"))
    nsteps = int(control.get("time_steps", 10))
    dt0 = float(control.get("step_size", 0.1))
    tend = nsteps * dt0
    eps = tend * 1e-12
    must_only = str(control.get("plot_level", "PLOT_MAJOR_ITRS")) == "PLOT_MUST_POINTS"
    stepper = control.child("time_stepper")
    controller = TimeStepController(stepper, model, dt0, tend) if stepper is not None else None

    state = system.state()
    results = Results()
    states = [state]
    system.update(state)  # stresses of the reference state (zero)
    results.states.append(collect(system, state))

    time, dt = 0.0, dt0
    if controller is not None:
        dt = controller.auto(0, time, dt)
    while tend - time > eps:
        new_time = min(time + dt, tend)
        try:
            new_state, iterations = solve_step(
                system, state, new_time, settings, linear_solver, say
            )
        except SolveError as err:
            if controller is not None and controller.retries < controller.max_retries:
                dt = controller.retry(dt)
                say(f"step to time {new_time:g} failed ({err}); retrying with dt {dt:g}")
                continue
            raise
        at_must = controller.at_must if controller is not None else True
        state, time = new_state, new_time
        say(f"converged at time {time:.6g} in {iterations} iterations")
        if not must_only or at_must:
            results.states.append(collect(system, state, iterations=iterations))
            states.append(state)
        if controller is not None:
            controller.retries = 0
            if time + eps < tend:
                # FEBio passes its iteration counter, one less than the
                # number of solves in the converged step.
                dt = controller.auto(iterations - 1, time, dt)
    return (results, states) if keep_states else results
