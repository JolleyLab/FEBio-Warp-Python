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
from ..coverage import IMPLEMENTED, NO_EFFECT, NOT_SUPPORTED, CoverageError
from ..discretization.fields import FIXED, PRESCRIBED, build_field_table
from ..elements.brick_domain import BrickDomain
from ..elements.fem_domain import ELEMENTS as FEM_ELEMENTS
from ..elements.fem_domain import FemSolidDomain
from ..elements.line_builders import (
    build_discrete_domains,
    build_truss_domain,
    discrete_coverage,
    is_truss,
    truss_coverage,
)
from ..loads import mass_damping, nodal_load, pressure
from ..materials.febio import bind_material
from ..results import Results, StateResult
from ..solvers.nonlinear.newton import NewtonSettings, solve_step
from .dynamics import TimeIntegration, initial_state, is_dynamic
from .static import TimeStepController


def domain_support(kind, element_type, domain_type) -> bool:
    """Element types this run path has beyond the fixture's hex8 and quad4."""
    return kind == "SolidDomain" and element_type in FEM_ELEMENTS and domain_type in (None, "elastic-solid")


def line_support(model, block, dom):
    """Coverage rows of a truss part, or None for any other part."""
    return truss_coverage(model, block, dom) if is_truss(block, dom) else None


def discrete_support(model):
    """Coverage rows of the Discrete section (springs)."""
    return discrete_coverage(model)


def dynamic_support(control):
    """Coverage of a DYNAMIC step: (status, note). FEBio's scheme with
    alpha_f = 1, which the defaults and rhoi = 0 or -1 give. Below 1 the
    stress is evaluated at an interpolated deformation (decision D3 of
    docs/design/dynamics.md), which is reserved."""
    ti = TimeIntegration.from_control(control)
    note = f"alpha_f {ti.alpha_f:g}, alpha_m {ti.alpha_m:g}, beta {ti.beta:g}, gamma {ti.gamma:g}"
    if ti.alpha_f != 1.0:
        return NOT_SUPPORTED, note + ": alpha_f below 1 is not yet supported"
    return IMPLEMENTED, note


def inspect_model(model):
    """The coverage report of this run path: what `build_system` can build."""
    report = inspect(
        model,
        load_support=load_support,
        domain_support=domain_support,
        line_support=line_support,
        discrete_support=discrete_support,
        dynamic_support=dynamic_support,
    )
    if is_dynamic(model):
        for block in model.mesh.blocks:
            if block.febio_type == "tet4":
                # tet4 runs through warp.fem here; its mass is not built yet.
                report.add("analysis", "DYNAMIC with tet4", NOT_SUPPORTED, f"part {block.name}", "mass of tet4")
    return report


def select_domain(model, block, fields, directors, device, solids="dedicated"):
    """The domain term for one element block.

    FEBio counterpart: FESolidDomainFactory, which picks the domain class
    from the element shape, the material class and the domain settings.

        quad4 shell   BrickDomain (warp.fem has no thick shell)
        hex8 solid    BrickDomain, or FemSolidDomain with solids="warp.fem"
        tet4 solid    FemSolidDomain
        line2 truss   TrussDomain

    penta6, tri3 and three-field domains are further branches that return
    other Terms. Springs are not parts: `build_system` adds them from the
    Discrete section.
    """
    dom = model.domain_of(block.name)
    if is_truss(block, dom):
        return build_truss_domain(model, block, fields, device)
    bound, _ = bind_material(model.material(dom.material), model)
    conn = block.connectivity
    nodes = model.mesh.nodes
    if dom.kind == "SolidDomain":
        if block.febio_type != "hex8" or solids == "warp.fem":
            return FemSolidDomain(
                block.name,
                block.febio_type,
                bound,
                nodes,
                conn,
                fields.entries_of("displacement"),
                fields.count,
                device,
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
            [
                fields.entries_of("shell displacement", conn),
                fields.entries_of("displacement", conn),
            ],
            axis=1,
        )
        febio_type = "elastic-shell"
    return BrickDomain(block.name, febio_type, bound, X, emap, fields.count, model, device)


# Loads this run path can build: FEBio type -> module with inspect_load and a Term.
LOADS = {
    pressure.TYPE: pressure,
    nodal_load.TYPE: nodal_load,
    nodal_load.FORCE_TYPE: nodal_load,
    mass_damping.TYPE: mass_damping,
}


def load_support(record, model):
    """Coverage of one load record, for the coverage report."""
    module = LOADS.get(record.type)
    if module is mass_damping and not is_dynamic(model):
        return NO_EFFECT, "static analysis"
    return module.inspect_load(record, model) if module is not None else None


def select_load(record, model, fields, directors, device):
    """The term for one load record (FEBio: the FEModelLoad class by type)."""
    if record.type in (nodal_load.TYPE, nodal_load.FORCE_TYPE):
        return nodal_load.NodalLoad(record, model, fields, directors, device)
    return pressure.PressureLoad(record, model, fields, directors, device)


def build_system(model, device=None, solids: str = "dedicated") -> System:
    """Check coverage, then build the field table and the term list.

    Raises CoverageError, carrying the report, if the model needs a feature
    that is not supported. The coverage rules are those of pull request 5,
    extended by what this run path adds (loads, tet4).
    solids: "dedicated" (hex8 by the hand-written brick kernel) or
    "warp.fem" (hex8 through warp.fem). tet4 always goes through warp.fem.
    """
    report = inspect_model(model)
    if not report.runnable:
        raise CoverageError(report)
    device = wp.get_device(device)
    fields = build_field_table(model)
    directors = shell_directors(model)
    terms = [select_domain(model, b, fields, directors, device, solids) for b in model.mesh.blocks]
    # Springs of the Discrete section: one term per spring law in use.
    terms += build_discrete_domains(model, fields, device)
    for record in model.loads:
        if record.type == mass_damping.TYPE:
            continue  # not a term: applied through the mass blocks, below
        terms.append(select_load(record, model, fields, directors, device))
    # Contact interfaces are appended here as further terms; not implemented
    # in this template.
    system = System(model, fields, terms, device)
    system.report = report
    if is_dynamic(model):
        # Reference density of each part, then the mass of every term and
        # the mass damping coefficients, once (analysis/dynamics.py).
        for term, block in zip(terms, model.mesh.blocks):
            material = model.material(model.domain_of(block.name).material)
            term.density = float(material.params.get("density", 1.0))
        for term, block in zip(terms, model.mesh.blocks):
            if term.mass_coefficients() is None:
                # hex8 through warp.fem (solids="warp.fem") and tet4: no mass yet.
                raise NotImplementedError(
                    f"part {block.name!r}: the mass of this domain is not built, so it cannot run in a dynamic step"
                )
        system.set_dynamics(TimeIntegration.from_control(model.steps[0].control), mass_damping.damping_by_part(model))
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
    # FEBio's nodal data "R" (log data Rx, Ry, Rz; plot variable "reaction
    # forces"): minus the internal force at every component that is free or
    # prescribed (FESolidSolver2::Residual takes it after InternalForces),
    # and zero at fixed components, which have no equation in FEBio. The
    # terms' data is current, since a converged step ends with an update.
    internal = wp.zeros(system.n, dtype=wp.vec3d, device=system.device)
    for term in system.terms:
        if term.internal:
            term.residual(state, None, internal)  # adds minus the internal force
    reaction = internal.numpy() * (fields.bc != FIXED)
    prescribed = fields.bc == PRESCRIBED
    if system.time_integration is not None and state.a_mass is not None and prescribed.any():
        # At a prescribed component FEBio also counts what its domains
        # assemble element by element after the internal forces: the inertial
        # force and body forces such as mass damping (FEResidualVector::
        # Assemble, m_Fr[-I - 2] -= fe[i]). Contact forces are not counted
        # there (sh_self_contact_fold agrees with FEBio without them).
        # Whether surface loads are is not yet checked against a FEBio run.
        dynamic = wp.zeros(system.n, dtype=wp.vec3d, device=system.device)
        system._dynamic_residual(state, dynamic)
        reaction[prescribed] += dynamic.numpy()[prescribed]
    out.reaction_force = reaction[fields.entries_of("displacement")]
    if state.v is not None:
        out.velocity = state.v.numpy()[fields.entries_of("displacement")]
    shell = fields.entries_of("shell displacement")
    if (shell >= 0).any():
        out.shell_displacement = np.zeros((len(shell), 3))
        out.shell_displacement[shell >= 0] = u[shell[shell >= 0]]
    for term in system.terms:
        if not hasattr(term, "plot"):
            continue
        if term.element_type == "discrete":
            # Springs have no stress. Their outputs are kept by discrete set
            # name, under FEBio's plot variable names.
            for variable in ("discrete element force", "discrete element signed force"):
                for name, values in term.plot_by_set(variable).items():
                    out.discrete.setdefault(name, {})[variable] = values
            continue
        if term.element_type == "line2":
            out.truss_stretch[term.name] = term.plot("truss stretch")
            out.axial_force[term.name] = term.plot("axial force")
        # A copy: on the CPU `numpy()` is a view of the device array, which
        # the next step overwrites.
        s = np.array(term.plot("stress"))  # (E, Q, 3, 3)
        mean = s.mean(axis=1)
        out.stress_points[term.name] = s
        # FEBio's order xx, yy, zz, xy, yz, xz; plain mean over points, as
        # FEBio's plot variable "stress".
        out.stress[term.name] = np.stack(
            [mean[:, i, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))],
            axis=1,
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

    if system.time_integration is not None:
        # At rest, with FEBio's initial accelerations if a load acts at time 0.
        state = initial_state(system, linear_solver)
    else:
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
            new_state, iterations = solve_step(system, state, new_time, settings, linear_solver, say)
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
                # FEBio passes m_niter, the number of Newton iterations of
                # the converged step (FEAnalysis.cpp, AutoTimeStep call).
                dt = controller.auto(iterations, time, dt)
    return (results, states) if keep_states else results
