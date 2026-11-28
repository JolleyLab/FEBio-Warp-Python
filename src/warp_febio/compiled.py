"""Compile a Model for Warp: coverage report, numbering and device arrays.

FEBio counterpart: FEModel::Init, which builds domains from the mesh, numbers
the equations and initializes shells
(https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEModel.cpp).

Data movement, stated once:
    host -> device   here, once, when the Model is compiled
    device -> host   at output steps only (analysis/static.py), plus the small
                     linear solve of this draft, which runs on the host

`CompiledModel` owns the device arrays. `State` holds the solution at one time
and is separate from the model, so several states can exist for one compiled
model (output history now; reverse-mode differentiation later).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import warp as wp
except ImportError as err:  # the model and I/O layers work without Warp
    raise ImportError(
        "Compiling and running a model needs Warp. Install it with "
        "pip install 'warp_febio[warp]'. Reading and writing FEBio files "
        "(warp_febio.io, warp_febio.model) does not need it."
    ) from err

from .coverage import (
    CONTROL_IMPLEMENTED,
    CONTROL_NO_EFFECT,
    IMPLEMENTED,
    NO_EFFECT,
    NOT_SUPPORTED,
    NOT_WRITTEN,
    SOLVER_IMPLEMENTED,
    SOLVER_NO_EFFECT,
    CoverageError,
    CoverageReport,
)
from .elements.brick8 import GAUSS_WEIGHTS, make_brick_kernel, shape_derivatives
from .materials.febio import bind_material
from .model import Model

# Degrees of freedom per node, FEBio names
# (FEBioMech::InitModule: "displacement" x, y, z; "shell displacement" sx, sy, sz).
DOF_NAMES = ("x", "y", "z", "sx", "sy", "sz")
PLOT_VARIABLES = {"displacement", "stress"}


def inspect(
    model: Model,
    load_support=None,
    domain_support=None,
    line_support=None,
    discrete_support=None,
    dynamic_support=None,
    contact_support=None,
) -> CoverageReport:
    """Coverage report for a Model. Allocates nothing on the device.

    load_support: optional function (record, model) -> (status, note) or None,
    through which a run path says which loads it can build. Without it every
    load is reported as not supported, as this fixture has none.
    domain_support: optional function (kind, element type, domain type) ->
    bool, for element types a run path has beyond this fixture's two.
    line_support: optional function (model, block, domain) -> list of report
    rows or None, for parts of two-node elements (trusses). It reports the
    domain, its settings and its material itself, since a truss material is
    not a solid material.
    discrete_support: optional function (model) -> list of report rows for the
    Discrete section (springs). Without it the section is not supported.
    dynamic_support: optional function (control record) -> (status, note), by
    which a run path says whether it can run a DYNAMIC step with that step's
    time integration settings. Without it a dynamic step is not supported.
    """
    rep = CoverageReport()
    for issue in model.issues:
        rep.add("input", issue.where, NOT_SUPPORTED, issue.where, issue.message)
    rep.add(
        "module",
        model.module,
        IMPLEMENTED if model.module == "solid" else NOT_SUPPORTED,
    )
    if model.globals is not None:
        rep.add(
            "globals",
            "Constants",
            NO_EFFECT,
            note="not used by the features implemented",
        )

    # Domains, elements and materials.
    for block in model.mesh.blocks:
        try:
            dom = model.domain_of(block.name)
        except KeyError:
            rep.add(
                "domain",
                block.name,
                NOT_SUPPORTED,
                note="part has no MeshDomains entry",
            )
            continue
        dtype = dom.params.attrs.get("type")
        where = f"part {block.name}"
        rows = line_support(model, block, dom) if line_support is not None else None
        if rows is not None:
            for row in rows:
                rep.add(*row)
            continue
        if dom.kind == "SolidDomain" and block.febio_type == "hex8" and dtype in (None, "elastic-solid"):
            rep.add("domain", "elastic-solid, hex8", IMPLEMENTED, where)
        elif dom.kind == "ShellDomain" and block.febio_type == "quad4" and dtype in (None, "elastic-shell"):
            rep.add("domain", "elastic-shell, quad4", IMPLEMENTED, where)
        elif domain_support is not None and domain_support(dom.kind, block.febio_type, dtype):
            rep.add(
                "domain",
                f"{dtype or 'elastic-solid'}, {block.febio_type}",
                IMPLEMENTED,
                where,
            )
        else:
            rep.add(
                "domain",
                f"{dom.kind} {dtype or '(default)'}, {block.febio_type}",
                NOT_SUPPORTED,
                where,
            )
        for c in dom.params.children:
            if dom.kind == "ShellDomain" and c.tag == "shell_thickness":
                rep.add("domain setting", c.tag, IMPLEMENTED, where)
            elif dom.kind == "ShellDomain" and c.tag == "shell_normal_nodal" and c.value == 1:
                rep.add("domain setting", c.tag, IMPLEMENTED, where)
            else:
                rep.add("domain setting", c.tag, NOT_SUPPORTED, where)
        try:
            mat = model.material(dom.material)
        except KeyError:
            rep.add("material", dom.material, NOT_SUPPORTED, where, "material not found")
            continue
        # One interpretation of a FEBio material, shared with the tests and
        # with compile: materials/febio.py. A solid mixture is reported with
        # its children's parameters under FEBio-style names (solid[0].E).
        _, issues = bind_material(mat, model)
        for issue in issues:
            rep.add(
                issue.category,
                issue.febio_name,
                issue.status,
                f"material {mat.name}",
                issue.note,
            )
        if mat.params.child("density") is not None:
            dynamic = dynamic_support is not None and any(
                str(s.control.get("analysis", "STATIC")).upper() == "DYNAMIC" for s in model.steps
            )
            rep.add(
                "material parameter",
                "density",
                IMPLEMENTED if dynamic else NO_EFFECT,
                f"material {mat.name}",
                "" if dynamic else "static analysis",
            )

    # Shells that share nodes with solid elements. FEBio then ties the solid to
    # the shell's back face (FESSIShellDomain::FindSSI), which is not done here.
    shell_nodes, solid_nodes = set(), set()
    for block in model.mesh.blocks:
        try:
            kind = model.domain_of(block.name).kind
        except KeyError:
            continue
        nodes = set(np.unique(block.connectivity).tolist())
        if kind == "ShellDomain":
            shell_nodes |= nodes
        elif kind == "SolidDomain":
            # FindSSI looks at solid domains only. A truss or a spring that
            # shares a node with a shell acts on the node's `displacement`
            # degrees of freedom (the front face) and ties nothing.
            solid_nodes |= nodes
    if shell_nodes & solid_nodes:
        rep.add(
            "domain",
            "shell attached to solid elements",
            NOT_SUPPORTED,
            note="shell-solid interface (FESSIShellDomain::FindSSI)",
        )

    for r in model.mesh_data:
        if r.tag == "ElementData" and r.attrs.get("type") == "shell thickness" and isinstance(r.value, np.ndarray):
            rep.add(
                "mesh data",
                "shell thickness",
                IMPLEMENTED,
                f"part {r.attrs.get('elem_set')}",
            )
        else:
            rep.add("mesh data", f"{r.tag} {r.attrs.get('type')}", NOT_SUPPORTED)
    for r in model.mesh.other:
        if r.tag == "Edge":
            rep.add("mesh", "Edge", IMPLEMENTED, r.attrs.get("name", ""), "as a node set")
        else:
            rep.add(
                "mesh",
                r.tag,
                NO_EFFECT,
                r.attrs.get("name", ""),
                "definition only; features that use it are listed separately",
            )

    # Steps.
    if len(model.steps) != 1:
        rep.add(
            "analysis",
            f"{len(model.steps)} steps",
            NOT_SUPPORTED,
            note="one step only in this draft",
        )
    for step in model.steps:
        _inspect_control(step.control, rep, f"step {step.id}", dynamic_support)
        for r in step.loads:
            # With one step, a load of the step acts for the whole analysis,
            # as a load of the model does. Activation by step is reserved.
            known = load_support(r, model) if load_support is not None and len(model.steps) == 1 else None
            status, note = known if known is not None else (NOT_SUPPORTED, "")
            rep.add("load", r.type or r.tag, status, f"step {step.id}: {r.name or ''}", note)
        for r in step.contact + step.other:
            rep.add(r.tag, r.type or r.tag, NOT_SUPPORTED, f"step {step.id}")
    for r in model.boundary + [b for s in model.steps for b in s.boundary]:
        _inspect_bc(r, model, rep)
    for r in model.loads:
        known = load_support(r, model) if load_support is not None else None
        status, note = known if known is not None else (NOT_SUPPORTED, "")
        rep.add("load", r.type or r.tag, status, r.name or "", note)
    for r in model.contact:
        # contact_support: optional function (record, model) -> (status, note).
        status, note = contact_support(r, model) if contact_support is not None else (NOT_SUPPORTED, "")
        rep.add("contact", r.type or r.tag, status, r.name or "", note)
    for r in model.other_sections:
        if r.tag == "Discrete" and discrete_support is not None:
            for row in discrete_support(model):
                rep.add(*row)
            continue
        if r.children:
            for c in r.children:
                rep.add(r.tag, c.type or c.tag, NOT_SUPPORTED, c.name or "")
    for lc in model.load_controllers:
        interp = str(lc.params.get("interpolate", "LINEAR")).upper()
        extend = str(lc.params.get("extend", "CONSTANT")).upper()
        ok = (
            lc.type == "loadcurve"
            and interp in ("LINEAR", "STEP", "SMOOTH")
            and extend == "CONSTANT"
            and lc.points is not None
            and len(lc.points) >= 2
        )
        rep.add(
            "load controller",
            f"{lc.type}, {interp}, {extend}",
            IMPLEMENTED if ok else NOT_SUPPORTED,
            f"id {lc.id}",
        )

    # Output requests.
    if model.output is not None:
        for sec in model.output.children:
            if sec.tag == "plotfile":
                for v in sec.all("var"):
                    name = v.attrs.get("type")
                    rep.add(
                        "plot variable",
                        name,
                        IMPLEMENTED if name in PLOT_VARIABLES else NOT_WRITTEN,
                    )
            else:
                for c in sec.children or [sec]:
                    rep.add(
                        "output",
                        f"{sec.tag}/{c.tag}",
                        NOT_WRITTEN,
                        note=str(c.attrs.get("data", "")),
                    )
    return rep


TIME_INTEGRATION_SETTINGS = ("rhoi", "alpha", "beta", "gamma")


def _inspect_control(control, rep, where, dynamic_support=None):
    dynamic = str(control.get("analysis", "STATIC")).upper() == "DYNAMIC"
    for c in control.children:
        if c.tag == "analysis":
            if dynamic and dynamic_support is not None:
                status, note = dynamic_support(control)
                rep.add("analysis", c.value, status, where, note)
            else:
                ok = str(c.value).upper() == "STATIC"
                rep.add("analysis", c.value, IMPLEMENTED if ok else NOT_SUPPORTED, where)
        elif c.tag == "plot_range":
            ok = list(np.atleast_1d(c.value)) == [0, -1]
            rep.add(
                "control",
                "plot_range",
                NO_EFFECT if ok else NOT_SUPPORTED,
                where,
                "0,-1: every state is written",
            )
        elif c.tag in CONTROL_IMPLEMENTED:
            rep.add("control", c.tag, IMPLEMENTED, where)
        elif c.tag in CONTROL_NO_EFFECT:
            rep.add("control", c.tag, NO_EFFECT, where, CONTROL_NO_EFFECT[c.tag])
        elif c.tag == "solver":
            if c.type not in (None, "solid"):
                rep.add("solver", c.type, NOT_SUPPORTED, where)
            for s in c.children:
                if s.tag in SOLVER_IMPLEMENTED:
                    rep.add("solver setting", s.tag, IMPLEMENTED, where)
                elif dynamic and dynamic_support is not None and s.tag in TIME_INTEGRATION_SETTINGS:
                    rep.add("solver setting", s.tag, IMPLEMENTED, where, "time integration")
                elif s.tag == "arc_length" and s.value != 0:
                    rep.add("solver setting", "arc_length", NOT_SUPPORTED, where)
                elif s.tag in SOLVER_NO_EFFECT:
                    rep.add(
                        "solver setting",
                        s.tag,
                        NO_EFFECT,
                        where,
                        SOLVER_NO_EFFECT[s.tag],
                    )
                else:
                    rep.add("solver setting", s.tag, NOT_SUPPORTED, where)
        elif c.tag == "plot_level":
            ok = str(c.value) in ("PLOT_MAJOR_ITRS", "PLOT_MUST_POINTS")
            rep.add(
                "control",
                f"plot_level {c.value}",
                IMPLEMENTED if ok else NOT_SUPPORTED,
                where,
            )
        elif c.tag == "time_stepper":
            ok = c.type in (None, "default")
            rep.add(
                "control",
                "time_stepper",
                IMPLEMENTED if ok else NOT_SUPPORTED,
                where,
                "step sizes follow the iteration count, which can differ from FEBio's (full Newton here)",
            )
            for s_ in c.children:
                if s_.tag not in (
                    "max_retries",
                    "opt_iter",
                    "dtmin",
                    "dtmax",
                    "aggressiveness",
                    "cutback",
                    "dtforce",
                ):
                    rep.add("time stepper setting", s_.tag, NOT_SUPPORTED, where)
        else:
            rep.add("control", c.tag, NOT_SUPPORTED, where)


_BC_FLAGS = {
    "zero displacement": ("x_dof", "y_dof", "z_dof"),
    "zero shell displacement": ("sx_dof", "sy_dof", "sz_dof"),
}
_BC_PRESCRIBED = {
    "prescribed displacement": ("x", "y", "z"),
    "prescribed shell displacement": ("sx", "sy", "sz"),
}


def _inspect_bc(r, model, rep):
    where = r.name or ""
    if r.tag != "bc" or r.type not in {**_BC_FLAGS, **_BC_PRESCRIBED}:
        rep.add("boundary condition", r.type or r.tag, NOT_SUPPORTED, where)
        return
    try:
        model.mesh.node_set_indices(r.attrs.get("node_set", ""))
    except KeyError:
        rep.add(
            "boundary condition",
            r.type,
            NOT_SUPPORTED,
            where,
            f"node set {r.attrs.get('node_set')!r} not resolved",
        )
        return
    known = set(_BC_FLAGS.get(r.type, ())) | ({"dof", "value", "relative"} if r.type in _BC_PRESCRIBED else set())
    status, note = IMPLEMENTED, ""
    for c in r.children:
        if c.tag not in known:
            status, note = NOT_SUPPORTED, f"setting {c.tag!r}"
    if r.type in _BC_PRESCRIBED:
        if r.get("relative", 0) != 0:
            status, note = NOT_SUPPORTED, "relative = 1"
        if r.get("dof") not in _BC_PRESCRIBED[r.type]:
            status, note = NOT_SUPPORTED, f"dof {r.get('dof')!r}"
    rep.add("boundary condition", r.type, status, where, note)


# ---------------------------------------------------------------------------
# compiled model and state


@dataclass
class CompiledDomain:
    """One part on the device: brick geometry, maps, material and work arrays."""

    def update_parameters(self, time: float):
        """Material parameters at `time`, for parameters on a load curve.

        FEBio evaluates load curves at the time of the step being solved and
        multiplies the parameter by the curve value. Parameters live in a
        device array here, so this is one small host-to-device write per step.
        """
        if self.material.time_dependent:
            pvec = self.material.model.param_vector(wp.float64)
            self.params.assign([pvec(*self.material.param_values(time))])

    name: str
    kind: str  # "SolidDomain" or "ShellDomain"
    material: Any  # BoundMaterial (materials/febio.py)
    kernel: Any
    count: int
    X: wp.array  # (E, 8) vec3d reference positions of the brick nodes
    dofmap: wp.array  # (E, 24) int32
    params: wp.array
    geom: wp.array
    fe: wp.array  # (E, 24)
    ke: wp.array  # (E, 24, 24)
    sigma: wp.array  # (E, Q) mat33d
    status: wp.array  # (E,)


@dataclass
class Prescribed:
    dofs: np.ndarray  # global degrees of freedom
    scale: float
    load_controller: int | None  # id, or None for a constant


@dataclass
class CompiledModel:
    model: Model
    report: CoverageReport
    device: Any
    dof_index: np.ndarray  # (N, 6) global degree of freedom, or -1
    ndof: int
    domains: list
    fixed: np.ndarray  # (ndof,) bool
    prescribed: list  # Prescribed
    free: np.ndarray  # indices of the equations to solve
    dN: wp.array  # (Q, 8, 3)
    gw: wp.array  # (Q,)
    directors: np.ndarray | None = None  # (N, 3) shell directors d0, host
    constrained_device: Any = None  # (ndof,) int32 on the device: 1 = fixed or prescribed

    def prescribed_values(self, time: float):
        """(dofs, values) of all prescribed degrees of freedom at `time`."""
        dofs, vals = [], []
        for p in self.prescribed:
            f = 1.0 if p.load_controller is None else self.model.load_controller(p.load_controller).value(time)
            dofs.append(p.dofs)
            vals.append(np.full(len(p.dofs), p.scale * f))
        if not dofs:
            return np.zeros(0, np.int64), np.zeros(0)
        return np.concatenate(dofs), np.concatenate(vals)

    def state(self, requires_grad: bool = False) -> State:
        """A zero state (reference configuration) for this model.

        The call mirrors Newton's `Model.state(requires_grad=...)`.
        """
        u = wp.zeros(self.ndof, dtype=wp.float64, device=self.device, requires_grad=requires_grad)
        return State(time=0.0, u=u)


@dataclass
class State:
    """The solution at one time. FEBio keeps this on nodes and material points."""

    time: float
    u: wp.array  # (ndof,) float64 global displacement vector


def shell_directors(model: Model) -> np.ndarray | None:
    """Initial nodal directors d0, as FEModel::InitShells computes them:
    at each shell node, the average over the attached element corners of the
    corner normal times the thickness there.
    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEModel.cpp
    """
    mesh = model.mesh
    D = np.zeros((len(mesh.nodes), 3))
    count = np.zeros(len(mesh.nodes), dtype=int)
    any_shell = False
    for block in mesh.blocks:
        dom = model.domain_of(block.name)
        if dom.kind != "ShellDomain":
            continue
        any_shell = True
        h = shell_thickness(model, block)
        n = block.connectivity.shape[1]
        for e, conn in enumerate(block.connectivity):
            r0 = mesh.nodes[conn]
            for j in range(n):
                a, b, c = r0[j], r0[(j + 1) % n], r0[j - 1]
                d = np.cross(b - a, c - a)
                D[conn[j]] += d / np.linalg.norm(d) * h[e, j]
                count[conn[j]] += 1
    if not any_shell:
        return None
    nz = count > 0
    D[nz] /= count[nz, None]
    return D


def shell_thickness(model: Model, block) -> np.ndarray:
    """Thickness at each node of each shell element, shape (E, nodes).

    From <ElementData type="shell thickness"> if present, otherwise from the
    domain's <shell_thickness> (FEBio ignores a domain value of 0).
    """
    n = block.connectivity.shape[1]
    data = model.element_data("shell thickness", block.name)
    dom = model.domain_of(block.name)
    h_dom = float(dom.params.get("shell_thickness", 0.0))
    # When both are given, FEBio uses the element data. Checked by running
    # FEBio 4.13.0 on the single-shell reference with each one and with both.
    if data is not None:
        return np.asarray(data.value, dtype=np.float64).reshape(block.count, n)
    if h_dom > 0.0:
        # FEShellDomainNew::AssignDefaultShellThickness
        return np.full((block.count, n), h_dom)
    raise ValueError(f"shell part {block.name!r} has no thickness")


def compile(model: Model, device=None) -> CompiledModel:
    """Check coverage, then build the device representation of a Model.

    Raises CoverageError, carrying the report, if the model needs a feature
    that is not yet supported.
    """
    report = inspect(model)
    if not report.runnable:
        raise CoverageError(report)
    device = wp.get_device(device)
    mesh = model.mesh
    N = len(mesh.nodes)

    # Degrees of freedom: x, y, z at every node; sx, sy, sz at shell nodes.
    active = np.zeros((N, 6), dtype=bool)
    active[:, :3] = True
    for block in mesh.blocks:
        if model.domain_of(block.name).kind == "ShellDomain":
            active[np.unique(block.connectivity), 3:] = True
    dof_index = np.full((N, 6), -1, dtype=np.int32)
    dof_index[active] = np.arange(active.sum(), dtype=np.int32)
    ndof = int(active.sum())

    directors = shell_directors(model)
    dN = wp.array(shape_derivatives(), dtype=wp.float64, device=device)
    gw = wp.array(GAUSS_WEIGHTS, dtype=wp.float64, device=device)

    domains = []
    for block in mesh.blocks:
        dom = model.domain_of(block.name)
        mat = model.material(dom.material)
        bound, _ = bind_material(mat, model)
        pvec = bound.model.param_vector(wp.float64)
        gvec = bound.model.geometry_vector(wp.float64)
        conn = block.connectivity
        E = block.count
        if dom.kind == "SolidDomain":
            X = mesh.nodes[conn]  # (E, 8, 3)
            dofmap = dof_index[conn][:, :, :3].reshape(E, 24)
        else:
            # Shell as a brick: nodes 0..3 are the back face (X - d0, shell
            # degrees of freedom), nodes 4..7 the front face (X, displacement
            # degrees of freedom). See elements/brick8.py.
            front = mesh.nodes[conn]
            back = front - directors[conn]
            X = np.concatenate([back, front], axis=1)
            dofmap = np.concatenate([dof_index[conn][:, :, 3:], dof_index[conn][:, :, :3]], axis=1).reshape(E, 24)
        nq = len(GAUSS_WEIGHTS)
        # One geometry row per element: material axes from the input's
        # mat_axis (identity where a model declares axes and the input has
        # none). Per-element axes from mesh data are not read yet.
        geometry = bound.geometry_array(E)
        domains.append(
            CompiledDomain(
                name=block.name,
                kind=dom.kind,
                material=bound,
                kernel=make_brick_kernel(bound),
                count=E,
                X=wp.array(X, dtype=wp.vec3d, device=device),
                dofmap=wp.array(dofmap.astype(np.int32), dtype=wp.int32, device=device),
                params=wp.array([pvec(*bound.param_values(0.0))], dtype=pvec, device=device),
                geom=wp.array(geometry, dtype=gvec, device=device),
                fe=wp.zeros((E, 24), dtype=wp.float64, device=device),
                ke=wp.zeros((E, 24, 24), dtype=wp.float64, device=device),
                sigma=wp.zeros((E, nq), dtype=wp.mat33d, device=device),
                status=wp.zeros(E, dtype=wp.int32, device=device),
            )
        )

    # Boundary conditions. FEBio counterparts: FEFixedDisplacement,
    # FEFixedShellDisplacement, FEPrescribedDisplacement, FEPrescribedShellDisplacement.
    fixed = np.zeros(ndof, dtype=bool)
    prescribed = []
    for r in model.boundary + [b for s in model.steps for b in s.boundary]:
        nodes = mesh.node_set_indices(r.attrs["node_set"])
        if r.type in _BC_FLAGS:
            for flag in _BC_FLAGS[r.type]:
                if r.get(flag, 0):
                    d = dof_index[nodes, DOF_NAMES.index(flag[: -len("_dof")])]
                    fixed[d[d >= 0]] = True
        else:
            d = dof_index[nodes, DOF_NAMES.index(r.get("dof"))]
            value = r.child("value")
            lc = value.attrs.get("lc")
            prescribed.append(
                Prescribed(
                    dofs=d[d >= 0],
                    scale=float(value.value),
                    load_controller=int(lc) if lc else None,
                )
            )
    constrained = fixed.copy()
    for p in prescribed:
        constrained[p.dofs] = True
    free = np.flatnonzero(~constrained)

    return CompiledModel(
        model=model,
        report=report,
        device=device,
        dof_index=dof_index,
        ndof=ndof,
        domains=domains,
        fixed=fixed,
        prescribed=prescribed,
        free=free,
        constrained_device=wp.array(constrained.astype(np.int32), dtype=wp.int32, device=device),
        dN=dN,
        gw=gw,
        directors=directors,
    )
