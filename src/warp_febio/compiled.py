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

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import warp as wp

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
from .materials.minimal import interface_kernel, minimal_kernel
from .model import Model

# Degrees of freedom per node, FEBio names
# (FEBioMech::InitModule: "displacement" x, y, z; "shell displacement" sx, sy, sz).
DOF_NAMES = ("x", "y", "z", "sx", "sy", "sz")
PLOT_VARIABLES = {"displacement", "stress"}


def material_kernel(febio_type: str, backend: str):
    """The material functions for a FEBio material type.

    backend "minimal"   : the self-contained neo-Hookean in materials/minimal.py
    backend "interface" : the material layer of draft pull request #2
    """
    if backend == "minimal":
        return minimal_kernel(febio_type)
    if backend == "interface":
        return interface_kernel(febio_type)
    raise ValueError(f"unknown material backend {backend!r}")


# ---------------------------------------------------------------------------
# coverage


def inspect(model: Model, material_backend: str = "minimal") -> CoverageReport:
    """Coverage report for a Model. Allocates nothing on the device."""
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
        if (
            dom.kind == "SolidDomain"
            and block.febio_type == "hex8"
            and dtype in (None, "elastic-solid")
        ):
            rep.add("domain", "elastic-solid, hex8", IMPLEMENTED, where)
        elif (
            dom.kind == "ShellDomain"
            and block.febio_type == "quad4"
            and dtype in (None, "elastic-shell")
        ):
            rep.add("domain", "elastic-shell, quad4", IMPLEMENTED, where)
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
            elif (
                dom.kind == "ShellDomain"
                and c.tag == "shell_normal_nodal"
                and c.value == 1
            ):
                rep.add("domain setting", c.tag, IMPLEMENTED, where)
            else:
                rep.add("domain setting", c.tag, NOT_SUPPORTED, where)
        try:
            mat = model.material(dom.material)
        except KeyError:
            rep.add(
                "material", dom.material, NOT_SUPPORTED, where, "material not found"
            )
            continue
        kern = material_kernel(mat.type, material_backend)
        if kern is None:
            rep.add(
                "material",
                mat.type,
                NOT_SUPPORTED,
                f"material {mat.name}",
                f"backend {material_backend}",
            )
            continue
        rep.add(
            "material",
            mat.type,
            IMPLEMENTED,
            f"material {mat.name}",
            f"backend {material_backend}",
        )
        for c in mat.params.children:
            if c.tag in kern.param_names:
                if "lc" in c.attrs:
                    rep.add(
                        "material parameter",
                        c.tag,
                        NOT_SUPPORTED,
                        f"material {mat.name}",
                        "load curve on a material parameter",
                    )
            elif c.tag == "density":
                rep.add(
                    "material parameter",
                    "density",
                    NO_EFFECT,
                    f"material {mat.name}",
                    "static analysis",
                )
            else:
                rep.add(
                    "material parameter", c.tag, NOT_SUPPORTED, f"material {mat.name}"
                )
        for name in kern.param_names:
            if mat.params.child(name) is None:
                rep.add(
                    "material parameter",
                    name,
                    NOT_SUPPORTED,
                    f"material {mat.name}",
                    "missing; FEBio defaults are not applied yet",
                )

    for r in model.mesh_data:
        if (
            r.tag == "ElementData"
            and r.attrs.get("type") == "shell thickness"
            and isinstance(r.value, np.ndarray)
        ):
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
            rep.add(
                "mesh", "Edge", IMPLEMENTED, r.attrs.get("name", ""), "as a node set"
            )
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
        _inspect_control(step.control, rep, f"step {step.id}")
        for r in step.loads + step.contact + step.other:
            rep.add(r.tag, r.type or r.tag, NOT_SUPPORTED, f"step {step.id}")
    for r in model.boundary + [b for s in model.steps for b in s.boundary]:
        _inspect_bc(r, model, rep)
    for r in model.loads:
        rep.add("load", r.type or r.tag, NOT_SUPPORTED, r.name or "")
    for r in model.contact:
        rep.add("contact", r.type or r.tag, NOT_SUPPORTED, r.name or "")
    for r in model.other_sections:
        if r.children:
            for c in r.children:
                rep.add(r.tag, c.type or c.tag, NOT_SUPPORTED, c.name or "")
    for lc in model.load_controllers:
        interp = str(lc.params.get("interpolate", "LINEAR")).upper()
        ok = (
            lc.type == "loadcurve"
            and interp == "LINEAR"
            and lc.points is not None
            and len(lc.points) >= 2
        )
        rep.add(
            "load controller",
            f"{lc.type}, {interp}",
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


def _inspect_control(control, rep, where):
    for c in control.children:
        if c.tag == "analysis":
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
        elif c.tag == "time_stepper":
            rep.add(
                "control",
                "time_stepper",
                NOT_SUPPORTED,
                where,
                "automatic time stepping; fixed steps only in this draft",
            )
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
    known = set(_BC_FLAGS.get(r.type, ())) | (
        {"dof", "value", "relative"} if r.type in _BC_PRESCRIBED else set()
    )
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

    name: str
    kind: str  # "SolidDomain" or "ShellDomain"
    material: Any  # MaterialKernel
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
    load_controller: Optional[int]  # id, or None for a constant


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
    directors: Optional[np.ndarray] = None  # (N, 3) shell directors d0, host

    def prescribed_values(self, time: float):
        """(dofs, values) of all prescribed degrees of freedom at `time`."""
        dofs, vals = [], []
        for p in self.prescribed:
            f = (
                1.0
                if p.load_controller is None
                else self.model.load_controller(p.load_controller).value(time)
            )
            dofs.append(p.dofs)
            vals.append(np.full(len(p.dofs), p.scale * f))
        if not dofs:
            return np.zeros(0, np.int64), np.zeros(0)
        return np.concatenate(dofs), np.concatenate(vals)

    def state(self) -> "State":
        """A zero state (reference configuration) for this model."""
        return State(
            time=0.0, u=wp.zeros(self.ndof, dtype=wp.float64, device=self.device)
        )


@dataclass
class State:
    """The solution at one time. FEBio keeps this on nodes and material points."""

    time: float
    u: wp.array  # (ndof,) float64 global displacement vector


def shell_directors(model: Model) -> Optional[np.ndarray]:
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

    From <ElementData type="shell thickness"> or from the domain's
    <shell_thickness> (FEBio ignores a domain value of 0).
    """
    n = block.connectivity.shape[1]
    data = model.element_data("shell thickness", block.name)
    dom = model.domain_of(block.name)
    h_dom = float(dom.params.get("shell_thickness", 0.0))
    if data is not None and h_dom > 0.0:
        # Which one FEBio applies last was not verified, so this is refused
        # rather than guessed.
        raise ValueError(
            f"shell part {block.name!r} has both <shell_thickness> and "
            "shell thickness ElementData; precedence is not settled in this draft"
        )
    if data is not None:
        return np.asarray(data.value, dtype=np.float64).reshape(block.count, n)
    if h_dom > 0.0:
        # FEShellDomainNew::AssignDefaultShellThickness
        return np.full((block.count, n), h_dom)
    raise ValueError(f"shell part {block.name!r} has no thickness")


def compile(
    model: Model, device=None, material_backend: str = "minimal"
) -> CompiledModel:
    """Check coverage, then build the device representation of a Model.

    Raises CoverageError, carrying the report, if the model needs a feature
    that is not yet supported.
    """
    report = inspect(model, material_backend)
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
        kern = material_kernel(mat.type, material_backend)
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
            dofmap = np.concatenate(
                [dof_index[conn][:, :, 3:], dof_index[conn][:, :, :3]], axis=1
            ).reshape(E, 24)
        values = [float(mat.value(name)) for name in kern.param_names]
        nq = len(GAUSS_WEIGHTS)
        gsize = kern.geometry_vector._length_
        # Material axes default to the identity where a model declares them.
        g0 = np.zeros(gsize)
        if gsize == 9:
            g0[:] = np.eye(3).ravel()
        domains.append(
            CompiledDomain(
                name=block.name,
                kind=dom.kind,
                material=kern,
                kernel=make_brick_kernel(kern),
                count=E,
                X=wp.array(X, dtype=wp.vec3d, device=device),
                dofmap=wp.array(dofmap.astype(np.int32), dtype=wp.int32, device=device),
                params=wp.array(
                    [kern.param_vector(*values)], dtype=kern.param_vector, device=device
                ),
                geom=wp.array(
                    np.tile(g0, (E, 1)), dtype=kern.geometry_vector, device=device
                ),
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
        dN=dN,
        gw=gw,
        directors=directors,
    )
