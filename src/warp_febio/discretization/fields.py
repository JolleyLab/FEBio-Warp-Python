"""Field table: the unknowns of a model, numbered by (node, variable).

TEMPLATE (run path draft, November 2026). Structure for discussion, not final.

FEBio counterpart
    DOFS and FEBioMech::GetVariableName register variables by name
    ("displacement", "shell displacement", "rotation", ...), and
    FESolver::InitEquations gives every free degree of freedom an equation
    number, fixed ones -1 and prescribed ones a coded negative number.
    https://github.com/febiosoftware/FEBio/blob/develop/FECore/DOFS.h
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEBioMech.cpp
    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FESolver.cpp

Same as FEBio
    The variable names and the three states of a component (free, fixed,
    prescribed).

Different, and why
    An unknown here is an *entry*: one (node, variable) pair holding a
    3-vector. Every FEBioMech variable the first targets use is a 3-vector, so
    the solution is one `wp.array(dtype=vec3d)` over entries and the global
    matrix has 3x3 blocks throughout (warp.sparse stores fixed-size blocks,
    and the contact draft already assembles 3x3 blocks). Constrained
    components stay in the system and are masked, so the sparsity pattern
    depends on the mesh only.

Reserved, not implemented
    "rotation" (beams) and rigid-body variables fit the same table. Their
    update is not additive, which is why `UPDATE_RULE` exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# FEBio variable names, and the component tags FEBio uses for them in input
# files (x_dof, sx_dof, <dof>z</dof>, ...).
VARIABLES = ("displacement", "shell displacement")
COMPONENT = {
    "x": ("displacement", 0),
    "y": ("displacement", 1),
    "z": ("displacement", 2),
    "sx": ("shell displacement", 0),
    "sy": ("shell displacement", 1),
    "sz": ("shell displacement", 2),
}
# How a solution increment is applied to each variable. The Newton loop looks
# the rule up instead of assuming addition; "rotation" would compose through
# the exponential map, as FEElasticBeamDomain does.
UPDATE_RULE = {"displacement": "add", "shell displacement": "add"}

FREE, FIXED, PRESCRIBED = 0, 1, 2

# Boundary conditions read by this template (the same four as the fixture of
# pull request 5).
_FIXED_BC = {
    "zero displacement": ("x", "y", "z"),
    "zero shell displacement": ("sx", "sy", "sz"),
}
_PRESCRIBED_BC = ("prescribed displacement", "prescribed shell displacement")


@dataclass
class Prescribed:
    """One prescribed-displacement condition: components, scale and load curve."""

    entries: np.ndarray  # (n,) entry index
    component: int  # 0, 1 or 2
    scale: float
    load_controller: int | None  # FEBio load controller id, or None


@dataclass
class FieldTable:
    """Which (node, variable) pairs exist, and the state of each component."""

    variables: tuple
    entry: np.ndarray  # (N, n_variables) entry index, or -1
    count: int  # number of entries
    bc: np.ndarray  # (count, 3) FREE, FIXED or PRESCRIBED
    prescribed: list = field(default_factory=list)

    def entries_of(self, variable: str, nodes=None) -> np.ndarray:
        """Entry index of `variable` at `nodes` (all nodes if None)."""
        col = self.entry[:, self.variables.index(variable)]
        return col if nodes is None else col[nodes]

    @property
    def free(self) -> np.ndarray:
        """(count, 3) float mask: 1 where a component is solved for."""
        return (self.bc == FREE).astype(np.float64)

    def prescribed_values(self, model, time: float) -> np.ndarray:
        """(count, 3) target value of every prescribed component at `time`.

        Load curves are evaluated on the host, as a handful of numbers per
        step; the result is written to the device once per step.
        """
        target = np.zeros((self.count, 3))
        for p in self.prescribed:
            f = 1.0 if p.load_controller is None else model.load_controller(p.load_controller).value(time)
            target[p.entries, p.component] = p.scale * f
        return target


def build_field_table(model) -> FieldTable:
    """Field table of a Model: displacement at every node, shell displacement
    at the nodes of shell parts, and the boundary conditions of the input."""
    mesh = model.mesh
    N = len(mesh.nodes)
    active = np.zeros((N, len(VARIABLES)), dtype=bool)
    active[:, 0] = True
    for block in mesh.blocks:
        if model.domain_of(block.name).kind == "ShellDomain":
            active[np.unique(block.connectivity), 1] = True
    entry = np.full(active.shape, -1, dtype=np.int32)
    entry[active] = np.arange(active.sum(), dtype=np.int32)
    count = int(active.sum())

    table = FieldTable(
        variables=VARIABLES,
        entry=entry,
        count=count,
        bc=np.full((count, 3), FREE, dtype=np.int32),
    )
    # Global conditions and those of each step. Per-step activation belongs to
    # the step driver; with one step, as here, the two are the same.
    for r in model.boundary + [b for s in model.steps for b in s.boundary]:
        nodes = mesh.node_set_indices(r.attrs["node_set"])
        if r.type in _FIXED_BC:
            for tag in _FIXED_BC[r.type]:
                if r.get(tag + "_dof", 0):
                    variable, comp = COMPONENT[tag]
                    e = table.entries_of(variable, nodes)
                    table.bc[e[e >= 0], comp] = FIXED
        elif r.type in _PRESCRIBED_BC:
            variable, comp = COMPONENT[r.get("dof")]
            e = table.entries_of(variable, nodes)
            e = e[e >= 0]
            value = r.child("value")
            lc = value.attrs.get("lc")
            table.bc[e, comp] = PRESCRIBED
            table.prescribed.append(Prescribed(e, comp, float(value.value), int(lc) if lc else None))
    return table
