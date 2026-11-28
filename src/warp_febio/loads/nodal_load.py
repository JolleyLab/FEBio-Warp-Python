"""Loads on the nodes of a node set, as terms: nodal_load and nodal_force.

DRAFT (chordae elements, November 2026). Added because the FEBio test suite
truss and spring problems are loaded this way (bt01 by a nodal_load, di03 by
a nodal_force). They are two of the small loads the design notes list
(nodal_load, nodal_force, body force).

FEBio counterpart
    FENodalDOFLoad (type "nodal_load"): FENodalLoad::LoadVector adds `scale`,
    times its load curve, to the residual at the chosen degree of freedom of
    every node of the set. FENodalForce (type "nodal_force") adds the vector
    `value`, times its load curve, at the displacement degrees of freedom.
    Neither has a stiffness.
    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FENodalLoad.cpp
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FENodalForce.cpp

Same as FEBio
    nodal_load: the parameters `dof`, `scale` (with a load curve), `relative`.
    nodal_force: the parameters `value` (with a load curve), `relative`,
    `shell_bottom`.

Different, and why
    The scale is a device array with one value per node, filled with the
    value of the input, so it can carry a gradient node by node. FEBio's
    `scale` may also be a map over the nodes; reading one is not implemented,
    but the array is where it would go. A force per free-edge node is how
    FEBio FINESSE (Laurence et al., arXiv 2407.09629) emulates chordae that
    cannot be seen in the image, so the forces on those nodes are natural
    unknowns of an inverse problem. A nodal force is
    also the port through which another tool (Newton, for example) can apply
    forces to the model and receive sensitivities back (decision S9 of
    docs/design/sensitivity.md).

Not implemented
    relative = 1; a value given as a map over the nodes; shell_bottom = 1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ..assembly.system import State, Term
from ..coverage import IMPLEMENTED, NOT_SUPPORTED
from ..discretization.fields import COMPONENT

vec3d = wp.vec3d

TYPE = "nodal_load"
FORCE_TYPE = "nodal_force"
_SETTINGS = ("dof", "scale", "relative")
_FORCE_SETTINGS = ("value", "relative", "shell_bottom")


@wp.kernel
def _add_nodal_load(
    entries: wp.array(dtype=wp.int32),
    direction: vec3d,
    scale: wp.array(dtype=wp.float64),
    factor: wp.array(dtype=wp.float64),
    residual: wp.array(dtype=vec3d),
):
    """residual += scale_i * (load curve value) * direction at each node."""
    i = wp.tid()
    wp.atomic_add(residual, entries[i], scale[i] * factor[0] * direction)


def inspect_load(record, model):
    """Coverage of one nodal_load or nodal_force record: (status, note)."""
    if record.type == FORCE_TYPE:
        unknown = [c.tag for c in record.children if c.tag not in _FORCE_SETTINGS]
        if unknown:
            return NOT_SUPPORTED, f"setting(s) not read: {', '.join(unknown)}"
        if record.get("relative", 0) or record.get("shell_bottom", 0):
            return NOT_SUPPORTED, "relative = 1 or shell_bottom = 1"
        if np.size(record.get("value", 0.0)) != 3:
            return NOT_SUPPORTED, "value is not a vector of three numbers"
        return IMPLEMENTED, ""
    unknown = [c.tag for c in record.children if c.tag not in _SETTINGS]
    if unknown:
        return NOT_SUPPORTED, f"setting(s) not read: {', '.join(unknown)}"
    if record.get("relative", 0):
        return NOT_SUPPORTED, "relative = 1"
    if record.get("dof") not in COMPONENT:
        return NOT_SUPPORTED, f"dof {record.get('dof')!r}"
    return IMPLEMENTED, ""


class NodalLoad(Term):
    """FEBio's nodal_load (one degree of freedom) or nodal_force (a vector)
    on the nodes of a node set."""

    internal = False

    def __init__(self, record, model, fields, directors, device):
        self.name = record.name or record.type
        self.model = model
        self.device = device
        nodes = model.mesh.node_set_indices(record.attrs["node_set"])
        if record.type == FORCE_TYPE:
            # nodal_force: the vector `value`. Held as a unit scale times a
            # fixed direction vector, so the kernel is shared; the gradient
            # with respect to `scale` is that of the load magnitude.
            variable = "displacement"
            value = record.child("value")
            direction, magnitude = np.asarray(value.value, dtype=np.float64), 1.0
        else:
            variable, component = COMPONENT[record.get("dof")]
            value = record.child("scale")
            direction, magnitude = np.zeros(3), float(value.value)
            direction[component] = 1.0
        entries = fields.entries_of(variable, nodes)
        self.entries = wp.array(entries[entries >= 0].astype(np.int32), dtype=wp.int32, device=device)
        self.direction = vec3d(*direction)
        lc = value.attrs.get("lc")
        self.load_controller = int(lc) if lc else None
        self.scale = wp.array(
            np.full(len(self.entries), magnitude), dtype=wp.float64, device=device, requires_grad=True
        )
        self.factor = wp.array([1.0], dtype=wp.float64, device=device)

    def update(self, state: State) -> int:
        if self.load_controller is not None:
            value = self.model.load_controller(self.load_controller).value(state.time)
            self.factor.assign([float(value)])
        return 0

    def residual(self, state: State, history: Any, out: wp.array) -> None:
        wp.launch(
            _add_nodal_load,
            dim=len(self.entries),
            inputs=[self.entries, self.direction, self.scale, self.factor],
            outputs=[out],
            device=self.device,
        )

    def parameters(self) -> dict:
        return {"scale": self.scale}
