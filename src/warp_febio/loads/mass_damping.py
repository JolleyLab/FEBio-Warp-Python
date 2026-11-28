"""FEBio's `mass damping` body load.

TEMPLATE (dynamics draft, November 2026). Structure for discussion, not final.

FEBio counterpart: FEMassDamping, a body force C v per unit mass on an element
set, with stiffness C / dt per unit mass.
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEMassDamping.cpp

The domains integrate density times shape function times C v, with v
interpolated from the nodes (BodyForce of the elastic domains), which is
C M v with the consistent mass M of the element set. So this load is not a
term with element code of its own: `build_system` gives its C to the system,
which applies it through the mass blocks (System.set_dynamics). The force,
FEBio's Newton stiffness and the exact derivative are in assembly/system.py.

Reserved: an element set that is not a whole part; a load curve on C.
"""

from __future__ import annotations

from ..coverage import IMPLEMENTED, NOT_SUPPORTED

TYPE = "mass damping"


def inspect_load(record, model):
    """Coverage of one mass damping record: (status, note)."""
    unknown = [c.tag for c in record.children if c.tag != "C"]
    if unknown:
        return NOT_SUPPORTED, f"setting(s) not read: {', '.join(unknown)}"
    c = record.child("C")
    if c is not None and "lc" in c.attrs:
        return NOT_SUPPORTED, "load curve on C"
    part = record.attrs.get("elem_set")
    names = [b.name for b in model.mesh.blocks]
    if part is not None and part not in names:
        return NOT_SUPPORTED, f"element set {part!r} is not a whole part"
    return IMPLEMENTED, "through the mass blocks of the part" + ("" if part else "s (all parts)")


def damping_by_part(model) -> dict:
    """{part name: C} summed over the model's mass damping loads. A load
    without an element set acts on every part, as in FEBio."""
    out = {}
    # The loads of the model and, with one step, those of the step
    # (analysis/steps.py, build_system).
    for record in model.loads + (model.steps[0].loads if len(model.steps) == 1 else []):
        if record.type != TYPE:
            continue
        part = record.attrs.get("elem_set")
        for name in [part] if part else [b.name for b in model.mesh.blocks]:
            out[name] = out.get(name, 0.0) + float(record.get("C", 0.0))
    return out
