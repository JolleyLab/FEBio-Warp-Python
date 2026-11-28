"""From a FEBio truss or discrete material record to a uniaxial law.

DRAFT (chordae elements, November 2026). Structure for discussion, not final.

The counterpart of materials/febio.py (`bind_material`) for the materials of
two-node elements. FEBio reads these from two places:

    <Material>   type "linear truss"                      (FETrussMaterial)
                 type "tension-only linear truss"         (FEBio-Warp only)
    <Discrete>   <discrete_material> of type "linear spring",
                 "tension-only linear spring", "nonlinear spring"
                                                          (FEDiscreteElasticMaterial)

Each resolver returns (law, values, issues): the uniaxial model, its parameter
values in the order of `law.param_names`, and findings for the coverage report
under FEBio's names. A material that cannot be evaluated returns law None and
an issue with status "not yet supported"; nothing is dropped silently.

Defaults are FEBio's (FETrussMaterial.cpp, FESpringMaterial.h,
FENonlinearSpring.cpp): truss v = 0.5 and density = 1; linear spring E = 1;
tension-only linear spring E = 0 and slack_length = 0; nonlinear spring
scale = 1 and measure = elongation.
"""

from __future__ import annotations

import numpy as np

from .febio import IMPLEMENTED, NOT_SUPPORTED, MaterialIssue
from .models.linear_spring import LINEAR_SPRING, TENSION_ONLY_LINEAR_SPRING
from .models.linear_truss import LINEAR_TRUSS, TENSION_ONLY_LINEAR_TRUSS
from .models.nonlinear_spring import EXTENSIONS, MEASURES, nonlinear_spring


def _unknown_children(record, known, febio_type, issues):
    for c in record.children:
        if c.tag not in known:
            issues.append(MaterialIssue("material parameter", f"{febio_type}.{c.tag}", NOT_SUPPORTED, "not read"))
        elif "lc" in c.attrs:
            issues.append(
                MaterialIssue("material parameter", f"{febio_type}.{c.tag}", NOT_SUPPORTED, "load curve on a parameter")
            )


def _failed(issues) -> bool:
    return any(i.status == NOT_SUPPORTED for i in issues)


# Truss material type strings and their laws. "tension-only linear truss" is
# not a FEBio type: it has the parameters of "linear truss" and no stress in
# compression (materials/models/linear_truss.py).
TRUSS_LAWS = {"linear truss": LINEAR_TRUSS, "tension-only linear truss": TENSION_ONLY_LINEAR_TRUSS}
FEBIO_WARP_ONLY = "not a FEBio material: FEBio-Warp only, FEBio will not read this input"


def resolve_truss_material(material):
    """A truss material -> (law, [E, v], density, issues)."""
    record = getattr(material, "params", material)
    febio_type = record.attrs.get("type")
    issues = []
    if febio_type not in TRUSS_LAWS:
        issues.append(MaterialIssue("material", str(febio_type), NOT_SUPPORTED, "not a truss material"))
        return None, None, None, issues
    _unknown_children(record, ("E", "v", "density"), febio_type, issues)
    E = record.get("E")
    v = float(record.get("v", 0.5))
    if E is None or not float(E) > 0.0:
        issues.append(MaterialIssue("material parameter", f"{febio_type}.E", NOT_SUPPORTED, "E must be positive"))
    if not -1.0 <= v <= 0.5:
        issues.append(MaterialIssue("material parameter", f"{febio_type}.v", NOT_SUPPORTED, "v must be in [-1, 0.5]"))
    if _failed(issues):
        return None, None, None, issues
    note = FEBIO_WARP_ONLY if febio_type != "linear truss" else ""
    issues.insert(0, MaterialIssue("material", febio_type, IMPLEMENTED, note))
    return TRUSS_LAWS[febio_type], [float(E), v], float(record.get("density", 1.0)), issues


def resolve_discrete_material(record):
    """One <discrete_material> -> (law, parameter values, issues)."""
    febio_type = record.attrs.get("type")
    issues = []
    law, values = None, None
    if febio_type == "linear spring":
        _unknown_children(record, ("E",), febio_type, issues)
        law, values = LINEAR_SPRING, [float(record.get("E", 1.0))]
    elif febio_type == "tension-only linear spring":
        _unknown_children(record, ("E", "slack_length"), febio_type, issues)
        law = TENSION_ONLY_LINEAR_SPRING
        values = [float(record.get("E", 0.0)), float(record.get("slack_length", 0.0))]
    elif febio_type == "nonlinear spring":
        _unknown_children(record, ("scale", "measure", "force"), febio_type, issues)
        measure = str(record.get("measure", "elongation"))
        force = record.child("force")
        if measure not in MEASURES:
            issues.append(MaterialIssue("material parameter", "nonlinear spring.measure", NOT_SUPPORTED, measure))
        if force is None or force.attrs.get("type") != "point":
            kind = None if force is None else force.attrs.get("type")
            issues.append(
                MaterialIssue("material parameter", "nonlinear spring.force", NOT_SUPPORTED, f"function type {kind!r}")
            )
        else:
            # FEPointFunction defaults: linear interpolation, constant extension.
            interpolate = str(force.get("interpolate", "linear")).lower()
            extend = str(force.get("extend", "constant")).lower()
            points = force.child("points")
            pts = [] if points is None else [np.atleast_1d(p.value) for p in points.all("pt")]
            if interpolate != "linear":
                issues.append(
                    MaterialIssue(
                        "material parameter", "nonlinear spring.force.interpolate", NOT_SUPPORTED, interpolate
                    )
                )
            if extend not in EXTENSIONS:
                issues.append(
                    MaterialIssue("material parameter", "nonlinear spring.force.extend", NOT_SUPPORTED, extend)
                )
            if len(pts) < 2:
                issues.append(
                    MaterialIssue("material parameter", "nonlinear spring.force.points", NOT_SUPPORTED, "two or more")
                )
            if not _failed(issues):
                law = nonlinear_spring(len(pts), measure, extend)
                values = [float(record.get("scale", 1.0))]
                for p in pts:
                    values += [float(p[0]), float(p[1])]
    else:
        issues.append(MaterialIssue("discrete material", str(febio_type), NOT_SUPPORTED))
    if _failed(issues):
        return None, None, issues
    issues.insert(0, MaterialIssue("discrete material", febio_type, IMPLEMENTED))
    return law, values, issues
