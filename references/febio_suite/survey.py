"""Survey the FEBio test suite against what FEBio-Warp can run.

Reads every input file in ``Verify/`` of a checkout of
https://github.com/febiosoftware/TestSuite and records, per problem, the
features it needs: material types (including nested ones), element types,
domain formulations, contact types, loads, discrete materials and analysis
type. Each problem is then put in one of four classes by its materials:

- ``supported``     every deformable material is one FEBio-Warp has
- ``needs_material`` in scope, but at least one material is missing
- ``out_of_scope``  fluid, biphasic, multiphasic, solute or FSI module
- ``rigid_only``    no deformable solid material

The classification is by material only. A ``supported`` problem may still
need an element, contact or solver feature that FEBio-Warp lacks; those are
what the per-problem tags are for.

Usage (standard library only, nothing in the suite is executed)::

    git clone https://github.com/febiosoftware/TestSuite
    python references/febio_suite/survey.py TestSuite \
        --out references/febio_suite/coverage.json

FEBio input format reference:
https://help.febio.org/docs/FEBioUser-4-7/UM47-Chapter-3.html
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

# Material types FEBio-Warp implements today. Extend this set when a model is
# added, then re-run the survey so coverage.json and docs/design/verification.md move
# together. Names are FEBio's own type strings.
SUPPORTED_MATERIALS = frozenset(
    {
        "neo-Hookean",
        "isotropic elastic",
        "isotropic Lee-Sacks",
    }
)

# Rigid bodies carry no constitutive model, so they never count as missing.
RIGID = "rigid body"

# Modules outside the planned scope of FEBio-Warp (solid mechanics only).
OUT_OF_SCOPE_MODULES = frozenset(
    {"fluid", "fluid-FSI", "biphasic", "multiphasic", "solute", "fluid-solutes"}
)

# Child elements of a material that carry a ``type`` attribute but are not
# material models: fiber and axis generators, and parameter value generators.
NOT_A_MATERIAL_TAG = frozenset({"fiber", "mat_axis"})
VALUE_GENERATOR_TYPES = frozenset({"map", "math", "const"})


def _material_types(mat: ET.Element) -> set[str]:
    """Type strings of a material and every material nested inside it.

    A nested material is any descendant that has a ``type`` attribute and
    child elements of its own, for example the ``solid`` entries of a solid
    mixture. Fiber generators and mapped parameters are excluded.
    """
    found = {mat.get("type")}

    def walk(node: ET.Element) -> None:
        for child in node:
            if child.tag in NOT_A_MATERIAL_TAG:
                continue
            ctype = child.get("type")
            if ctype and ctype not in VALUE_GENERATOR_TYPES and len(child) > 0:
                found.add(ctype)
            walk(child)

    walk(mat)
    found.discard(None)
    return found


def _material_parameter_flags(material_section: ET.Element) -> tuple[bool, bool]:
    """Whether any material parameter is driven by a load curve or mapped."""
    load_curve = any(e.get("lc") for e in material_section.iter())
    mapped = any(e.get("type") == "map" for e in material_section.iter())
    return load_curve, mapped


def survey_file(path: Path) -> dict:
    root = ET.parse(path).getroot()
    module = root.find("Module")
    module_type = module.get("type") if module is not None else None

    materials: set[str] = set()
    top_level: set[str] = set()
    load_curve = mapped = False
    section = root.find("Material")
    if section is not None:
        for mat in section.findall("material"):
            top_level.add(mat.get("type"))
            materials |= _material_types(mat)
        load_curve, mapped = _material_parameter_flags(section)

    def types(xpath: str) -> list[str]:
        return sorted({e.get("type") for e in root.iterfind(xpath) if e.get("type")})

    record = {
        "problem": path.stem,
        "format": root.get("version"),
        "module": module_type,
        "analysis": sorted({e.text.strip() for e in root.iter("analysis") if e.text}),
        "materials": sorted(materials),
        "elements": types(".//Elements"),
        "domains": sorted(
            set(types(".//SolidDomain"))
            | set(types(".//ShellDomain"))
            | set(types(".//BeamDomain"))
        ),
        "contact": types(".//contact"),
        "discrete": types(".//discrete_material"),
        "body_loads": types(".//body_load"),
        "surface_loads": types(".//surface_load"),
        "material_load_curve": load_curve,
        "material_mapped": mapped,
    }

    deformable = materials - {RIGID}
    if module_type in OUT_OF_SCOPE_MODULES:
        record["class"] = "out_of_scope"
    elif not deformable:
        record["class"] = "rigid_only"
    elif deformable <= SUPPORTED_MATERIALS:
        record["class"] = "supported"
    else:
        record["class"] = "needs_material"
    record["missing_materials"] = (
        sorted(deformable - SUPPORTED_MATERIALS)
        if record["class"] == "needs_material"
        else []
    )
    return record


def _has(key: str, *values: str):
    return lambda p: bool(set(p[key]) & set(values))


# Features of the valve and device models, as predicates on a problem record.
APPLICATION_FEATURES = {
    "quad4 shell elements": _has("elements", "quad4"),
    "hex8 solid elements": _has("elements", "hex8"),
    "tet4 solid elements": _has("elements", "tet4"),
    "dynamic analysis": _has("analysis", "DYNAMIC"),
    "neo-Hookean": _has("materials", "neo-Hookean"),
    "isotropic elastic": _has("materials", "isotropic elastic"),
    "isotropic Lee-Sacks": _has("materials", "isotropic Lee-Sacks"),
    "Holzapfel-Gasser-Ogden (uncoupled)": _has("materials", "Holzapfel-Gasser-Ogden"),
    "HGO unconstrained": _has("materials", "HGO unconstrained"),
    "elastic-shell domain": _has("domains", "elastic-shell"),
    "three-field-solid domain": _has("domains", "three-field-solid"),
    "linear-beam domain": _has("domains", "linear-beam"),
    "linear-truss domain": _has("domains", "linear-truss"),
    "contact potential": _has("contact", "contact potential"),
    "sliding-elastic contact": _has("contact", "sliding-elastic"),
    "sticky contact": _has("contact", "sticky"),
    "nonlinear spring": _has("discrete", "nonlinear spring"),
    "linear spring": _has("discrete", "linear spring"),
    "mass damping": _has("body_loads", "mass damping"),
    "pressure load": _has("surface_loads", "pressure"),
    "load curve on a material parameter": lambda p: p["material_load_curve"],
    "mapped material parameter": lambda p: p["material_mapped"],
}


def _suite_commit(suite: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(suite), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def survey(suite: Path) -> dict:
    files = sorted((suite / "Verify").glob("*.feb"))
    problems = [survey_file(f) for f in files]

    classes = collections.Counter(p["class"] for p in problems)

    # A material "unlocks" a problem when it is the only one missing.
    unlocks = collections.Counter(
        p["missing_materials"][0] for p in problems if len(p["missing_materials"]) == 1
    )

    in_scope = [p for p in problems if p["class"] != "out_of_scope"]

    def count(pred, pool=problems) -> int:
        return sum(1 for p in pool if pred(p))

    summary = {
        "problems": len(problems),
        "format_4_0": count(lambda p: p["format"] == "4.0"),
        "classes": dict(sorted(classes.items())),
        "single_material_unlocks": dict(unlocks.most_common()),
        # Features the driving applications rely on. Each entry is
        # [problems in the whole suite, problems within scope, in-scope names
        # (first few)]. docs/design/verification.md quotes the in-scope numbers.
        "application_features": {
            name: [
                count(pred),
                count(pred, in_scope),
                [p["problem"] for p in in_scope if pred(p)][:6],
            ]
            for name, pred in APPLICATION_FEATURES.items()
        },
    }
    return {
        "suite": "https://github.com/febiosoftware/TestSuite",
        "suite_commit": _suite_commit(suite),
        "supported_materials": sorted(SUPPORTED_MATERIALS),
        "summary": summary,
        "problems": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("suite", type=Path, help="path to a TestSuite checkout")
    parser.add_argument("--out", type=Path, help="write the full survey as JSON")
    args = parser.parse_args()

    result = survey(args.suite)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "problems"}, indent=1))


if __name__ == "__main__":
    main()
