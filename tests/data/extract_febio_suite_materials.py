"""Extract the parameter sets the FEBio test suite uses for the materials
FEBio-Warp implements.

    python tests/data/extract_febio_suite_materials.py /path/to/TestSuite

Writes tests/data/febio_suite_materials.json: for each FEBio material type we
have a model for, every distinct set of parameter values found in the suite,
with the problems that use it. Materials nested inside other materials (for
example a neo-Hookean solid inside a mixture or a biphasic material) count,
because the constitutive routine is the same.

Only numbers are stored, with the suite commit they came from. The input files
themselves are not copied. Parameter sets that are not plain numbers (mapped or
math-valued parameters) are counted and skipped.

The suite is https://github.com/febiosoftware/TestSuite.
"""

import json
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET

# FEBio type -> (FEBio-Warp model name, parameter tags the model takes)
MODELS = {
    "neo-Hookean": ("neo_hookean", ("E", "v")),
    "isotropic elastic": ("isotropic_elastic", ("E", "v")),
    "isotropic Lee-Sacks": ("lee_sacks_isotropic", ("c0", "c1", "c2", "k")),
    "HGO unconstrained": (
        "hgo_unconstrained",
        ("c", "k1", "k2", "kappa", "gamma", "k"),
    ),
}


def main(suite):
    suite = pathlib.Path(suite)
    found = {name: {} for name, _ in MODELS.values()}
    skipped = {name: 0 for name, _ in MODELS.values()}
    uses = {name: 0 for name, _ in MODELS.values()}
    for path in sorted((suite / "Verify").glob("*.feb")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        section = root.find("Material")
        if section is None:
            continue
        for element in section.iter():
            if element.get("type") not in MODELS:
                continue
            name, tags = MODELS[element.get("type")]
            uses[name] += 1
            values, plain = [], True
            for tag in tags:
                child = element.find(tag)
                try:
                    if child is None or child.get("type") in ("map", "math"):
                        raise ValueError
                    values.append(float(child.text))
                except (ValueError, TypeError):
                    plain = False
                    break
            if not plain:
                skipped[name] += 1
                continue
            entry = found[name].setdefault(tuple(values), {"problems": [], "load_curve": False})
            if path.stem not in entry["problems"]:
                entry["problems"].append(path.stem)
            entry["load_curve"] |= any(element.find(tag).get("lc") is not None for tag in tags)
    commit = subprocess.run(
        ["git", "-C", str(suite), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    record = {
        "source": "https://github.com/febiosoftware/TestSuite",
        "suite_commit": commit,
        "note": "Parameter values only. See extract_febio_suite_materials.py.",
        "models": {
            name: {
                "febio_type": febio_type,
                "parameters": list(MODELS[febio_type][1]),
                "uses_in_suite": uses[name],
                "uses_skipped_not_plain_numbers": skipped[name],
                "sets": [
                    {
                        "values": list(values),
                        "problems": entry["problems"][:6],
                        "problem_count": len(entry["problems"]),
                        "load_curve": entry["load_curve"],
                    }
                    for values, entry in sorted(found[name].items())
                ],
            }
            for febio_type, (name, _) in MODELS.items()
        },
    }
    out = pathlib.Path(__file__).with_name("febio_suite_materials.json")
    out.write_text(json.dumps(record, indent=1) + "\n")
    for name, info in record["models"].items():
        print(
            f"{name}: {info['uses_in_suite']} uses, {len(info['sets'])} distinct sets,"
            f" {info['uses_skipped_not_plain_numbers']} skipped"
        )


if __name__ == "__main__":
    main(sys.argv[1])
