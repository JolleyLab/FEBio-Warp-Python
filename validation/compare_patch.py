"""Compare the new synthetic-patch run to the saved Warp1.12.1 prototype field."""

from pathlib import Path
import argparse
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, default=root / "outputs/patch")
args = parser.parse_args()
reference_dir = root / "validation/patch_pressurization"
reference_report = json.loads((reference_dir / "prototype_1_12_1.json").read_text())
new_report = json.loads((args.run / "report.json").read_text())
# Exact generated mesh identity is required before using matching vertex indices.
assert new_report["mesh_sha256"] == reference_report["mesh_sha256"]
for key in ("material", "integration", "pressure_Pa"):
    assert new_report[key] == reference_report[key], f"Mismatched settings: {key}"
reference = np.loadtxt(
    reference_dir / "prototype_displacement.csv", delimiter=",", skiprows=1
)
state = np.load(args.run / "state.npz")
assert np.array_equal(reference[:, 0], np.arange(len(state["rest"])))
displacement_mm = (state["positions"] - state["rest"]) * 1000.0
difference = displacement_mm - reference[:, 1:]
result = dict(
    comparison="identical synthetic fixture; Warp1.12.1 to cleaned Warp1.19 implementation",
    relative_nodal_L2=float(
        np.linalg.norm(difference) / np.linalg.norm(reference[:, 1:])
    ),
    maximum_vector_difference_mm=float(np.linalg.norm(difference, axis=1).max()),
    max_displacement_mm=new_report["max_displacement_mm"],
)
# A bounded float32 migration regression threshold, not an accuracy tolerance vs FEBio.
assert result["relative_nodal_L2"] < 1e-4, result
print(json.dumps(result, indent=2))
