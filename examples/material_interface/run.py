"""Run the material conformance checks and record the numbers.

    python examples/material_interface/run.py

Writes validation/material_interface/results.json and prints a table. The same
checks run in tests/test_material_interface.py; this script exists so the
figures can be read without running anything.
"""

import json
import pathlib
import sys

import numpy as np
import warp as wp

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from test_material_interface import PARAMETERS, legacy  # noqa: E402
from warp_febio.materials import model as model_api  # noqa: E402
from warp_febio.materials import models  # noqa: E402,F401
from warp_febio.materials.conformance import check_model, sample_states  # noqa: E402

ROWS = [
    ("cauchy_vs_febio", "Cauchy stress vs FEBio closed form"),
    ("tangent_vs_febio", "6x6 spatial tangent vs FEBio closed form"),
    ("cauchy_asymmetry", "Cauchy stress asymmetry"),
    ("tangent_asymmetry", "6x6 tangent asymmetry"),
    ("stress_vs_energy_fd", "Cauchy stress vs finite difference of energy"),
    ("tangent_vs_stress_fd", "Spatial tangent vs finite difference of stress"),
    ("objectivity_stress", "Objectivity, sigma(RF) = R sigma(F) R^T"),
    ("rest_stress", "Stress at F = I and at a rigid rotation"),
    ("pk1_vs_cauchy", "First Piola stress consistent with Cauchy stress"),
    ("previous_stress", "First Piola stress vs previous implementation"),
    ("previous_tangent", "First Piola tangent action vs previous implementation"),
    ("fp32_vs_fp64_stress", "FP32 vs FP64 stress"),
    ("parameter_gradient_vs_fd", "Parameter gradient, tape vs finite difference"),
]


def main():
    wp.config.quiet = True
    F = sample_states()
    H = 0.2 * np.random.default_rng(11).normal(size=F.shape)
    results = {}
    for model in model_api.all_models():
        params = PARAMETERS[model.name]
        metrics = check_model(model, params)
        new = model_api.evaluate(model, F, params, H)
        W, P, dP = legacy(model.name, F, H, params)
        rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())  # noqa: E731
        metrics["previous_energy"] = rel(new["energy"], W)
        metrics["previous_stress"] = rel(new["pk1"], P)
        metrics["previous_tangent"] = rel(new["tangent_action"], dP)
        results[model.name] = {
            "febio_type": model.febio_type,
            "authored": model.authored,
            "parameters": params,
            "source": model.source,
            "metrics": metrics,
        }

    out = ROOT / "validation" / "material_interface"
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "warp": wp.__version__,
        "device": "cpu",
        "precision": "FP64 unless stated",
        "states": int(len(F)),
        "note": "Relative errors, normalized by the largest reference magnitude over the batch.",
        "models": results,
    }
    (out / "results.json").write_text(json.dumps(record, indent=2) + "\n")

    names = list(results)
    print("| Check | " + " | ".join(f"{n} ({results[n]['authored']})" for n in names) + " |")
    print("|---|" + "---|" * len(names))
    for key, label in ROWS:
        print(f"| {label} | " + " | ".join(f"{results[n]['metrics'][key]:.1e}" for n in names) + " |")


if __name__ == "__main__":
    main()
