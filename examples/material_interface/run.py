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

from test_material_interface import (  # noqa: E402
    GEOMETRY,
    PARAMETERS,
    PILOT,
    closed_form_for,
    legacy,
)

from warp_febio.materials import model as model_api  # noqa: E402
from warp_febio.materials import models  # noqa: E402,F401
from warp_febio.materials.conformance import check_model, sample_states  # noqa: E402

ROWS = [
    ("cauchy_vs_febio", "Cauchy stress vs FEBio closed form"),
    ("tangent_vs_febio", "6x6 spatial tangent vs FEBio closed form"),
    ("combined_vs_actions", "One-pass tangent vs six tangent actions"),
    ("stress_vs_energy_fd", "Stress vs finite difference of energy"),
    ("tangent_vs_stress_fd", "Tangent vs finite difference of stress"),
    ("objectivity_stress", "Objectivity"),
    ("parameter_gradient_vs_fd", "Tape: energy gradient w.r.t. parameters"),
    ("stress_parameter_gradient_vs_fd", "Tape: stress gradient w.r.t. parameters"),
    ("stress_deformation_gradient_vs_fd", "Tape: stress gradient w.r.t. F"),
    ("geometry_gradient_vs_fd", "Tape: energy gradient w.r.t. material axes"),
    ("fp32_vs_fp64_stress", "FP32 vs FP64 stress"),
    ("previous_stress", "Stress vs pilot implementation"),
]


def main():
    wp.config.quiet = True
    F = sample_states()
    H = 0.2 * np.random.default_rng(11).normal(size=F.shape)
    results = {}
    for model in model_api.all_models():
        params = PARAMETERS[model.name]
        metrics = check_model(
            model,
            params,
            closed_form_for(model.name),
            geometry=GEOMETRY.get(model.name),
        )
        if model.name in PILOT:
            new = model_api.evaluate(model, F, params, H)
            W, P, dP = legacy(model.name, F, H, params)
            rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
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
        cells = [(f"{results[n]['metrics'][key]:.1e}" if key in results[n]["metrics"] else "n/a") for n in names]
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
