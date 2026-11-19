"""Evaluate a prescribed deformation: this is not a spatial solid solve."""

import argparse
import numpy as np
from warp_febio.materials.lee_sacks import evaluate
from warp_febio.io.results import write_report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", default="cpu")
parser.add_argument("--output", default="outputs/material_point/report.json")
args = parser.parse_args()
# Synthetic coefficients in Pa, chosen for demonstration only; not tissue calibration.
parameters = dict(c0=200000.0, c1=2968400.0, c2=0.2661, k=5000000.0)
F = np.array([np.diag([1.1, 0.98, 0.98])])
response = evaluate(F, parameters, device=args.device)
write_report(
    args.output,
    dict(
        case="Lee-Sacks prescribed material point",
        device=args.device,
        parameters_Pa_except_c2=parameters,
        F=F.tolist(),
        energy_J_m3=response["energy"].tolist(),
        pk1_Pa=response["pk1"].tolist(),
        cauchy_Pa=response["cauchy"].tolist(),
        J=response["J"].tolist(),
    ),
)
print("Saved", args.output)
