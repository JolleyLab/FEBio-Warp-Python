"""Verify a tiny linear elastic solid using Warp FEM, on CPU or CUDA."""

import argparse
from warp_febio_pilot.backends.warp_fem_elasticity import run
from warp_febio_pilot.io.results import write_report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", default="cpu")
parser.add_argument("--output", default="outputs/fem_solid/report.json")
args = parser.parse_args()
report = run(args.device)
write_report(args.output, report)
print(report)
