"""The gradient check case: one hex8 under a follower pressure.

    python examples/run_path/gradient_check_case.py [device]     device: cpu (default) or cuda:0

Runs references/single_element/hex8_pressure.feb, then compares the adjoint
gradient of the top-face displacement with central differences of complete
runs, for every parameter the system's terms expose. Template code.
"""

import os
import sys

import numpy as np
import warp as wp

from warp_febio.analysis.steps import build_system
from warp_febio.io import read_feb
from warp_febio.sensitivity.check import check_parameter_gradients

wp.config.quiet = True
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"  # for a GPU: cuda:0
CASE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "references",
    "single_element",
    "hex8_pressure.feb",
)

system = build_system(read_feb(CASE), device=DEVICE)
top = system.fields.entries_of("displacement", [4, 5, 6, 7])  # nodes 5..8


def output(u):
    """Mean z displacement of the top face, and its derivative."""
    dJ_du = np.zeros_like(u)
    dJ_du[top, 2] = 0.25
    return u[top, 2].mean(), dJ_du


names = {
    "Part1.material": system.terms[0].parameter_names,
    "TopPressure.pressure": ("pressure",),
}
print(f"{'parameter':<32}{'value':>8}{'adjoint':>16}{'central diff.':>16}{'rel. error':>12}")
for row in check_parameter_gradients(system, output):
    label = f"{row.parameter} [{names[row.parameter][row.index[-1]]}]"
    print(f"{label:<32}{row.value:>8.3g}{row.adjoint:>16.9f}{row.finite_difference:>16.9f}{row.relative_error:>12.1e}")
