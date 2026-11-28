# SPDX-FileCopyrightText: Copyright (c) 2026 The FEBio-Warp-Python Authors
# SPDX-License-Identifier: Apache-2.0
"""Problems of the FEBio test suite, against the values the suite stores.

DRAFT (verification, November 2026).

The suite (https://github.com/febiosoftware/TestSuite) runs FEBio with
-task=test and stores, for each problem, the number of time steps, the number
of equilibrium iterations and the L2 norm of the solution vector at the free
degrees of freedom after the last step (code/linuxGoldStandards.py). The
values are read from the checkout that FEBIO_TEST_SUITE points to, so nothing
of the suite is copied here; the tests are skipped without it.

Of the 549 inputs of the suite at commit 986005f, the coverage report accepts
23. They fall into three groups.

EXACT: FEBio ran these with full Newton and its line search did not act, or
its quasi-Newton path happened to take the same iterations. This code takes
FEBio's steps and iterations exactly and reproduces the norm.

NORM: FEBio ran these with BFGS (max_ups 10) or its line search shortened
steps, so iteration counts and automatic step sizes differ. The solution norm
at the end still agrees, to FEBio's convergence tolerance.

Not here, with the reason:
    bt01, di01, di02   in test_chordae_elements.py (bt01 needs an iterative
                       solve: its first tangent is singular)
    dy07, dy16         in test_dynamics.py
    di03, sh06         a negative Jacobian without a line search
    sh58               follows FEBio for 14 steps, until FEBio's line search
                       acts (docs/design/dynamics.md)
    sh52, mi16         the solution at the end is zero, so the stored norm is
                       what the iteration leaves at its tolerance and not a
                       value to compare. In mi16 a pressure follows a load
                       curve that returns to zero: FEBio's stored norm is
                       4.9e-4 (its final displacement is 7e-5 of the peak
                       during the run, inside its dtol of 1e-3) and this code
                       ends at 2e-8. At the first step, with the load on, the
                       two agree to 5e-7 of the peak (checked once against
                       FEBio 4.13.0).
"""

import ast
import os
import pathlib
import re
import unittest

import numpy as np
import warp as wp
from support import DEVICE, full_tier, setUpModule, tearDownModule  # noqa: F401  (device for the module)

from warp_febio.analysis.steps import build_system, inspect_model, run
from warp_febio.discretization.fields import FREE
from warp_febio.io import read_feb

SUITE = os.environ.get("FEBIO_TEST_SUITE")

# problem: tolerance on the solution norm, and what it exercises.
EXACT = {
    "co62": (1e-6, "contact potential between two blocks; the reference input for bk_contact_potential_patch"),
    "mi06": (1e-9, "hex8, static"),
    "mi24": (1e-9, "hex8, static"),
    "sh54": (1e-9, "quad4 shells, static"),
}
NORM = {
    "mi02": (1e-5, "FEBio: BFGS"),
    "mi25": (1e-5, "FEBio: BFGS"),
    "mi31": (1e-5, "FEBio: BFGS, 61 steps there and 15 here"),
    "sh02": (1e-5, "quad4 shells; FEBio: BFGS"),
    "sh03": (1e-5, "quad4 shells; FEBio: BFGS"),
    "sh04": (1e-5, "quad4 shells; FEBio: BFGS"),
    "sh05": (1e-5, "quad4 shells; FEBio: BFGS"),
    "sh11": (1e-5, "quad4 shells; FEBio: BFGS"),
    "sh56": (1e-8, "quad4 shells; FEBio: full Newton, 59 iterations there and 64 here"),
}


def stored_values():
    """{problem: (steps, iterations, norm)} from the suite's own file."""
    text = (pathlib.Path(SUITE) / "code" / "linuxGoldStandards.py").read_text()
    out = {}
    for match in re.finditer(r"'(\w+)':\s*(\[[^\]]*\])", text):
        row = ast.literal_eval(match.group(2))
        try:
            out[match.group(1)] = (int(row[1]), int(row[2]), float(row[6]))
        except (ValueError, IndexError):
            continue  # entries of another shape (other modules of the suite)
    return out


@unittest.skipUnless(SUITE, "set FEBIO_TEST_SUITE to a checkout of the FEBio test suite")
class SuiteProblems(unittest.TestCase):
    def solve(self, name):
        model = read_feb(pathlib.Path(SUITE) / "Verify" / f"{name}.feb")
        self.assertTrue(inspect_model(model).runnable)
        system = build_system(model, device=DEVICE)
        results, states = run(system, keep_states=True)
        norm = float(np.linalg.norm(states[-1].u.numpy()[system.fields.bc == FREE]))
        return len(results.states) - 1, sum(s.iterations for s in results.states[1:]), norm

    def test_steps_iterations_and_norm(self):
        stored = stored_values()
        for name, (tolerance, _) in EXACT.items():
            with self.subTest(problem=name):
                steps, iterations, norm = stored[name]
                mine = self.solve(name)
                self.assertEqual(mine[:2], (steps, iterations))
                self.assertLess(abs(mine[2] - norm) / norm, tolerance)

    @full_tier("nine more suite problems compared on the solution norm; about 20 s on CPU")
    def test_norm_where_febio_takes_another_path(self):
        stored = stored_values()
        for name, (tolerance, _) in NORM.items():
            with self.subTest(problem=name):
                _, _, norm = stored[name]
                self.assertLess(abs(self.solve(name)[2] - norm) / norm, tolerance)


if __name__ == "__main__":
    wp.config.quiet = True
    unittest.main()
