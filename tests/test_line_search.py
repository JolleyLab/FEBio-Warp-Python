# SPDX-FileCopyrightText: Copyright (c) 2026 The FEBio-Warp-Python Authors
# SPDX-License-Identifier: Apache-2.0
"""FEBio's line search (solvers/nonlinear/newton.py, line_search) on scalar
functions, where what it should do can be stated exactly.

DRAFT (nonlinear solve, November 2026). In a run it is checked against the
FEBio test suite: problems sh56 and sh58 take FEBio's iteration counts only
with it, and di03 and sh06 run only with it (test_febio_suite.py).
"""

import unittest

from warp_febio.assembly.system import SolveError
from warp_febio.solvers.nonlinear.newton import NewtonSettings, line_search


def tracked(function):
    calls = []

    def evaluate(s):
        calls.append(s)
        return function(s)

    return evaluate, calls


class LineSearch(unittest.TestCase):
    def test_defaults_are_febio(self):
        s = NewtonSettings()
        self.assertEqual((s.lstol, s.lsmin, s.lsiter, s.ls_check_jacobians), (0.9, 0.01, 5, False))

    def test_a_good_newton_step_is_taken_whole(self):
        """r(1) small against r(0): one evaluation, s = 1."""
        evaluate, calls = tracked(lambda s: 2.0 * (1.0 - s) + 0.1)
        self.assertEqual(line_search(NewtonSettings(), 2.1, evaluate), 1.0)
        self.assertEqual(calls, [1.0])

    def test_an_overshooting_step_is_shortened(self):
        """r(s) = r0 (1 - 4 s) changes sign at s = 1/4: the full step
        overshoots four times. The step taken is shorter and meets lstol."""
        r0 = 3.0
        evaluate, calls = tracked(lambda s: r0 * (1.0 - 4.0 * s))
        s = line_search(NewtonSettings(), r0, evaluate)
        self.assertLess(s, 1.0)
        self.assertGreater(s, 0.0)
        self.assertLessEqual(abs(r0 * (1.0 - 4.0 * s)) / r0, 0.9)
        self.assertEqual(calls[0], 1.0)
        self.assertEqual(calls[-1], s)  # the model is left at the step taken

    def test_after_lsiter_updates_the_best_step_seen_is_taken(self):
        """A function no step improves: after lsiter updates the step with
        the smallest |r| among those tried is taken, and evaluated last."""
        r = lambda s: 5.0 + 20.0 * s * s
        evaluate, calls = tracked(r)
        s = line_search(NewtonSettings(lsmin=0.0), 5.0, evaluate)
        tried = calls[:-1]
        self.assertEqual(len(tried), 5)
        self.assertEqual(s, min(tried, key=lambda t: abs(r(t))))
        self.assertEqual(calls[-1], s)

    def test_a_step_below_lsmin_becomes_one_half(self):
        """FELineSearch: 'the line search step is simply set to 0.5'."""
        evaluate, calls = tracked(lambda s: 1.0 + 1e4 * s)
        s = line_search(NewtonSettings(), 1.0, evaluate)
        self.assertEqual(s, 0.5)
        self.assertEqual(calls[-1], 0.5)

    def test_check_jacobians_halves_a_step_whose_update_fails(self):
        def function(s):
            if s > 0.3:
                raise SolveError("negative Jacobian")
            return 1.0 - s

        evaluate, calls = tracked(function)
        s = line_search(NewtonSettings(ls_check_jacobians=True), 1.0, evaluate)
        self.assertEqual(s, 0.25)
        self.assertEqual(calls, [1.0, 0.5, 0.25])
        # Without the flag the failure is the caller's to handle: the step is cut.
        with self.assertRaises(SolveError):
            line_search(NewtonSettings(), 1.0, function)


if __name__ == "__main__":
    unittest.main()
