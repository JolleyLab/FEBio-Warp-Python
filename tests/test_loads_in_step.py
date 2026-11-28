# SPDX-FileCopyrightText: Copyright (c) 2026 The FEBio-Warp-Python Authors
# SPDX-License-Identifier: Apache-2.0
"""A load written inside the single step of a model.

DRAFT (run path template, November 2026).

FEBio activates such a load when the step starts (FEAnalysis::Activate). With
one step that is the whole analysis, so it is built as a load of the model.
"""

import pathlib
import unittest

import numpy as np
import warp as wp
from support import DEVICE, setUpModule, tearDownModule  # noqa: F401  (device for the module)

from warp_febio.analysis.steps import build_system, inspect_model, run
from warp_febio.io import read_feb

INPUT = pathlib.Path(__file__).resolve().parents[1] / "references" / "small_meshes" / "two_material_pressure.feb"


class LoadsInAStep(unittest.TestCase):
    def moved(self):
        model = read_feb(INPUT)
        model.steps[0].loads, model.loads = model.loads, []
        return model

    def test_same_result_as_a_load_of_the_model(self):
        in_model = run(build_system(read_feb(INPUT), device=DEVICE))
        model = self.moved()
        entry = next(e for e in inspect_model(model).entries if e.category == "load")
        self.assertEqual(entry.status, "implemented")
        self.assertIn("step 1", entry.where)
        in_step = run(build_system(model, device=DEVICE))
        self.assertEqual(len(in_step.states), len(in_model.states))
        self.assertGreater(abs(in_model.states[-1].displacement).max(), 0.0)
        for a, b in zip(in_step.states, in_model.states):
            np.testing.assert_array_equal(a.displacement, b.displacement)

    def test_refused_with_several_steps(self):
        """Activation by step is not built: with two steps the load of a
        step is not treated as a load of the model."""
        model = self.moved()
        model.steps.append(model.steps[0])
        report = inspect_model(model)
        self.assertFalse(report.runnable)
        entry = next(e for e in report.entries if e.category == "load")
        self.assertEqual(entry.status, "not yet supported")


if __name__ == "__main__":
    wp.config.quiet = True
    unittest.main()
