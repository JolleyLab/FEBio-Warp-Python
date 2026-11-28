# SPDX-FileCopyrightText: Copyright (c) 2026 The FEBio-Warp-Python Authors
# SPDX-License-Identifier: Apache-2.0
"""Details of the truss, spring and nodal load code that follow FEBio, each
found in a review against the FEBio source at e4df3f8.

DRAFT (chordae elements, November 2026).
"""

import pathlib
import unittest

import numpy as np
import warp as wp
from support import DEVICE, setUpModule, tearDownModule  # noqa: F401  (device for the module)

from warp_febio.analysis.steps import build_system, inspect_model, run
from warp_febio.elements.truss_domain import TrussDomain
from warp_febio.io import read_feb
from warp_febio.model import Record

INPUTS = pathlib.Path(__file__).resolve().parents[1] / "references" / "small_problems" / "inputs"


class TrussMass(unittest.TestCase):
    def test_mass_uses_the_geometric_length_not_the_rest_length(self):
        """FELinearTrussDomain::ElementMassMatrix uses m_a0 * m_L0. A chord
        given a longer rest length is slack, not heavier."""
        X = np.array([[[0.0, 0, 0], [2.0, 0, 0]]])
        args = ("one", X, np.array([[0, 1]]), 0.5, [[1.0, 0.5]], 2, DEVICE)
        geometric = TrussDomain(*args, density=3.0)
        longer = TrussDomain(*args, density=3.0, rest_length=[2.6])
        np.testing.assert_array_equal(longer.mass_blocks(), geometric.mass_blocks())
        self.assertAlmostEqual(geometric.mass_blocks().sum(), 3.0 * 0.5 * 2.0, delta=1e-14)


class DiscreteEntries(unittest.TestCase):
    def entries(self, model):
        return [e for e in inspect_model(model).entries if e.febio_name == "discrete"]

    def test_a_wire_is_not_run_as_a_spring(self):
        """FEBioDiscreteSection: type "wire" makes a deformable spring
        domain, which is not built here; type "spring" or none is a spring."""
        model = read_feb(INPUTS / "ch_spring_chords.feb")
        self.assertTrue(inspect_model(model).runnable)
        section = next(s for s in model.other_sections if s.tag == "Discrete")
        link = section.all("discrete")[0]
        link.attrs["type"] = "spring"
        self.assertTrue(inspect_model(model).runnable)
        link.attrs["type"] = "wire"
        report = inspect_model(model)
        self.assertFalse(report.runnable)
        self.assertEqual([e.status for e in self.entries(model)].count("not yet supported"), 1)

    def test_settings_inside_discrete_are_not_ignored(self):
        model = read_feb(INPUTS / "ch_spring_chords.feb")
        section = next(s for s in model.other_sections if s.tag == "Discrete")
        section.all("discrete")[0].children.append(Record("some_setting", value=1))
        self.assertFalse(inspect_model(model).runnable)


class TrussDensity(unittest.TestCase):
    def density_entry(self, analysis):
        model = read_feb(INPUTS / "ch_truss_chords.feb")
        truss = next(m for m in model.materials if m.type == "linear truss")
        truss.params.set("density", 2.0)
        model.steps[0].control.set("analysis", analysis)
        return next(e for e in inspect_model(model).entries if e.febio_name == "density" and truss.name in e.where)

    def test_reported_by_analysis_type(self):
        self.assertEqual(self.density_entry("STATIC").status, "no effect here")
        self.assertEqual(self.density_entry("DYNAMIC").status, "implemented")


class NodalLoadDefault(unittest.TestCase):
    def test_no_scale_is_febio_default_of_zero(self):
        """FENodalDOFLoad: m_scale = 0 unless the input gives it."""
        model = read_feb(INPUTS / "ch_truss_chords.feb")
        load = next(r for r in model.loads if r.type == "nodal_load")
        load.children = [c for c in load.children if c.tag != "scale"]
        results = run(build_system(model, device=DEVICE))
        self.assertLess(abs(results.states[-1].displacement).max(), 1e-12)


if __name__ == "__main__":
    wp.config.quiet = True
    unittest.main()
