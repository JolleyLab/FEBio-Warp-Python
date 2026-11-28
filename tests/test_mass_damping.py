# SPDX-FileCopyrightText: Copyright (c) 2026 The FEBio-Warp-Python Authors
# SPDX-License-Identifier: Apache-2.0
"""The mass damping load, in small: a closed form, and what the coverage
report accepts.

DRAFT (dynamics template, November 2026). The comparisons with FEBio runs
(bk_mass_damping, sh_lee_sacks_pressure) are in test_dynamics.py.
"""

import pathlib
import unittest

import numpy as np
import warp as wp
from support import DEVICE, setUpModule, tearDownModule  # noqa: F401  (device for the module)

from warp_febio.analysis.steps import build_system, inspect_model
from warp_febio.assembly.system import DAMPING_TANGENTS
from warp_febio.io import read_feb
from warp_febio.loads import mass_damping
from warp_febio.solvers.nonlinear.newton import NewtonSettings, solve_step

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASE = ROOT / "references" / "small_problems" / "inputs" / "bk_mass_damping.feb"
SMALL_MESHES = ROOT / "references" / "small_meshes"


def damped_free_body(name, C):
    """A small mesh with nothing holding it, dynamic, with mass damping C on its part."""
    model = read_feb(SMALL_MESHES / f"{name}.feb")
    model.boundary = []
    model.loads = [next(r for r in read_feb(CASE).loads if r.type == mass_damping.TYPE)]
    model.loads[0].set("C", C)
    model.steps[0].control.set("analysis", "DYNAMIC")
    return build_system(model, device=DEVICE)


class Decay(unittest.TestCase):
    """A free body with a uniform velocity under mass damping stays stress
    free and slows down uniformly: M a + C M v = 0. With FEBio's default time
    integration (beta = 1/4, gamma = 1/2), starting from a = 0, one step gives

        v1 = v0 / (1 + C dt / 2),   u1 = v0 dt - C dt^2 v1 / 4,   a1 = -C v1,

    whatever stiffness Newton uses for the damping. The three choices differ
    in how many iterations they take to get there."""

    def test_one_step_has_the_closed_form_for_every_damping_tangent(self):
        C, dt = 3.0, 0.1
        v0 = np.array([0.3, -0.2, 0.5])
        v1 = v0 / (1.0 + C * dt / 2.0)
        for name in ("hex_block", "shell_patch"):
            counts = {}
            for choice in DAMPING_TANGENTS:
                with self.subTest(mesh=name, damping_tangent=choice):
                    system = damped_free_body(name, C)
                    system.damping_tangent = choice
                    state0 = system.state()
                    state0.v.assign(np.tile(v0, (system.n, 1)))
                    state, counts[choice] = solve_step(
                        system, state0, dt, NewtonSettings(dtol=1e-13, etol=0.0, max_refs=200)
                    )
                    # The derivative converges at once; the other two
                    # converge linearly and stop a little short of round-off.
                    tol = 1e-10 if choice == "exact" else 1e-8
                    np.testing.assert_allclose(state.v.numpy(), np.tile(v1, (system.n, 1)), rtol=0, atol=tol)
                    np.testing.assert_allclose(
                        state.u.numpy(), np.tile(v0 * dt - C * dt**2 * v1 / 4.0, (system.n, 1)), rtol=0, atol=tol
                    )
                    np.testing.assert_allclose(state.a.numpy(), np.tile(-C * v1, (system.n, 1)), rtol=0, atol=100 * tol)
            # The derivative solves this linear problem at once. FEBio's form
            # needs more iterations, and most on a shell, where it has the
            # opposite sign (decision D4 of docs/design/dynamics.md).
            self.assertLessEqual(counts["exact"], 2)
            self.assertGreater(counts["febio"], counts["exact"])
            if name == "shell_patch":
                self.assertGreater(counts["febio"], counts["consistent"])
            else:
                self.assertEqual(counts["febio"], counts["consistent"])


class Coverage(unittest.TestCase):
    def test_damping_by_part(self):
        model = read_feb(CASE)
        self.assertEqual(mass_damping.damping_by_part(model), {"Part1": 2.0})
        # Without an element set the load acts on every part, as in FEBio.
        del model.loads[-1].attrs["elem_set"]
        self.assertEqual(mass_damping.damping_by_part(model), {"Part1": 2.0})

    def test_what_is_accepted(self):
        model = read_feb(CASE)
        load = next(r for r in model.loads if r.type == mass_damping.TYPE)
        status, _ = mass_damping.inspect_load(load, model)
        self.assertEqual(status, "implemented")
        load.attrs["elem_set"] = "some_elements"
        status, note = mass_damping.inspect_load(load, model)
        self.assertEqual(status, "not yet supported")
        self.assertIn("not a whole part", note)
        load.attrs["elem_set"] = "Part1"
        load.child("C").attrs["lc"] = "1"
        self.assertEqual(mass_damping.inspect_load(load, model)[0], "not yet supported")

    def test_no_effect_in_a_static_analysis(self):
        model = read_feb(CASE)
        model.steps[0].control.set("analysis", "STATIC")
        entry = next(e for e in inspect_model(model).entries if e.febio_name == "mass damping")
        self.assertEqual(entry.status, "no effect here")


if __name__ == "__main__":
    wp.config.quiet = True
    unittest.main()
