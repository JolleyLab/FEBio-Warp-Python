# SPDX-FileCopyrightText: Copyright (c) 2026 The FEBio-Warp-Python Authors
# SPDX-License-Identifier: Apache-2.0
"""Small checks of the pieces of a dynamic step, with closed forms.

DRAFT (dynamics template, November 2026).

No FEBio result is needed here: the mass has closed forms, the parameter rules
are FEBio's source, and a body that nothing holds moves in a way the time
integration scheme reproduces exactly. The comparisons with FEBio runs are in
test_dynamics.py.
"""

import pathlib
import unittest

import numpy as np
import warp as wp
from support import DEVICE, setUpModule, tearDownModule  # noqa: F401  (device for the module)

from warp_febio.analysis.dynamics import TimeIntegration, lumped_mass, new_dynamic_state, update_kinematics
from warp_febio.analysis.steps import build_system, inspect_model
from warp_febio.elements.brick8 import consistent_mass
from warp_febio.elements.truss_domain import TrussDomain
from warp_febio.io import read_feb
from warp_febio.model import Record
from warp_febio.solvers.linear.warp_linear import dense
from warp_febio.solvers.nonlinear.newton import NewtonSettings, solve_step

ROOT = pathlib.Path(__file__).resolve().parents[1]
SMALL_MESHES = ROOT / "references" / "small_meshes"

UNIT_CUBE = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64
)


def free_body(name):
    """A small mesh of references/small_meshes made dynamic, with its
    boundary conditions removed: nothing holds it and nothing loads it."""
    model = read_feb(SMALL_MESHES / f"{name}.feb")
    model.boundary = []
    model.loads = []
    model.steps[0].control.set("analysis", "DYNAMIC")
    return model, build_system(model, device=DEVICE)


class TimeIntegrationRules(unittest.TestCase):
    """FEBio's parameter rules (FESolidSolver2::Init)."""

    def settings(self, **values):
        solver = Record("solver")
        for tag, value in values.items():
            solver.set(tag, value)
        return TimeIntegration.from_solver(solver)

    def test_rules(self):
        self.assertEqual(TimeIntegration.from_solver(None), TimeIntegration(1.0, 1.0, 0.25, 0.5))
        # rhoi = 0, the setting of both application inputs.
        self.assertEqual(self.settings(rhoi=0), TimeIntegration(1.0, 2.0, 1.0, 1.5))
        ti = self.settings(rhoi=1)
        self.assertEqual((ti.alpha_f, ti.alpha_m, ti.beta, ti.gamma), (0.5, 0.5, 0.25, 0.5))
        self.assertEqual(self.settings(rhoi=-1, beta=0.3), TimeIntegration(1.0, 1.0, 0.25, 0.5))
        self.assertEqual(self.settings(alpha=1.0, beta=0.3, gamma=0.6), TimeIntegration(1.0, 1.0, 0.3, 0.6))

    def test_a_static_step_has_none(self):
        control = Record("Control")
        control.set("analysis", "STATIC")
        self.assertIsNone(TimeIntegration.from_control(control))

    def test_alpha_f_below_one_is_refused(self):
        model, _ = free_body("hex_block")
        model.steps[0].control.child("solver").set("rhoi", 1)
        report = inspect_model(model)
        self.assertFalse(report.runnable)
        entry = next(e for e in report.entries if e.category == "analysis")
        self.assertIn("alpha_f below 1", entry.note)


class ConsistentMass(unittest.TestCase):
    def test_unit_cube_has_the_closed_form(self):
        """m_ab = rho V / 27 on the diagonal, halved for each coordinate in
        which the two nodes differ: the trilinear hex8 with 2 x 2 x 2 Gauss."""
        m = consistent_mass(UNIT_CUBE[None], 3.0)[0]
        apart = (UNIT_CUBE[:, None, :] != UNIT_CUBE[None, :, :]).sum(axis=2)
        np.testing.assert_allclose(m, 3.0 / 27.0 / 2.0**apart, rtol=1e-13)
        self.assertAlmostEqual(m.sum(), 3.0, delta=1e-13)

    def test_total_is_density_times_volume_for_a_sheared_and_scaled_brick(self):
        A = np.array([[2.0, 0.3, 0.0], [0.0, 1.5, 0.2], [0.1, 0.0, 0.5]])
        X = UNIT_CUBE @ A.T
        m = consistent_mass(X[None], 2.0)[0]
        self.assertAlmostEqual(m.sum(), 2.0 * np.linalg.det(A), delta=1e-12)
        np.testing.assert_allclose(m, m.T, rtol=0, atol=1e-15)

    def test_mass_of_a_solid_and_of_a_shell_model(self):
        """Row sums of the system's mass add up to density times volume. The
        mass matrix is symmetric and, with nothing constrained, positive definite."""
        for name in ("hex_block", "shell_patch"):
            with self.subTest(mesh=name):
                model, system = free_body(name)
                state = system.state()
                system.update(state)
                system.tangent(state)  # the terms write their block positions
                term = system.terms[0]
                volume = consistent_mass(term._X_host, 1.0).sum()
                self.assertAlmostEqual(lumped_mass(system).sum(), term.density * volume, delta=1e-12 * volume)
                M = dense(system.mass_matrix())
                np.testing.assert_allclose(M, M.T, rtol=0, atol=1e-14 * abs(M).max())
                self.assertGreater(np.linalg.eigvalsh(M).min(), 0.0)

    def test_truss_mass_is_in_the_order_of_its_stiffness_blocks(self):
        X = np.array([[[0.0, 0, 0], [2.0, 0, 0]]])
        term = TrussDomain("one", X, np.array([[0, 1]]), 0.5, [[1.0, 0.5]], 2, DEVICE, density=3.0)
        m = 3.0 * 0.5 * 2.0  # rho A L
        np.testing.assert_allclose(term.mass_coefficients(), [m / 3, m / 6, m / 6, m / 3], rtol=1e-14)


class Kinematics(unittest.TestCase):
    def test_update_is_febio_formula(self):
        """FESolidSolver2::UpdateKinematics, for random states."""
        _, system = free_body("hex_block")
        rng = np.random.default_rng(0)
        n = system.n
        state0 = system.state()
        u0, v0, a0, u1 = (rng.normal(size=(n, 3)) for _ in range(4))
        for array, values in ((state0.u, u0), (state0.v, v0), (state0.a, a0)):
            array.assign(values)
        state = new_dynamic_state(system, state0, 0.1)
        state.u.assign(u1)
        update_kinematics(system, state, state0)
        dt, beta, gamma = 0.1, 0.25, 0.5
        a = (u1 - u0) / (beta * dt**2) - v0 / (beta * dt) + (1 - 0.5 / beta) * a0
        np.testing.assert_allclose(state.a.numpy(), a, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(state.v.numpy(), v0 + ((1 - gamma) * a0 + gamma * a) * dt, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(state.a_mass.numpy(), a, rtol=1e-13, atol=1e-13)  # alpha_m = 1

    def test_start_of_a_step_keeps_the_displacement(self):
        """PrepStep: with u = u_n, a = -v_n / (beta dt) + (1 - 1/(2 beta)) a_n."""
        _, system = free_body("hex_block")
        state0 = system.state()
        state0.v.assign(np.full((system.n, 3), 2.0))
        state = new_dynamic_state(system, state0, 0.5)
        update_kinematics(system, state, state0)
        np.testing.assert_allclose(state.a.numpy(), -2.0 / (0.25 * 0.5), rtol=1e-14)
        np.testing.assert_array_equal(state.u.numpy(), state0.u.numpy())


class FreeBody(unittest.TestCase):
    """A body that nothing holds, given a uniform velocity, keeps it: after a
    step u = v0 dt, v = v0, a = 0, with no stress. The stiffness alone is
    singular for such a body (rigid motions); the mass makes the step
    solvable. Solids and shells, where both faces of the shell move."""

    def test_uniform_velocity_is_kept(self):
        v0 = np.array([0.3, -0.2, 0.5])
        for name in ("hex_block", "shell_patch"):
            with self.subTest(mesh=name):
                _, system = free_body(name)
                state0 = system.state()
                state0.v.assign(np.tile(v0, (system.n, 1)))
                state = state0
                for step in range(1, 4):
                    state, iterations = solve_step(system, state, 0.1 * step, NewtonSettings(dtol=1e-10))
                    self.assertLessEqual(iterations, 3)
                np.testing.assert_allclose(state.u.numpy(), np.tile(0.3 * v0, (system.n, 1)), rtol=0, atol=1e-11)
                np.testing.assert_allclose(state.v.numpy(), np.tile(v0, (system.n, 1)), rtol=0, atol=1e-10)
                self.assertLess(abs(state.a.numpy()).max(), 1e-8)
                self.assertLess(abs(system.terms[0].plot("stress")).max(), 1e-9)


if __name__ == "__main__":
    wp.config.quiet = True
    unittest.main()
