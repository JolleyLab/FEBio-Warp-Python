# SPDX-FileCopyrightText: Copyright (c) 2026 The FEBio-Warp-Python Authors
# SPDX-License-Identifier: Apache-2.0
"""Implicit dynamics and mass damping (analysis/dynamics.py), against FEBio.

DRAFT (dynamics template, November 2026).

Small closed-form checks of the pieces are in test_dynamics_units.py and
test_mass_damping.py. The references here are the two dynamic cases of
references/small_problems, whose
FEBio results were stored before this code existed:

    bk_mass_damping          one hex8, a pressure stepped on at time zero,
                             mass damping. Exercises the initial
                             accelerations (M a = F at time zero).
    sh_lee_sacks_pressure    36 quad4 shells, isotropic Lee-Sacks, a pressure
                             ramp, mass damping: the valve setting in small.

Both use FEBio's default time integration (alpha_f = alpha_m = 1, beta = 1/4,
gamma = 1/2) and fixed time steps.
"""

import hashlib
import json
import os
import pathlib
import unittest

import numpy as np
import warp as wp
from support import DEVICE, setUpModule, tearDownModule  # noqa: F401  (device for the module)

from warp_febio.analysis.dynamics import new_dynamic_state, update_kinematics
from warp_febio.analysis.steps import build_system, run
from warp_febio.io import read_feb
from warp_febio.solvers.linear.warp_linear import dense

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = ROOT / "references" / "small_problems"
# A checkout of https://github.com/febiosoftware/TestSuite; see test_chordae_elements.py.
SUITE = os.environ.get("FEBIO_TEST_SUITE")


def load(case):
    ref = json.loads((CASES / "febio" / f"{case}.json").read_text())
    deck = CASES / ref["input"]
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == ref["input_sha256"]
    return ref, read_feb(deck)


def record(ref, name):
    rec = next(r for r in ref["records"] if name in r["data"])
    return rec["items"], np.array(rec["values"]), rec["data"]


class FebioAgreementTests(unittest.TestCase):
    def test_solid_with_mass_damping(self):
        """Displacement, velocity and stress over 200 steps. The load is on at
        time zero, so the run starts from FEBio's initial accelerations."""
        ref, model = load("bk_mass_damping")
        system = build_system(model, device=DEVICE)
        system.damping_tangent = "febio"  # FEBio's Newton tangent, to follow its iterations
        results = run(system)
        # FEBio 4.13.0 at e4df3f8 takes 902 equilibrium iterations on this input.
        self.assertEqual(sum(s.iterations for s in results.states[1:]), 902)
        items, logged, names = record(ref, "uz")
        node = model.mesh.node_index(items)[0]
        self.assertEqual(len(results.states), len(logged))
        uz = np.array([s.displacement[node, 2] for s in results.states])
        vz = np.array([s.velocity[node, 2] for s in results.states])
        self.assertLess(abs(uz - logged[:, 0, names.index("uz")]).max() / abs(logged[:, 0, 0]).max(), 1e-10)
        self.assertLess(abs(vz - logged[:, 0, names.index("vz")]).max() / abs(logged[:, 0, 1]).max(), 1e-10)
        _, logged, _ = record(ref, "sx")
        stress = np.stack([s.stress["Part1"] for s in results.states])
        self.assertLess(abs(stress - logged).max() / abs(logged).max(), 1e-10)

    def test_lee_sacks_shell_patch_with_pressure_and_mass_damping(self):
        """The valve setting in small. Agreement is to about 3e-8 here, and
        not to round-off as for the solid: with FEBio's damping stiffness the
        iteration converges linearly and both codes stop at dtol = 1e-7. With
        dtol = 1e-10 in both, the displacement agrees to 6e-13 (checked once
        with FEBio; not a stored reference)."""
        ref, model = load("sh_lee_sacks_pressure")
        system = build_system(model, device=DEVICE)
        system.damping_tangent = "febio"
        results = run(system)
        # FEBio's shell domain adds the damping stiffness with the opposite
        # sign of its solid domain (brick_domain.py). Following it, the
        # iteration count is FEBio's 1397 to within a few; with the sign of
        # the solid it would be 744.
        self.assertAlmostEqual(sum(s.iterations for s in results.states[1:]), 1397, delta=15)
        items, logged, _ = record(ref, "uz")
        nodes = model.mesh.node_index(items)
        u = np.stack([s.displacement[nodes] for s in results.states])
        self.assertEqual(len(u), len(logged))
        self.assertLess(abs(u - logged).max() / abs(logged).max(), 5e-7)
        _, logged, _ = record(ref, "sx")
        stress = np.stack([s.stress["Part1"] for s in results.states])
        self.assertLess(abs(stress - logged).max() / abs(logged).max(), 5e-7)


@unittest.skipUnless(SUITE, "set FEBIO_TEST_SUITE to a checkout of the FEBio test suite")
class SuiteTests(unittest.TestCase):
    """Dynamic problems of the FEBio test suite, against the suite's stored
    step count, iteration count and solution norm (code/linuxGoldStandards.py
    at suite commit 986005f; see SuiteTests in test_chordae_elements.py).

        dy16   trusses with mass, nodal load
        dy07   40 hex8, prescribed displacement

    Both take FEBio's steps and iterations exactly. FEBio ran them with BFGS
    (max_ups 10) at dtol 1e-3, so the norm agrees to that tolerance.

    Not here: sh58 (40 quad4 shells, nodal force, automatic time stepping).
    Its first 14 steps have FEBio's step sizes and iteration counts, and
    displacements equal to the precision of FEBio's plot file. From step 15
    FEBio's line search shortens steps, which this code does not have, so the
    iteration counts, then the step sizes and the final norm (2 percent)
    differ. Checked once with FEBio 4.13.0; docs/design/dynamics.md.
    """

    GOLD = {"dy16": (100, 276, 0.0638959555364, 1e-4), "dy07": (100, 200, 3.60040926788, 1e-5)}

    def test_dynamic_truss_and_solid_problems(self):
        from warp_febio.discretization.fields import FREE

        for name, (steps, iterations, norm, tolerance) in self.GOLD.items():
            with self.subTest(problem=name):
                system = build_system(read_feb(pathlib.Path(SUITE) / "Verify" / f"{name}.feb"), device=DEVICE)
                results, states = run(system, keep_states=True)
                self.assertEqual(len(results.states) - 1, steps)
                self.assertEqual(sum(s.iterations for s in results.states[1:]), iterations)
                mine = float(np.linalg.norm(states[-1].u.numpy()[system.fields.bc == FREE]))
                self.assertLess(abs(mine - norm) / norm, tolerance)


class DampingTangentTests(unittest.TestCase):
    """The three stiffnesses Newton can use for mass damping
    (System.damping_tangent, decision D4) converge to the same solution, in
    very different numbers of iterations."""

    def run_with(self, case, choice):
        _, model = load(case)
        system = build_system(model, device=DEVICE)
        if choice is not None:
            system.damping_tangent = choice
        results = run(system)
        return system, results, sum(s.iterations for s in results.states[1:])

    def test_default_is_the_exact_derivative(self):
        system, _, _ = self.run_with("bk_mass_damping", None)
        self.assertEqual(system.damping_tangent, "exact")

    def test_same_solution_fewer_iterations(self):
        _, febio, n_febio = self.run_with("sh_lee_sacks_pressure", "febio")
        _, consistent, n_consistent = self.run_with("sh_lee_sacks_pressure", "consistent")
        _, exact, n_exact = self.run_with("sh_lee_sacks_pressure", "exact")
        u = febio.states[-1].displacement
        for other in (consistent, exact):
            self.assertLess(abs(other.states[-1].displacement - u).max() / abs(u).max(), 5e-7)
        # 1400, 744 and 325 when this was written.
        self.assertLess(n_exact, 0.6 * n_consistent)
        self.assertLess(n_consistent, 0.6 * n_febio)


class GradientTests(unittest.TestCase):
    def test_gradient_through_one_dynamic_step(self):
        """Adjoint gradients of a displacement after one dynamic step, with
        respect to every parameter of the system, against central
        differences. The step starts from rest with no load at time zero, so
        the state it starts from does not depend on the parameters, and the
        adjoint of the step (the transpose of the exact dynamic tangent) is
        the whole derivative. Gradients over several steps need the reverse
        sweep of docs/design/sensitivity.md and are not built."""
        from warp_febio.sensitivity.check import check_parameter_gradients

        _, model = load("sh_lee_sacks_pressure")
        control = model.steps[0].control
        control.set("time_steps", 1)
        control.set("step_size", 0.002)
        control.child("solver").set("dtol", 1e-11)
        system = build_system(model, device=DEVICE)
        node = system.fields.entries_of("displacement", model.mesh.node_index([25]))[0]

        def output(u):
            dJ = np.zeros_like(u)
            dJ[node, 2] = 1.0
            return float(u[node, 2]), dJ

        rows = check_parameter_gradients(system, output, rel_step=1e-4)
        self.assertGreater(len(rows), 3)
        self.assertGreater(max(abs(r.finite_difference) for r in rows), 0.0)
        for r in rows:
            scale = max(abs(r.finite_difference), 1e-12)
            self.assertLess(abs(r.adjoint - r.finite_difference) / scale, 1e-4, msg=f"{r.parameter}{r.index}")


class TangentTests(unittest.TestCase):
    def test_exact_tangent_is_the_derivative_of_the_dynamic_residual(self):
        """With exact=True the tangent of a dynamic step, mass and damping
        included, is minus the derivative of the residual with respect to
        the displacement at the end of the step. FEBio's Newton tangent
        ("febio") differs from it in the damping part, here for a solid
        (decision D4)."""
        _, model = load("bk_mass_damping")
        system = build_system(model, device=DEVICE)
        rng = np.random.default_rng(1)
        n = system.n
        state0 = system.state()
        for array in (state0.u, state0.v, state0.a):
            array.assign(0.01 * rng.normal(size=(n, 3)))
        u1 = state0.u.numpy() + 0.01 * rng.normal(size=(n, 3))

        def residual(u):
            state = new_dynamic_state(system, state0, 0.1)
            state.u.assign(u)
            update_kinematics(system, state, state0)
            system.update(state)
            return state, system.residual(state).numpy().copy()

        state, _ = residual(u1)
        system.tangent(state, exact=True)
        K = dense(system.matrix())
        system.damping_tangent = "febio"
        system.tangent(state, exact=False)
        K_newton = dense(system.matrix())
        h = 1e-6
        K_fd = np.zeros_like(K)
        for j in range(3 * n):
            du = np.zeros(3 * n)
            du[j] = h
            plus, minus = residual(u1 + du.reshape(n, 3))[1], residual(u1 - du.reshape(n, 3))[1]
            K_fd[:, j] = -(plus - minus).reshape(-1) / (2 * h)
        self.assertLess(abs(K - K_fd).max() / abs(K).max(), 1e-7)
        # gamma / (beta dt) against 1 / dt: a factor of two at the defaults.
        ti = system.time_integration
        # The pressure load also has an exact and a Newton form: FEBio's
        # symmetrized stiffness drops the antisymmetric part, which the
        # symmetric part of the difference leaves out.
        difference = K - K_newton
        difference = 0.5 * (difference + difference.T) / (ti.gamma / ti.beta - 1.0) * 0.1
        system.mass_matrix()  # leaves the mass in the block buffer
        M = dense(system.matrix())  # without the constraints
        np.testing.assert_allclose(difference, 2.0 * M, atol=1e-12)  # C = 2


if __name__ == "__main__":
    wp.config.quiet = True
    unittest.main()
