"""Truss and spring elements (chordae) in a run, against FEBio and with gradients.

Three levels:

1. One term at a time, against FEBio's formulas written out in NumPy from
   FELinearTrussDomain.cpp, FESpringMaterial.cpp and FENonlinearSpring.cpp,
   and against finite differences of the term's own residual.
2. Whole inputs through the run path, against FEBio's results for the same
   inputs (references/small_problems, FEBio 4.13 built from source).
3. Gradients of a chord tension and of a displacement with respect to every
   parameter of the system, by the adjoint, against central differences of
   FEBio-Warp itself. FEBio has no counterpart for these.
"""

import hashlib
import importlib.util
import json
import os
import pathlib
import unittest

import numpy as np
import warp as wp
from support import DEVICE, setUpModule, tearDownModule  # noqa: F401  (device for the module)

from warp_febio.analysis.steps import build_system, run
from warp_febio.assembly.system import State
from warp_febio.discretization.fields import FREE
from warp_febio.elements.discrete_domain import DiscreteDomain
from warp_febio.elements.line2 import sum_axial_force
from warp_febio.elements.truss_domain import TrussDomain
from warp_febio.io import read_feb
from warp_febio.materials.discrete import chord_curve_closed_form, resolve_discrete_material
from warp_febio.materials.models.linear_spring import (
    LINEAR_SPRING,
    SMOOTH_TENSION_ONLY_LINEAR_SPRING,
    TENSION_ONLY_LINEAR_SPRING,
)
from warp_febio.materials.models.linear_truss import TENSION_ONLY_LINEAR_TRUSS
from warp_febio.materials.models.nonlinear_spring import nonlinear_spring
from warp_febio.sensitivity.adjoint import solve_on_tape
from warp_febio.sensitivity.check import check_parameter_gradients

wp.config.quiet = True

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = ROOT / "references" / "small_problems"
SMALL_MESHES = ROOT / "references" / "small_meshes"
# A checkout of https://github.com/febiosoftware/TestSuite. Its inputs are not
# copied into this repository, so the tests that use them are skipped without it.
SUITE = os.environ.get("FEBIO_TEST_SUITE")

# The chord curve of the valve models: zero up to a slack elongation, then a
# straight line. Here slack 0.4 and slope 1, with scale 60.
CURVE = [60.0, 0.0, 0.0, 0.4, 0.0, 1.4, 1.0]


def random_elements(n, seed=0):
    """Node positions and displacements for n two-node elements that share nodes."""
    rng = np.random.default_rng(seed)
    N = n + 3
    X = rng.uniform(-1.0, 1.0, (N, 3)) * 3.0
    u = rng.uniform(-0.3, 0.3, (N, 3))
    pairs = np.stack([rng.integers(0, N, n), rng.integers(0, N, n)], axis=1)
    pairs[pairs[:, 0] == pairs[:, 1], 1] = (pairs[pairs[:, 0] == pairs[:, 1], 0] + 1) % N
    return X, u, pairs.astype(np.int32)


def evaluate(term, u):
    """Residual (N, 3) and dense tangent (3N, 3N) of one term at displacement u."""
    n = term.n
    blocks = term.block_count
    term.bind_blocks(
        wp.zeros(blocks, dtype=wp.int32, device=DEVICE),
        wp.zeros(blocks, dtype=wp.int32, device=DEVICE),
        wp.zeros(blocks, dtype=wp.mat33d, device=DEVICE),
    )
    state = State(0.0, wp.array(u, dtype=wp.vec3d, device=DEVICE))
    assert term.update(state) == 0
    R = wp.zeros(n, dtype=wp.vec3d, device=DEVICE)
    term.residual(state, None, R)
    term.linearize(state, None)
    K = np.zeros((n, 3, n, 3))
    np.add.at(K, (term.rows.numpy(), slice(None), term.cols.numpy(), slice(None)), term.vals.numpy())
    return R.numpy(), K.reshape(3 * n, 3 * n)


def residual_only(term, u):
    state = State(0.0, wp.array(u, dtype=wp.vec3d, device=DEVICE))
    term.update(state)
    R = wp.zeros(term.n, dtype=wp.vec3d, device=DEVICE)
    term.residual(state, None, R)
    return R.numpy()


def finite_difference_tangent(term, u, h=1e-6):
    """K = -dR/du by central differences of the term's residual."""
    n = term.n
    K = np.zeros((3 * n, 3 * n))
    for j in range(3 * n):
        d = np.zeros(3 * n)
        d[j] = h
        d = d.reshape(n, 3)
        K[:, j] = -(residual_only(term, u + d) - residual_only(term, u - d)).reshape(-1) / (2 * h)
    return K


def scatter(n, pairs, per_element_force, per_element_block):
    """FEBio's assembly of a two-node element: +f on the first node and -f on
    the second in the residual; [[K, -K], [-K, K]] in the stiffness."""
    R = np.zeros((n, 3))
    K = np.zeros((n, 3, n, 3))
    for (a, b), f, k in zip(pairs, per_element_force, per_element_block):
        R[a] += f
        R[b] -= f
        K[a, :, a] += k
        K[b, :, b] += k
        K[a, :, b] -= k
        K[b, :, a] -= k
    return R, K.reshape(3 * n, 3 * n)


class TrussTermTests(unittest.TestCase):
    """TrussDomain against FELinearTrussDomain's expressions."""

    def setUp(self):
        self.X, self.u, self.pairs = random_elements(40, seed=1)
        rng = np.random.default_rng(2)
        self.area = rng.uniform(0.3, 1.5, len(self.pairs))
        self.E, self.v = 150.0, 0.3
        self.term = TrussDomain(
            "chords",
            self.X[self.pairs],
            self.pairs,
            self.area,
            [[self.E, self.v]],
            len(self.X),
            DEVICE,
            poisson_ratio=self.v,
        )

    def febio(self):
        x = self.X + self.u
        forces, blocks, stress, stretch = [], [], [], []
        for (a, b), A in zip(self.pairs, self.area):
            L = np.linalg.norm(self.X[b] - self.X[a])
            r = x[b] - x[a]
            l = np.linalg.norm(r)
            n = r / l
            lam = l / L
            tau = self.E * np.log(lam)  # FELinearTrussMaterial::Stress
            V = L * A
            T = tau * V / l
            k = V / (l * l) * (self.E - 2.0 * tau)  # ElementStiffness
            forces.append(T * n)  # ElementInternalForces, first node
            blocks.append(k * np.outer(n, n) + (T / l) * np.eye(3))
            J = lam ** (1.0 - 2.0 * self.v)  # Update
            stress.append(tau / J * np.outer(n, n))
            stretch.append(lam)
        return forces, blocks, np.array(stress), np.array(stretch)

    def test_matches_febio_source(self):
        forces, blocks, stress, stretch = self.febio()
        R_ref, K_ref = scatter(len(self.X), self.pairs, forces, blocks)
        R, K = evaluate(self.term, self.u)
        np.testing.assert_allclose(R, R_ref, rtol=0, atol=1e-11 * abs(R_ref).max())
        np.testing.assert_allclose(K, K_ref, rtol=0, atol=1e-11 * abs(K_ref).max())
        np.testing.assert_allclose(self.term.plot("stress")[:, 0], stress, rtol=0, atol=1e-11 * abs(stress).max())
        np.testing.assert_allclose(self.term.plot("truss stretch"), stretch, rtol=1e-13)
        # The network has members in tension and in compression: FEBio's
        # linear truss has no cut-off.
        N = self.term.plot("axial force")
        self.assertGreater((N > 0).sum(), 5)
        self.assertGreater((N < 0).sum(), 5)

    def test_stiffness_is_derivative_of_internal_force(self):
        _, K = evaluate(self.term, self.u)
        K_fd = finite_difference_tangent(self.term, self.u)
        np.testing.assert_allclose(K, K_fd, rtol=0, atol=1e-7 * abs(K).max())

    def test_rest_length_defaults_to_geometry_and_scales_the_stretch(self):
        L = np.linalg.norm(self.X[self.pairs[:, 1]] - self.X[self.pairs[:, 0]], axis=1)
        np.testing.assert_allclose(self.term.rest_length.numpy(), L, rtol=1e-15)
        evaluate(self.term, self.u)
        before = self.term.plot("truss stretch").copy()
        self.term.rest_length.assign(1.1 * L)
        evaluate(self.term, self.u)
        np.testing.assert_allclose(self.term.plot("truss stretch"), before / 1.1, rtol=1e-13)

    def test_mass_blocks_are_febio_consistent_mass(self):
        m = self.term.mass_blocks()
        # FEBio's mass uses the geometric length (m_L0), whatever the rest length.
        L = np.linalg.norm(self.X[self.pairs[:, 1]] - self.X[self.pairs[:, 0]], axis=1)
        self.term.rest_length.assign(1.2 * self.term.rest_length.numpy())
        np.testing.assert_array_equal(self.term.mass_blocks(), m)
        np.testing.assert_allclose(m[:, 0, 0], self.term.density * self.area * L / 3.0, rtol=1e-14)
        np.testing.assert_allclose(m[:, 0, 1], self.term.density * self.area * L / 6.0, rtol=1e-14)

    def test_tension_only_truss(self):
        """The tension-only truss (FEBio-Warp only): FEBio's linear truss where
        a member is stretched, nothing where it is not, an exact tangent."""
        args = (self.X[self.pairs], self.pairs, self.area, [[self.E, self.v]], len(self.X), DEVICE)
        slack = TrussDomain("chords", *args, model=TENSION_ONLY_LINEAR_TRUSS, poisson_ratio=self.v)
        R, K = evaluate(slack, self.u)
        N = slack.plot("axial force")
        evaluate(self.term, self.u)
        N_febio = self.term.plot("axial force")
        np.testing.assert_allclose(N, np.where(N_febio > 0, N_febio, 0.0), rtol=1e-14)
        self.assertGreater((N == 0).sum(), 5)
        self.assertGreater((N > 0).sum(), 5)
        np.testing.assert_allclose(K, finite_difference_tangent(slack, self.u), rtol=0, atol=1e-7 * abs(K).max())
        # A slack member has no stress and no stiffness at all.
        np.testing.assert_array_equal(slack.plot("stress")[N == 0], 0.0)

    def test_zero_length_truss_reports_a_status(self):
        X = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        term = TrussDomain("one", X[None], np.array([[0, 1]]), 1.0, [[1.0, 0.5]], 2, DEVICE)
        collapsed = State(0.0, wp.array([[0.0, 0, 0], [-1.0, 0, 0]], dtype=wp.vec3d, device=DEVICE))
        self.assertNotEqual(term.update(collapsed), 0)


def smooth_tension_only(dl, p):
    """Force and slope of the smooth tension-only linear spring, on the host."""
    E, slack, w = p
    x = dl - slack
    if x >= w:
        return E * x, E
    if x <= -w:
        return 0.0, 0.0
    return E * (x + w) ** 2 / (4.0 * w), E * (x + w) / (2.0 * w)


class SpringTermTests(unittest.TestCase):
    """DiscreteDomain against FEBio's spring materials."""

    def laws(self):
        return {
            "linear spring": (LINEAR_SPRING, [[15.0], [40.0]], lambda dl, p: (p[0] * dl, p[0])),
            "tension-only linear spring": (
                TENSION_ONLY_LINEAR_SPRING,
                [[15.0, 0.4], [40.0, 0.1]],
                lambda dl, p: (p[0] * (dl - p[1]), p[0]) if dl - p[1] >= 0 else (0.0, 0.0),
            ),
            # FEBio-Warp only; the reference is the law's own closed form.
            "smooth tension-only linear spring": (
                SMOOTH_TENSION_ONLY_LINEAR_SPRING,
                [[15.0, 0.4, 0.3], [40.0, 0.1, 0.5]],
                smooth_tension_only,
            ),
            "nonlinear spring": (
                nonlinear_spring(3, "elongation", "extrapolate"),
                [CURVE, [30.0, 0.0, 0.0, 0.2, 0.0, 1.2, 1.0]],
                lambda dl, p: (p[0] * (dl - p[3]), p[0]) if dl >= p[3] else (0.0, 0.0),
            ),
        }

    def build(self, law, rows, seed=3):
        X, u, pairs = random_elements(40, seed=seed)
        region = np.arange(len(pairs)) % len(rows)
        term = DiscreteDomain("springs", law, X[pairs], pairs, rows, len(X), DEVICE, region=region)
        return term, X, u, pairs, region

    def test_force_and_stiffness_match_febio_source(self):
        for name, (law, rows, febio_law) in self.laws().items():
            with self.subTest(law=name):
                term, X, u, pairs, region = self.build(law, rows)
                x = X + u
                forces, blocks, signed = [], [], []
                for (a, b), g in zip(pairs, region):
                    L0 = np.linalg.norm(X[b] - X[a])
                    r = x[b] - x[a]
                    Lt = np.linalg.norm(r)
                    e = r / Lt
                    F, E = febio_law(Lt - L0, rows[g])
                    forces.append(F * e)  # FESpringMaterial::Force
                    blocks.append((E - F / Lt) * np.outer(e, e) + (F / Lt) * np.eye(3))  # ::Stiffness
                    signed.append(F)
                R_ref, K_ref = scatter(len(X), pairs, forces, blocks)
                R, K = evaluate(term, u)
                np.testing.assert_allclose(R, R_ref, rtol=0, atol=1e-11 * abs(R_ref).max())
                np.testing.assert_allclose(K, K_ref, rtol=0, atol=1e-11 * abs(K_ref).max())
                np.testing.assert_allclose(term.plot("discrete element signed force"), signed, rtol=0, atol=1e-11)
                if name != "linear spring":
                    # Both branches of a tension-only law are exercised.
                    self.assertGreater((np.array(signed) > 0).sum(), 3)
                    self.assertGreater((np.array(signed) == 0).sum(), 3)

    def test_stiffness_is_derivative_of_force(self):
        for name, (law, rows, _) in self.laws().items():
            with self.subTest(law=name):
                term, _, u, _, _ = self.build(law, rows)
                _, K = evaluate(term, u)
                np.testing.assert_allclose(K, finite_difference_tangent(term, u), rtol=0, atol=1e-7 * abs(K).max())

    def test_tension_only_linear_spring_is_the_three_point_curve(self):
        """FEBio's closed form and the curve of the valve inputs are one law:
        E = scale times the slope, slack_length = the knot."""
        closed, _, u, _, _ = self.build(TENSION_ONLY_LINEAR_SPRING, [[60.0, 0.4]])
        curve, _, _, _, _ = self.build(nonlinear_spring(3, "elongation", "extrapolate"), [CURVE])
        R1, K1 = evaluate(closed, u)
        R2, K2 = evaluate(curve, u)
        np.testing.assert_allclose(R1, R2, rtol=0, atol=1e-12 * abs(R1).max())
        np.testing.assert_allclose(K1, K2, rtol=0, atol=1e-12 * abs(K1).max())

    def test_zero_length_spring_follows_febio_guard(self):
        X = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        term = DiscreteDomain("one", LINEAR_SPRING, X[None], np.array([[0, 1]]), [[7.0]], 2, DEVICE)
        R, K = evaluate(term, np.array([[0.0, 0, 0], [-1.0, 0, 0]]))
        self.assertTrue(np.isfinite(K).all())
        np.testing.assert_allclose(R, 0.0, atol=0)
        # if (Lt == 0) { F = 0; Lt = 1; e = vec3d(1, 1, 1); } gives E e (x) e.
        np.testing.assert_allclose(K[:3, :3], 7.0 * np.ones((3, 3)), rtol=1e-14)


class SmoothTensionOnlySpringTests(unittest.TestCase):
    """The smooth tension-only linear spring (FEBio-Warp only): FEBio's
    tension-only linear spring outside a band around the slack length, a
    continuous stiffness inside it, and a two-sided gradient at the corner."""

    # Values with exact binary forms, so that an elongation equal to the slack
    # length, or at an edge of the band, is so to the last bit.
    E, SLACK, W = 15.0, 0.5, 0.0625

    def one_spring(self, law, row, elongation):
        """One spring along x, stretched by `elongation`."""
        X = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        term = DiscreteDomain("one", law, X[None], np.array([[0, 1]]), [row], 2, DEVICE)
        state = State(0.0, wp.array([[0.0, 0, 0], [elongation, 0, 0]], dtype=wp.vec3d, device=DEVICE))
        return term, state

    def force(self, law, row, elongation):
        term, state = self.one_spring(law, row, elongation)
        self.assertEqual(term.update(state), 0)
        return float(term.data["axial_force"].numpy()[0]), float(term.data["axial_tangent"].numpy()[0])

    def test_is_febio_tension_only_spring_outside_the_band_and_at_zero_width(self):
        w = self.W
        for dl in (-0.25, 0.125, self.SLACK - w, self.SLACK + w, 0.75, 2.0):
            febio = self.force(TENSION_ONLY_LINEAR_SPRING, [self.E, self.SLACK], dl)
            self.assertEqual(self.force(SMOOTH_TENSION_ONLY_LINEAR_SPRING, [self.E, self.SLACK, w], dl), febio)
        for dl in (-0.25, 0.484375, self.SLACK, 0.515625, 2.0):
            febio = self.force(TENSION_ONLY_LINEAR_SPRING, [self.E, self.SLACK], dl)
            self.assertEqual(self.force(SMOOTH_TENSION_ONLY_LINEAR_SPRING, [self.E, self.SLACK, 0.0], dl), febio)

    def test_force_and_stiffness_are_continuous_and_bounded_in_the_band(self):
        w = self.W
        dl = np.linspace(self.SLACK - 1.5 * w, self.SLACK + 1.5 * w, 301)
        X = np.zeros((len(dl), 2, 3))
        X[:, 1, 0] = 1.0
        pairs = np.arange(2 * len(dl), dtype=np.int32).reshape(-1, 2)
        u = np.zeros((2 * len(dl), 3))
        u[1::2, 0] = dl
        state = State(0.0, wp.array(u, dtype=wp.vec3d, device=DEVICE))
        values = {}
        for law, row in (
            (SMOOTH_TENSION_ONLY_LINEAR_SPRING, [self.E, self.SLACK, w]),
            (TENSION_ONLY_LINEAR_SPRING, [self.E, self.SLACK]),
        ):
            term = DiscreteDomain("band", law, X, pairs, [row], 2 * len(dl), DEVICE)
            self.assertEqual(term.update(state), 0)
            values[law.name] = (term.data["axial_force"].numpy().copy(), term.data["axial_tangent"].numpy().copy())
        F, k = values["smooth_tension_only_linear_spring"]
        F_febio, k_febio = values["tension_only_linear_spring"]
        # The stiffness rises from 0 to E without a jump; FEBio's jumps by E.
        self.assertLess(abs(np.diff(k)).max(), 1.01 * self.E * (dl[1] - dl[0]) / (2 * w))
        self.assertAlmostEqual(abs(np.diff(k_febio)).max(), self.E)
        self.assertTrue((np.diff(F) >= 0).all())
        # The difference from FEBio's force is at most E w / 4, at the slack length.
        difference = abs(F - F_febio)
        self.assertAlmostEqual(difference.max(), self.E * w / 4.0, delta=1e-12)
        self.assertAlmostEqual(dl[difference.argmax()], self.SLACK, delta=1e-12)

    def test_gradient_at_the_slack_length_is_two_sided(self):
        """At the slack length the smooth law's gradient with respect to the
        slack length equals the central difference. FEBio's law has a corner
        there: reverse mode returns the slope of the active side (-E), and
        the central difference is half of it."""
        h = 1e-6

        def gradient_and_difference(law, row):
            term, state = self.one_spring(law, row, self.SLACK)
            tape = wp.Tape()
            with tape:
                N = term.axial_force(state)
            tape.backward(grads={N: wp.ones(1, dtype=wp.float64, device=DEVICE)})
            adjoint = float(term.params.grad.numpy()[0][1])
            values = []
            for sign in (1.0, -1.0):
                trial = list(row)
                trial[1] += sign * h
                values.append(self.force(law, trial, self.SLACK)[0])
            return adjoint, (values[0] - values[1]) / (2 * h)

        adjoint, fd = gradient_and_difference(SMOOTH_TENSION_ONLY_LINEAR_SPRING, [self.E, self.SLACK, self.W])
        self.assertAlmostEqual(adjoint, -self.E / 2.0, delta=1e-12)
        self.assertAlmostEqual(adjoint, fd, delta=1e-6 * self.E)
        adjoint, fd = gradient_and_difference(TENSION_ONLY_LINEAR_SPRING, [self.E, self.SLACK])
        self.assertAlmostEqual(adjoint, -self.E, delta=1e-12)
        self.assertAlmostEqual(fd, -self.E / 2.0, delta=1e-6 * self.E)

    def test_input_type_and_coverage_note(self):
        model = read_feb(CASES / "inputs" / "ch_spring_chords.feb")
        section = next(s for s in model.other_sections if s.tag == "Discrete")
        record = next(r for r in section.all("discrete_material") if r.attrs["type"] == "tension-only linear spring")
        record.attrs["type"] = "smooth tension-only linear spring"
        record.set("smoothing_length", 0.01)
        law, values, issues = resolve_discrete_material(record)
        self.assertIs(law, SMOOTH_TENSION_ONLY_LINEAR_SPRING)
        self.assertEqual(values, [15.0, 0.05, 0.01])
        self.assertIn("FEBio-Warp only", issues[0].note)
        from warp_febio.analysis.steps import inspect_model

        self.assertTrue(inspect_model(model).runnable)
        # In a run it stays within E w / 4 per spring of FEBio's law.
        smooth = run(build_system(model, device=DEVICE))
        febio = run(build_system(read_feb(CASES / "inputs" / "ch_spring_chords.feb"), device=DEVICE))
        a = smooth.states[-1].discrete["tension_only"]["discrete element signed force"]
        b = febio.states[-1].discrete["tension_only"]["discrete element signed force"]
        self.assertLess(abs(a - b).max(), 15.0 * 0.01 / 4.0 + 1e-12)
        record.set("smoothing_length", -1.0)
        self.assertIsNone(resolve_discrete_material(record)[0])

    def test_chord_curves_of_the_mitral_input_have_a_closed_form(self):
        """Each of the eight chord sets of the mitral input is a three-point
        curve that is FEBio's tension-only linear spring, so it can be given
        to the smooth law. With zero smoothing the forces are the curve's."""
        from warp_febio.discretization.fields import build_field_table
        from warp_febio.elements.line_builders import build_discrete_domains

        model = read_feb(MitralChordSetTests.MITRAL)
        fields = build_field_table(model)
        (curve,) = build_discrete_domains(model, fields, DEVICE)
        rows = [chord_curve_closed_form(row) for row in curve.params.numpy()]
        self.assertTrue(all(r is not None for r in rows))
        self.assertEqual(sorted({r[0] for r in rows}), [30.0, 60.0])
        self.assertEqual(sorted({r[1] for r in rows}), [2.0, 3.0, 4.0, 12.0])
        self.assertIsNone(chord_curve_closed_form([15.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0]))
        closed = DiscreteDomain(
            "closed",
            SMOOTH_TENSION_ONLY_LINEAR_SPRING,
            curve.X.numpy(),
            curve.emap.numpy(),
            [[E, slack, 0.0] for E, slack in rows],
            fields.count,
            DEVICE,
            region=curve.region.numpy(),
        )
        rng = np.random.default_rng(0)
        X, emap = curve.X.numpy(), curve.emap.numpy()
        axis = (X[:, 1] - X[:, 0]) / np.linalg.norm(X[:, 1] - X[:, 0], axis=1)[:, None]
        u = np.zeros((fields.count, 3))
        u[emap[:, 0]] = -axis * rng.uniform(0.0, 8.0, (curve.count, 1))
        state = State(0.0, wp.array(u, dtype=wp.vec3d, device=DEVICE))
        N_curve = curve.axial_force(state).numpy()
        self.assertGreater((N_curve > 0).sum(), 20)
        np.testing.assert_allclose(closed.axial_force(state).numpy(), N_curve, rtol=0, atol=1e-12 * N_curve.max())


class ReferenceStateTests(unittest.TestCase):
    """An undeformed tension-only element is active, on every device.

    FEBio computes the initial and the current length with the same code, so
    the stretch of an undeformed element is exactly 1 and a tension-only law
    is active there. Here the rest length is computed on the device for the
    same reason (line2.geometric_length). With a rest length from NumPy on the
    host, about one element in twenty starts slack on CUDA, with no stiffness
    in the first iteration.
    """

    def elements(self):
        rng = np.random.default_rng(1)
        n = 20000
        X = rng.uniform(-30.0, 30.0, (n + 3, 3))
        a = rng.integers(0, n + 3, n)
        b = (a + 1 + rng.integers(0, n + 2, n)) % (n + 3)
        pairs = np.stack([a, b], axis=1).astype(np.int32)
        return X, pairs, State(0.0, wp.zeros(n + 3, dtype=wp.vec3d, device=DEVICE))

    def check(self, term, state):
        self.assertEqual(term.update(state), 0)
        np.testing.assert_array_equal(term.data["length"].numpy(), term.rest_length.numpy())
        np.testing.assert_array_equal(term.data["axial_force"].numpy(), 0.0)
        self.assertEqual(int((term.data["axial_tangent"].numpy() <= 0).sum()), 0)

    def test_tension_only_truss_is_active_at_its_rest_length(self):
        X, pairs, state = self.elements()
        term = TrussDomain(
            "chords", X[pairs], pairs, 1.0, [[150.0, 0.5]], len(X), DEVICE, model=TENSION_ONLY_LINEAR_TRUSS
        )
        self.check(term, state)

    def test_tension_only_spring_is_active_at_its_rest_length(self):
        X, pairs, state = self.elements()
        term = DiscreteDomain("springs", TENSION_ONLY_LINEAR_SPRING, X[pairs], pairs, [[15.0, 0.0]], len(X), DEVICE)
        self.check(term, state)


class MitralChordSetTests(unittest.TestCase):
    """The chordae of the mitral input kept in references/febio (100 springs
    in 8 discrete sets), built from the input by the run path's own builder."""

    MITRAL = ROOT / "references/febio/originals/cardiofem/valve_pressurization/mitral_p2prolapse.feb"

    def setUp(self):
        from warp_febio.discretization.fields import build_field_table
        from warp_febio.elements.line_builders import build_discrete_domains

        model = read_feb(self.MITRAL)
        self.fields = build_field_table(model)
        self.terms = build_discrete_domains(model, self.fields, DEVICE)

    def test_all_sets_are_one_term_and_one_launch(self):
        (term,) = self.terms
        self.assertEqual(term.count, 100)
        self.assertEqual(term.set_names, ("A1", "A2_1", "A2_2", "A3", "P1", "P2_1", "P3", "P2_2"))
        self.assertEqual(term.params.shape, (8,))
        # Each set's parameter row and rest lengths are views of the term's
        # arrays, named by the set.
        named = term.parameters()
        self.assertEqual(len(named), 16)
        self.assertEqual(named["A1.rest_length"].shape, (20,))
        named["P3.material"].assign(named["P3.material"].numpy() * 2.0)
        self.assertEqual(term.params.numpy()[term.set_names.index("P3"), 0], 60.0)

    def test_gradient_of_total_tension_with_respect_to_every_set_parameter(self):
        (term,) = self.terms
        # Stretch each chord along its own axis by up to 8, so that some
        # pass their slack length (2 to 12 in this input) and some do not.
        rng = np.random.default_rng(0)
        X, emap = term.X.numpy(), term.emap.numpy()
        axis = X[:, 1] - X[:, 0]
        axis /= np.linalg.norm(axis, axis=1)[:, None]
        u = np.zeros((self.fields.count, 3))
        u[emap[:, 0]] = -axis * rng.uniform(0.0, 8.0, (term.count, 1))
        state = State(0.0, wp.array(u, dtype=wp.vec3d, device=DEVICE, requires_grad=True))
        weights = wp.array(np.ones(term.count), dtype=wp.float64, device=DEVICE)
        total = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(
                sum_axial_force,
                dim=term.count,
                inputs=[term.axial_force(state), weights],
                outputs=[total],
                device=DEVICE,
            )
        tape.backward(total)
        gradient = term.params.grad.numpy().copy()
        tension = term.axial_force(state).numpy()
        self.assertGreater((tension > 0).sum(), 20)
        self.assertGreater((tension == 0).sum(), 20)
        base = term.params.numpy().copy()
        for index in np.ndindex(base.shape):
            values = []
            for sign in (1.0, -1.0):
                trial = base.copy()
                trial[index] += sign * 1e-6
                term.params.assign(trial)
                values.append(float(term.axial_force(state).numpy().sum()))
            term.params.assign(base)
            fd = (values[0] - values[1]) / 2e-6
            self.assertAlmostEqual(gradient[index], fd, delta=1e-6 * max(abs(fd), 1.0), msg=str(index))
        # A set whose chords all stayed slack has no sensitivity to its stiffness.
        slack_sets = [k for k in range(8) if not (tension[term.region.numpy() == k] > 0).any()]
        for k in slack_sets:
            self.assertEqual(gradient[k, 0], 0.0)


def load_reference(case):
    path = CASES / "febio" / f"{case}.json"
    if not path.exists():
        return None
    ref = json.loads(path.read_text())
    deck = CASES / ref["input"]
    if hashlib.sha256(deck.read_bytes()).hexdigest() != ref["input_sha256"]:
        raise AssertionError(f"{case}: reference was not generated from this input file")
    return ref


def record(ref, name):
    """(items, values[time, item, column], column names) of the log record holding `name`."""
    rec = next(r for r in ref["records"] if name in r["data"])
    return rec["items"], np.array(rec["values"]), rec["data"]


def chord_terms(system):
    return [t for t in system.terms if getattr(t, "element_type", "") in ("line2", "discrete")]


class FebioAgreementTests(unittest.TestCase):
    """Runs of FEBio inputs against FEBio's log data for the same inputs.

    FEBio writes log data with 12 significant digits. FEBio converges to its
    own tolerances (dtol 1e-7, etol 1e-12 in these inputs), so agreement is
    limited by the two Newton solves and not by the elements; 1e-8 relative
    to the largest value is asserted.
    """

    TOLERANCE = 1e-8

    def run_case(self, case):
        ref = load_reference(case)
        if ref is None:
            self.skipTest(f"no FEBio reference for {case}")
        model = read_feb(CASES / ref["input"])
        system = build_system(model, device=DEVICE)
        return ref, model, system, run(system)

    def assert_close(self, value, reference):
        scale = abs(reference).max()
        self.assertLess(abs(value - reference).max() / scale, self.TOLERANCE)

    def test_one_spring_from_slack_into_tension_and_back(self):
        ref, model, system, results = self.run_case("ch_chord_slack_tension")
        _, logged, names = record(ref, "discrete element force")
        force = logged[:, 0, names.index("discrete element force")]
        elongation = logged[:, 0, names.index("discrete element elongation")]
        (term,) = chord_terms(system)
        mine = np.array([s.discrete["chord"]["discrete element signed force"][0] for s in results.states])
        self.assertEqual(len(mine), len(force))
        self.assert_close(mine, force)
        self.assertGreater((force == 0).sum(), 3)
        self.assertGreater((force > 0).sum(), 3)
        # The reaction at the driven node of the spring.
        items, logged, names = record(ref, "Rz")
        node = model.mesh.node_index(items)[0]
        Rz = np.array([s.reaction_force[node, 2] for s in results.states])
        self.assert_close(Rz, logged[:, 0, names.index("Rz")])
        self.assertGreater(elongation.max(), 5.0)

    def test_block_on_four_truss_chords(self):
        ref, model, system, results = self.run_case("ch_truss_chords")
        items, logged, _ = record(ref, "ux")
        nodes = model.mesh.node_index(items)
        u = np.stack([s.displacement[nodes] for s in results.states])
        self.assert_close(u, logged)
        items, logged, _ = record(ref, "Rx")
        upper = model.mesh.node_index(items)
        self.assert_close(np.stack([s.reaction_force[upper] for s in results.states]), logged)
        self.assertEqual([s.iterations for s in results.states[1:]], [3] * 10)  # as FEBio
        _, logged, _ = record(ref, "sx")
        stress = np.stack([s.stress["Chords"] for s in results.states])
        self.assert_close(stress, logged)
        # Chord tensions are not a FEBio output. Two checks on them: they
        # follow from FEBio's logged Cauchy stress as N = s A L / l (v = 0.5,
        # so J = 1; A = 1), and their sum balances the applied load of
        # 4 x 2.5 N along z, to the tolerance of the Newton solve.
        (term,) = chord_terms(system)
        N = results.states[-1].axial_force["Chords"]
        s_axial = logged[-1, :, :3].sum(axis=1)  # trace of s e (x) e
        stretch = results.states[-1].truss_stretch["Chords"]
        np.testing.assert_allclose(N, s_axial / stretch, rtol=1e-8)
        e = term.data["axis"].numpy()
        np.testing.assert_allclose((N * e[:, 2]).sum(), 10.0, rtol=1e-6)

    def test_block_on_four_spring_chords(self):
        ref, model, system, results = self.run_case("ch_spring_chords")
        items, logged, _ = record(ref, "ux")
        nodes = model.mesh.node_index(items)
        self.assert_close(np.stack([s.displacement[nodes] for s in results.states]), logged)
        items, logged, _ = record(ref, "Rx")
        upper = model.mesh.node_index(items)
        self.assert_close(np.stack([s.reaction_force[upper] for s in results.states]), logged)
        # One log record per discrete set, in the order of the input. Here a
        # term holds the sets that share a law.
        febio_force = {}
        for rec in ref["records"]:
            if "discrete element force" in rec["data"]:
                column = rec["data"].index("discrete element force")
                febio_force[len(febio_force)] = np.array(rec["values"])[:, :, column]
        logged = np.concatenate([febio_force[k] for k in sorted(febio_force)], axis=1)
        mine = np.stack(
            [
                np.concatenate(
                    [s.discrete[name]["discrete element signed force"] for name in ("linear", "tension_only", "curve")]
                )
                for s in results.states
            ]
        )
        self.assert_close(mine, logged)
        # FEBio's iteration counts for this input (full Newton, max_ups 0).
        self.assertEqual([s.iterations for s in results.states[1:]], [3, 4, 3, 3, 3, 3, 3, 3, 3, 3])
        # The run starts with three of the four chords slack and ends with all in tension.
        self.assertEqual((logged[1] == 0).sum(), 3)
        self.assertTrue((logged[-1] > 0).all())


@unittest.skipUnless(SUITE, "set FEBIO_TEST_SUITE to a checkout of the FEBio test suite")
class SuiteTests(unittest.TestCase):
    """FEBio test suite problems, against the suite's own stored results.

    The suite runs FEBio with -task=test, which logs the L2 norm of the
    solution vector (FELogSolutionNorm: total displacement at the free degrees
    of freedom) when an input has no data records of its own, and stores the
    value of the last step with the step and iteration counts in
    code/linuxGoldStandards.py. The values below are from that file at suite
    commit 986005f. They come from FEBio release builds, so this check does
    not depend on the FEBio build used elsewhere in these tests.

    The suite inputs use dtol 0.001, and bt01, di01 and di02 use BFGS, so the
    norm is compared to 1e-6. The iteration counts happen to match here.
    """

    #                 steps, iterations, solution norm,    linear solver
    GOLD = {
        "bt01": (10, 20, 2.1851076382e-05, "cg"),
        "di01": (10, 30, 1.94423248387, "direct"),
        "di02": (10, 30, 3.93956169936, "direct"),
    }

    def test_truss_and_spring_problems(self):
        for name, (steps, iterations, norm, solver) in self.GOLD.items():
            with self.subTest(problem=name):
                system = build_system(read_feb(pathlib.Path(SUITE) / "Verify" / f"{name}.feb"), device=DEVICE)
                # bt01 is a truss with mechanisms at zero stress: its first
                # tangent is singular (rank 55 of 66), which the dense check
                # solver refuses. Conjugate gradients solve it.
                results, states = run(system, linear_solver=solver, keep_states=True)
                u = states[-1].u.numpy()
                mine = float(np.linalg.norm(u[system.fields.bc == FREE]))
                self.assertEqual(len(results.states) - 1, steps)
                self.assertEqual(sum(s.iterations for s in results.states), iterations)
                self.assertLess(abs(mine - norm) / norm, 1e-6)


def febio_log_file(path, columns):
    """FEBio log data written to a file: (states, items, columns)."""
    text = path.read_text()
    states = text.count("*Step")
    rows = [line.split() for line in text.splitlines() if line and not line.startswith("*")]
    return np.array(rows, dtype=float)[:, 1:].reshape(states, -1, columns)


class ShellPatchOnTrussChordsTests(unittest.TestCase):
    """references/small_meshes/shell_patch_truss_chords.feb: a curved patch of
    nine quad4 shells, clamped on one edge, its opposite edge held by four
    truss chords from a fixed point, lifted by nodal loads. Trusses sharing
    nodes with shells, which no FEBio test suite problem has."""

    def test_matches_febio(self):
        model = read_feb(SMALL_MESHES / "shell_patch_truss_chords.feb")
        results = run(build_system(model, device=DEVICE))
        u_ref = febio_log_file(SMALL_MESHES / "febio" / "shell_patch_truss_chords_nodes.txt", 3)
        s_ref = febio_log_file(SMALL_MESHES / "febio" / "shell_patch_truss_chords_elements.txt", 6)
        u = np.stack([s.displacement for s in results.states])
        stress = np.stack([np.concatenate([s.stress[b.name] for b in model.mesh.blocks]) for s in results.states])
        self.assertLess(abs(u - u_ref).max() / abs(u_ref).max(), 1e-9)
        self.assertLess(abs(stress[:, :9] - s_ref[:, :9]).max() / abs(s_ref[:, :9]).max(), 1e-9)  # shells
        self.assertLess(abs(stress[:, 9:] - s_ref[:, 9:]).max() / abs(s_ref[:, 9:]).max(), 1e-9)  # trusses
        # FEBio's iteration counts for this input (full Newton, max_ups 0).
        self.assertEqual([s.iterations for s in results.states[1:]], [3] * 10)
        self.assertTrue((results.states[-1].axial_force["Chords"] > 0).all())


def make_tension_only(model):
    """Switch the truss material of a model to the tension-only truss."""
    material = next(m for m in model.materials if m.type == "linear truss")
    material.type = "tension-only linear truss"
    material.params.attrs["type"] = "tension-only linear truss"


class TensionOnlyTrussRunTests(unittest.TestCase):
    """The tension-only truss in a run. FEBio has no such material, so the
    checks are: it gives FEBio's linear truss result when every chord is in
    tension, and a slack chord carries nothing where a linear truss pushes."""

    def run_case(self, tension_only, lengthen=None, path=CASES / "inputs" / "ch_truss_chords.feb"):
        model = read_feb(path)
        if tension_only:
            make_tension_only(model)
        system = build_system(model, device=DEVICE)
        (term,) = chord_terms(system)
        if lengthen is not None:
            lengths = term.rest_length.numpy().copy()
            lengths[0] *= lengthen
            term.rest_length.assign(lengths)
        return model, system, run(system)

    def test_equals_febio_linear_truss_when_all_chords_are_in_tension(self):
        ref = load_reference("ch_truss_chords")
        model, _, results = self.run_case(tension_only=True)
        items, logged, _ = record(ref, "ux")
        u = np.stack([s.displacement[model.mesh.node_index(items)] for s in results.states])
        self.assertLess(abs(u - logged).max() / abs(logged).max(), 1e-8)
        _, logged, _ = record(ref, "sx")
        stress = np.stack([s.stress["Chords"] for s in results.states])
        self.assertLess(abs(stress - logged).max() / abs(logged).max(), 1e-8)

    def test_a_slack_chord_carries_nothing_where_a_linear_truss_pushes(self):
        # The quad4 patch held by four chords, the first made 30 percent longer.
        patch = SMALL_MESHES / "shell_patch_truss_chords.feb"
        _, _, linear = self.run_case(tension_only=False, lengthen=1.3, path=patch)
        _, _, slack = self.run_case(tension_only=True, lengthen=1.3, path=patch)
        N_linear = linear.states[-1].axial_force["Chords"]
        N_slack = slack.states[-1].axial_force["Chords"]
        self.assertLess(N_linear[0], 0.0)  # the linear truss is in compression
        self.assertEqual(N_slack[0], 0.0)
        self.assertTrue((N_slack[1:] > 0).all())
        # The push of the first chord changes what its neighbor carries.
        self.assertGreater(abs(N_linear[1] - N_slack[1]) / N_slack[1], 0.05)

    def test_coverage_report_says_it_is_not_a_febio_material(self):
        from warp_febio.analysis.steps import inspect_model

        model = read_feb(CASES / "inputs" / "ch_truss_chords.feb")
        make_tension_only(model)
        report = inspect_model(model)
        self.assertTrue(report.runnable)
        entry = next(e for e in report.entries if e.febio_name == "tension-only linear truss")
        self.assertIn("FEBio-Warp only", entry.note)


def tension_gradients(system, weights):
    """d(sum_i w_i N_i)/dp for every parameter array, by the adjoint.

    The solve is one step on the tape, and the tension is computed after it
    by the domains' own kernels, so the tape adds the direct dependence of
    the tension on the parameters to the dependence through the solution.
    """
    params = system.parameters()
    for array in params.values():
        array.grad.zero_()
    tape = wp.Tape()
    state = solve_on_tape(system, tape, "direct")
    J = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
    with tape:
        for term, w in zip(chord_terms(system), weights):
            wp.launch(sum_axial_force, dim=term.count, inputs=[term.axial_force(state), w], outputs=[J], device=DEVICE)
    tape.backward(J)
    gradients = {key: array.grad.numpy().copy() for key, array in params.items()}
    tape.zero()
    return float(J.numpy()[0]), gradients


def tension(system, weights):
    _, states = run(system, keep_states=True)
    return sum(
        float(np.dot(w.numpy(), t.axial_force(states[-1]).numpy())) for t, w in zip(chord_terms(system), weights)
    )


class ChordTensionGradientTests(unittest.TestCase):
    """Adjoint gradients through the solve, with respect to every parameter
    of the system, against central differences of FEBio-Warp itself."""

    def check_tension(self, case, prepare=None, only=None, step=1e-5, floor=10.0, edit=None):
        path = case if isinstance(case, pathlib.Path) else CASES / "inputs" / f"{case}.feb"
        model = read_feb(path)
        if edit is not None:
            edit(model)
        system = build_system(model, device=DEVICE)
        if prepare is not None:
            prepare(system)
        rng = np.random.default_rng(0)
        weights = [
            wp.array(rng.uniform(0.5, 1.5, t.count), dtype=wp.float64, device=DEVICE) for t in chord_terms(system)
        ]
        J, gradients = tension_gradients(system, weights)
        self.assertAlmostEqual(J, tension(system, weights), delta=1e-12 * abs(J))
        checked = 0
        for key, array in system.parameters().items():
            if only is not None and key not in only:
                continue
            base = array.numpy().copy()
            for index in np.ndindex(base.shape):
                h = step * max(abs(base[index]), floor)
                values = []
                for sign in (1.0, -1.0):
                    trial = base.copy()
                    trial[index] += sign * h
                    array.assign(trial)
                    values.append(tension(system, weights))
                array.assign(base)
                fd = (values[0] - values[1]) / (2 * h)
                self.assertAlmostEqual(gradients[key][index], fd, delta=1e-5 * max(abs(fd), 1e-3), msg=f"{key}{index}")
                checked += 1
        return gradients, checked

    def test_truss_chord_tension(self):
        def distinct_areas(system):
            (term,) = chord_terms(system)
            term.area.assign(np.array([1.0, 0.7, 1.3, 0.9]))

        gradients, checked = self.check_tension("ch_truss_chords", distinct_areas)
        self.assertEqual(checked, 16)  # 12 chord and block values, and the load on each of 4 nodes
        self.assertEqual(gradients["Chords.material"][0, 1], 0.0)  # v does not enter the force

    def test_tension_only_truss_chord_tension_with_a_slack_chord(self):
        """One chord is given a rest length longer than its span. As a linear
        truss it would push; as a tension-only truss it carries nothing, its
        parameters have zero gradient, and the rest match central differences."""

        def lengthen_first(system):
            (term,) = chord_terms(system)
            lengths = term.rest_length.numpy().copy()
            lengths[0] *= 1.3
            term.rest_length.assign(lengths)

        # On the shell patch, whose clamped edge holds it whatever the chords
        # do. The block of ch_truss_chords is held by its chords alone, and
        # with one slack a second one going slack in the first iteration
        # leaves it without support in a static step.
        gradients, checked = self.check_tension(
            SMALL_MESHES / "shell_patch_truss_chords.feb",
            lengthen_first,
            only=("Chords.area", "Chords.rest_length", "Chords.material"),
            step=1e-4,
            floor=1e-2,
            edit=make_tension_only,
        )
        self.assertEqual(checked, 10)
        self.assertEqual(gradients["Chords.rest_length"][0], 0.0)
        self.assertEqual(gradients["Chords.area"][0], 0.0)
        self.assertTrue((gradients["Chords.area"][1:] != 0).all())

    def test_spring_chord_tension_with_chords_slack_at_the_solution(self):
        """Two of the four chords are given a slack length longer than the
        pull, so they carry nothing at the solution. Their parameters then
        have zero gradient, and the others still match central differences."""

        def keep_two_slack(system):
            # Addressed by the discrete set's FEBio name.
            system.parameters()["tension_only.material"].assign(np.array([[15.0, 1.0]]))

        gradients, checked = self.check_tension("ch_spring_chords", keep_two_slack)
        self.assertEqual(checked, 16)
        for key, value in gradients.items():
            if "tension_only" in key:
                np.testing.assert_array_equal(value, 0.0)

    def test_spring_chord_tension(self):
        gradients, checked = self.check_tension("ch_spring_chords")
        self.assertEqual(checked, 16)
        # Lengthening the slack of the tension-only springs lowers the total tension.
        self.assertLess(gradients["tension_only.material"][0, 1], 0.0)
        # Parameters are keyed by FEBio's discrete set names.
        for name in ("linear", "tension_only", "curve"):
            self.assertIn(f"{name}.material", gradients)
            self.assertIn(f"{name}.rest_length", gradients)
        self.assertEqual(gradients["tension_only.rest_length"].shape, (2,))

    def test_chord_tension_on_a_shell_patch_with_respect_to_chord_area(self):
        """The valve setting in small: quad4 shells held by truss chords.
        Cross-sectional area is the chord parameter Mangine, Laurence et al.
        (arXiv 2411.09599) found most influential."""
        gradients, checked = self.check_tension(
            SMALL_MESHES / "shell_patch_truss_chords.feb",
            only=("Chords.area", "Chords.rest_length"),
            step=1e-4,
            floor=1e-2,
        )
        self.assertEqual(checked, 8)
        self.assertTrue((abs(gradients["Chords.area"]) > 0).all())

    def test_displacement_output_through_trusses(self):
        system = build_system(read_feb(CASES / "inputs" / "ch_truss_chords.feb"), device=DEVICE)

        def output(u):
            dJ = np.zeros_like(u)
            dJ[:8, 2] = 1.0
            return float(u[:8, 2].sum()), dJ

        rows = check_parameter_gradients(system, output)
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertLess(row.relative_error, 2e-5, msg=f"{row.parameter}{row.index}")


class ChordForceOnNodesTests(unittest.TestCase):
    """The resultant chord force on a set of nodes (`force_on`): the force on
    a papillary tip, as an output and as an objective."""

    def build(self, prepare=None):
        model = read_feb(CASES / "inputs" / "ch_spring_chords.feb")
        system = build_system(model, device=DEVICE)
        if prepare is not None:
            prepare(system)
        return model, system

    def weights(self, model, system, node_set):
        entries = system.fields.entries_of("displacement", model.mesh.node_set_indices(node_set))
        return [t.entry_weights(entries) for t in chord_terms(system)]

    def force(self, system, state, weights):
        return sum(t.force_on(state, w).numpy()[0] for t, w in zip(chord_terms(system), weights))

    def test_force_on_the_apex_against_febio(self):
        """Four springs of three laws (three terms) meet at one fixed node.
        The force they exert on it follows from FEBio's logged spring forces,
        and along the pull it balances FEBio's reactions at the driven face of
        the block, which is all that holds the block against the chords."""
        ref = load_reference("ch_spring_chords")
        if ref is None:
            self.skipTest("no FEBio reference for ch_spring_chords")
        model, system = self.build()
        _, states = run(system, keep_states=True)
        at_apex = self.weights(model, system, "apex")
        mine = self.force(system, states[-1], at_apex)
        # From FEBio's signed force of each spring and the spring axes.
        logged = []
        for rec in ref["records"]:
            if "discrete element force" in rec["data"]:
                logged += list(np.array(rec["values"])[-1, :, rec["data"].index("discrete element force")])
        apex = set(system.fields.entries_of("displacement", model.mesh.node_set_indices("apex")).tolist())
        x = system.fields.entries_of("displacement")
        position = np.zeros((system.fields.count, 3))
        position[x] = model.mesh.nodes
        position += states[-1].u.numpy()
        expected = np.zeros(3)
        k = 0
        for term in chord_terms(system):
            for a, b in term.emap.numpy():
                e = (position[b] - position[a]) / np.linalg.norm(position[b] - position[a])
                expected += (float(a in apex) - float(b in apex)) * logged[k] * e
                k += 1
        self.assertEqual(k, 4)
        np.testing.assert_allclose(mine, expected, rtol=0, atol=1e-10 * abs(expected).max())
        items, reactions, _ = record(ref, "Rx")
        # FEBio logs a reaction only at the prescribed component (z); at the
        # fixed components its R is zero. Its sign is that of the force the
        # block exerts on its support, so the sum is minus the force on the apex.
        self.assertAlmostEqual(mine[2], -reactions[-1, :, 2].sum(), delta=1e-8 * abs(mine[2]))
        self.assertGreater(abs(mine).max(), 1.0)
        # Every chord has one end at the apex and one on the block, so the
        # force on all other nodes together is the opposite.
        everywhere = [wp.array(1.0 - w.numpy(), dtype=wp.float64, device=DEVICE) for w in at_apex]
        np.testing.assert_allclose(self.force(system, states[-1], everywhere), -mine, rtol=1e-12)

    def test_gradient_of_the_force_on_two_insertions(self):
        """One component of the chord force on two of the four insertion
        nodes, through the solve, with respect to every spring parameter."""
        model, system = self.build()
        ends = np.concatenate([t.emap.numpy()[:, 1] for t in chord_terms(system)])[:2]
        weights = [t.entry_weights(ends) for t in chord_terms(system)]
        direction = np.array([0.3, -0.2, 1.0])

        def value():
            _, states = run(system, keep_states=True)
            return float(self.force(system, states[-1], weights) @ direction)

        params = system.parameters()
        for array in params.values():
            array.grad.zero_()
        tape = wp.Tape()
        state = solve_on_tape(system, tape, "direct")
        with tape:
            forces = [t.force_on(state, w) for t, w in zip(chord_terms(system), weights)]
        seed = wp.array([direction], dtype=wp.vec3d, device=DEVICE)
        tape.backward(grads=dict.fromkeys(forces, seed))
        gradients = {key: array.grad.numpy().copy() for key, array in params.items()}
        tape.zero()
        checked = 0
        for key, array in params.items():
            if ".material" not in key and ".rest_length" not in key:
                continue
            if not any(key.startswith(name) for name in ("linear.", "tension_only.", "curve.")):
                continue
            base = array.numpy().copy()
            for index in np.ndindex(base.shape):
                h = 1e-5 * max(abs(base[index]), 10.0)
                values = []
                for sign in (1.0, -1.0):
                    trial = base.copy()
                    trial[index] += sign * h
                    array.assign(trial)
                    values.append(value())
                array.assign(base)
                fd = (values[0] - values[1]) / (2 * h)
                self.assertAlmostEqual(gradients[key][index], fd, delta=1e-5 * max(abs(fd), 1e-3), msg=f"{key}{index}")
                checked += 1
        self.assertEqual(checked, 14)
        self.assertGreater(sum(int((abs(g) > 1e-6).sum()) for g in gradients.values()), 6)


class ChordLengthRecoveryTests(unittest.TestCase):
    """examples/chordae/recover_chord_lengths.py: a target shape is made with
    changed chord rest lengths, and the lengths are fitted to it with adjoint
    gradients. The setting of FEBio FINESSE (arXiv 2407.09629) with the chords
    in the model: the shape is known and the chords are not."""

    def test_rest_lengths_and_tensions_are_recovered_from_a_target_shape(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("SciPy is not installed")
        path = ROOT / "examples" / "chordae" / "recover_chord_lengths.py"
        spec = importlib.util.spec_from_file_location("recover_chord_lengths", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        true_lengths, fitted_lengths, true_tension, fitted_tension, evaluations = module.recover(
            DEVICE, log=lambda *_: None
        )
        np.testing.assert_allclose(fitted_lengths, true_lengths, rtol=1e-8)
        np.testing.assert_allclose(fitted_tension, true_tension, rtol=1e-6)
        # The target tensions differ by a factor of 30 between chords, so the
        # fit is not recovering four similar numbers.
        self.assertGreater(true_tension.max() / true_tension.min(), 10.0)
        self.assertLess(evaluations, 40)


class CoverageTests(unittest.TestCase):
    def test_new_inputs_are_runnable_and_unknown_spring_curves_are_refused(self):
        from warp_febio.analysis.steps import inspect_model

        for case in ("ch_truss_chords", "ch_spring_chords", "ch_chord_slack_tension"):
            self.assertTrue(inspect_model(read_feb(CASES / "inputs" / f"{case}.feb")).runnable, case)
        model = read_feb(CASES / "inputs" / "ch_spring_chords.feb")
        section = next(s for s in model.other_sections if s.tag == "Discrete")
        section.all("discrete_material")[2].child("force").set("interpolate", "smooth")
        report = inspect_model(model)
        self.assertFalse(report.runnable)
        self.assertIn("nonlinear spring.force.interpolate", [e.febio_name for e in report.missing()])


if __name__ == "__main__":
    unittest.main()
