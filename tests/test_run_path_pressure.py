"""Pressure load as a term, and the gradient check case.

The case is one unit hex8, neo-Hookean, with symmetry conditions on three
faces and a follower pressure on the top face
(references/single_element/hex8_pressure.feb). The deformation is homogeneous
uniaxial stress, so there is a closed-form solution to compare with, and it
depends on E, v and the pressure, so all three gradients are nonzero.

No FEBio result is stored for this input yet; the comparison here is with the
closed form. FEBio's neo-Hookean: sigma = mu/J (b - I) + lambda ln(J)/J I.
"""

import os
import unittest

import numpy as np

from warp_febio.io import read_feb
from warp_febio.model.model import Record, Surface

ROOT = os.path.join(os.path.dirname(__file__), "..")
SINGLE = os.path.join(ROOT, "references", "single_element")
CASE = os.path.join(SINGLE, "hex8_pressure.feb")
E0, V0, P0 = 1.0, 0.3, 0.2  # the values in the input

try:
    import warp as wp

    wp.config.quiet = True
    HAVE_WARP = True
except ImportError:
    HAVE_WARP = False


def closed_form(E, v, p, linear=False):
    """Stretches (lateral a, axial c) of a neo-Hookean cube with free sides
    under pressure p on the top face. Follower load: Cauchy stress
    sz = -p. linear = 1 (load on the reference area): nominal stress -p."""
    mu, lam = E / (2 * (1 + v)), E * v / ((1 + v) * (1 - 2 * v))

    def f(x):
        a, c = x
        J = a * a * c
        load = p * c if linear else p * J
        return np.array(
            [mu * (a * a - 1) + lam * np.log(J), mu * (c * c - 1) + lam * np.log(J) + load]
        )

    x = np.array([1.0, 1.0])
    for _ in range(50):
        jac = np.zeros((2, 2))
        for k in range(2):
            d = np.zeros(2)
            d[k] = 1e-7
            jac[:, k] = (f(x + d) - f(x - d)) / 2e-7
        x = x - np.linalg.solve(jac, f(x))
    return x


def set_load(model, tag, value):
    model.loads[0].set(tag, value)


def build(model=None):
    from warp_febio.analysis.steps import build_system

    return build_system(model or read_feb(CASE), device="cpu")


def exact_tangent_error(system, u0):
    """Largest relative difference between tangent(exact=True) and minus the
    central-difference derivative of the residual, and the dense tangents."""
    from warp_febio.assembly.system import State
    from warp_febio.solvers.linear.warp_linear import dense

    def residual(u, time=1.0):
        state = State(time, wp.array(u, dtype=wp.vec3d, device="cpu"))
        system.update(state)
        return state, system.residual(state).numpy().reshape(-1)

    state, _ = residual(u0)
    K = dense(system.matrix(system.tangent(state, exact=True)))
    Ks = dense(system.matrix(system.tangent(state)))
    h, fd = 1e-6, np.zeros_like(K)
    for j in range(K.shape[0]):
        e = np.zeros(K.shape[0])
        e[j] = h
        e = e.reshape(-1, 3)
        fd[:, j] = -(residual(u0 + e)[1] - residual(u0 - e)[1]) / (2 * h)
    return abs(K - fd).max() / abs(K).max(), K, Ks


@unittest.skipUnless(HAVE_WARP, "Warp is not installed (pip install 'warp_febio[warp]')")
class Pressure(unittest.TestCase):
    def test_coverage(self):
        """The run path builds the load; the fixture of pull request 5 still
        refuses it, with the report."""
        from warp_febio.compiled import compile
        from warp_febio.coverage import CoverageError
        from warp_febio.loads.pressure import PressureLoad

        system = build()
        self.assertTrue(system.report.runnable)
        self.assertEqual([type(t).__name__ for t in system.terms], ["BrickDomain", "PressureLoad"])
        self.assertIsInstance(system.terms[1], PressureLoad)
        with self.assertRaises(CoverageError):
            compile(read_feb(CASE), device="cpu")

    def test_agrees_with_the_closed_form(self):
        from warp_febio.analysis.steps import run

        results = run(build())
        self.assertEqual([s.time for s in results.states], [0.0, 0.25, 0.5, 0.75, 1.0])
        for s in results.states[1:]:
            p = P0 * s.time  # the load curve is a ramp to 1
            a, c = closed_form(E0, V0, p)
            u = s.displacement[6]  # node 7, at (1, 1, 1)
            np.testing.assert_allclose(u, [a - 1, a - 1, c - 1], rtol=0, atol=1e-9)
            sx, sy, sz, sxy, syz, sxz = s.stress["Part1"][0]
            self.assertAlmostEqual(sz, -p, delta=1e-9)  # Cauchy stress = -pressure
            for other in (sx, sy, sxy, syz, sxz):
                self.assertAlmostEqual(other, 0.0, delta=1e-9)

    def test_linear_flag_loads_the_reference_area(self):
        from warp_febio.analysis.steps import run

        model = read_feb(CASE)
        set_load(model, "linear", 1)
        system = build(model)
        self.assertIsNone(system.terms[1].linearize(system.state(), None))
        s = run(system).states[-1]
        a, c = closed_form(E0, V0, P0, linear=True)
        np.testing.assert_allclose(s.displacement[6], [a - 1, a - 1, c - 1], rtol=0, atol=1e-9)

    def test_stiffness_forms(self):
        """exact=True is minus the derivative of the residual. FEBio's default
        symmetrized form is symmetric and, on a facet with free edges, is not
        that derivative."""
        rng = np.random.default_rng(1)
        system = build()
        err, K, Ks = exact_tangent_error(system, 0.05 * rng.standard_normal((system.n, 3)))
        self.assertLess(err, 1e-8)
        self.assertLess(abs(Ks - Ks.T).max() / abs(Ks).max(), 1e-12)
        self.assertGreater(abs(K - K.T).max() / abs(K).max(), 1e-3)
        self.assertGreater(abs(K - Ks).max() / abs(K).max(), 1e-3)

    def test_stiffness_on_a_shell_front_and_back_face(self):
        """Pressure on a quad4 shell: the front face carries displacement,
        the back face (shell_bottom) shell displacement, with the sign
        flipped as in FEBio."""
        rng = np.random.default_rng(2)
        for bottom in (0, 1):
            with self.subTest(shell_bottom=bottom):
                model = read_feb(os.path.join(SINGLE, "quad4_shell_neo_hookean.feb"))
                model.mesh.surfaces["Face"] = Surface(
                    "Face", ["quad4"], [np.array([0, 1, 2, 3], dtype=np.int32)],
                    np.array([1], dtype=np.int32),
                )
                load = Record("surface_load", {"name": "P", "surface": "Face", "type": "pressure"})
                load.set("pressure", 0.05)
                load.set("shell_bottom", bottom)
                model.loads.append(load)
                system = build(model)
                term = system.terms[1]
                variable = "shell displacement" if bottom else "displacement"
                np.testing.assert_array_equal(
                    term.emap.numpy()[0], system.fields.entries_of(variable)
                )
                self.assertEqual(term.factor(0.0), -1.0 if bottom else 1.0)
                err, _, _ = exact_tangent_error(system, 0.03 * rng.standard_normal((system.n, 3)))
                self.assertLess(err, 1e-8)

    def test_newton_with_each_stiffness_form(self):
        """Both forms reach the same solution. With the exact form the
        iteration converges quadratically; with FEBio's default symmetrized
        form on this open facet it takes more iterations."""
        from warp_febio.analysis.steps import run

        counts = {}
        for symmetric in (1, 0):
            model = read_feb(CASE)
            set_load(model, "symmetric_stiffness", symmetric)
            results = run(build(model))
            a, c = closed_form(E0, V0, P0)
            np.testing.assert_allclose(
                results.states[-1].displacement[6], [a - 1, a - 1, c - 1], rtol=0, atol=1e-9
            )
            counts[symmetric] = sum(s.iterations for s in results.states)
        self.assertLess(counts[0], counts[1])

    def test_gradients(self):
        """d(output)/d(E, v, pressure) by the adjoint, against central
        differences of complete runs and against the closed form.
        Output: z displacement of the top face, J = c - 1."""
        from warp_febio.analysis.steps import run
        from warp_febio.sensitivity.adjoint import parameter_gradient

        system = build()
        domain, load = system.terms
        top = system.fields.entries_of("displacement", [4, 5, 6, 7])
        base = np.array([E0, V0, P0])

        def output(x):
            domain.set_parameters(x[:2])
            load.pressure.assign([x[2]])
            _, states = run(system, keep_states=True)
            return states[-1].u.numpy()[top, 2].mean(), states[-1]

        J, state = output(base)
        dJ_du = np.zeros((system.n, 3))
        dJ_du[top, 2] = 0.25
        gradients, _ = parameter_gradient(system, state, dJ_du)
        self.assertEqual(sorted(gradients), ["Part1.material", "TopPressure.pressure"])
        self.assertEqual(domain.parameter_names, ("E", "v"))
        adjoint = np.array([*gradients["Part1.material"][0], gradients["TopPressure.pressure"][0]])

        fd, closed = np.zeros(3), np.zeros(3)
        for k in range(3):
            h = 1e-5 * base[k]
            d = np.zeros(3)
            d[k] = h
            fd[k] = (output(base + d)[0] - output(base - d)[0]) / (2 * h)
            closed[k] = (closed_form(*(base + d))[1] - closed_form(*(base - d))[1]) / (2 * h)
        output(base)

        self.assertAlmostEqual(J, closed_form(*base)[1] - 1, delta=1e-9)
        self.assertTrue((abs(closed) > 1e-2).all())  # all three matter here
        np.testing.assert_allclose(adjoint, fd, rtol=1e-6)
        np.testing.assert_allclose(adjoint, closed, rtol=1e-6)

        # With the tangent Newton uses (FEBio's symmetrized pressure
        # stiffness) the adjoint gradient is wrong, which is why the adjoint
        # asks every term for its exact tangent.
        approximate, _ = parameter_gradient(system, state, dJ_du, exact=False)
        wrong = np.array(
            [*approximate["Part1.material"][0], approximate["TopPressure.pressure"][0]]
        )
        self.assertGreater(abs(wrong - closed).max() / abs(closed).max(), 1e-2)

    def test_gradient_checker_covers_every_parameter(self):
        """sensitivity/check.py walks all parameter arrays of all terms."""
        from warp_febio.sensitivity.check import check_parameter_gradients

        system = build()
        top = system.fields.entries_of("displacement", [4, 5, 6, 7])

        def output(u):
            dJ_du = np.zeros_like(u)
            dJ_du[top, 2] = 0.25
            return u[top, 2].mean(), dJ_du

        rows = check_parameter_gradients(system, output)
        self.assertEqual(
            [(r.parameter, r.index) for r in rows],
            [("Part1.material", (0, 0)), ("Part1.material", (0, 1)), ("TopPressure.pressure", (0,))],
        )
        for r in rows:
            self.assertGreater(abs(r.finite_difference), 1e-2)
            self.assertLess(r.relative_error, 1e-6)


PATCH = os.path.join(
    ROOT, "references", "febio", "originals", "patchwarp", "static_pressure_and_a2",
    "pressure_only_static",
)
try:
    import scipy.sparse
    import scipy.sparse.linalg

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


@unittest.skipUnless(HAVE_WARP and HAVE_SCIPY, "needs Warp, and SciPy for a sparse direct solve")
class PressureAgainstFEBio(unittest.TestCase):
    def test_shell_patch_under_pressure(self):
        """A FEBio 4.9 run of a 1,863-element quad4 shell patch under a
        follower pressure (static, isotropic elastic), stored in the
        repository from the pilot phase. Only the FEBio input and FEBio's
        plot file are used. The first three FEBio output times are compared;
        the plot file is single precision.

        The linear solve here is SciPy's sparse direct solver on the host,
        because Warp's iterative solvers take about 1,000 iterations per
        solve on this thin shell with a block-diagonal preconditioner."""
        from warp_febio.analysis.steps import build_system, collect
        from warp_febio.io import read_xplt
        from warp_febio.solvers.nonlinear import newton

        def sparse_direct(A, b, method="direct", tol=0.0):
            n, nnz = A.nrow, A.nnz_sync()
            M = scipy.sparse.bsr_matrix(
                (A.values.numpy()[:nnz], A.columns.numpy()[:nnz], A.offsets.numpy()[: n + 1]),
                shape=(3 * n, 3 * n),
            ).tocsc()
            x = scipy.sparse.linalg.spsolve(M, b.numpy().reshape(-1))
            return wp.array(x.reshape(n, 3), dtype=wp.vec3d, device=b.device), 1

        model = read_feb(PATCH + ".feb")
        febio = read_xplt(PATCH + ".xplt")
        U = febio.variable("displacement")
        system = build_system(model, device="cpu")
        settings = newton.NewtonSettings.from_record(model.steps[0].control.child("solver"))
        settings.dtol, settings.etol = 1e-8, 1e-12
        original, newton.solve = newton.solve, sparse_direct
        try:
            state = system.state()
            for k in (1, 2, 3):
                state, _ = newton.solve_step(system, state, float(febio.times[k]), settings)
                u = collect(system, state).displacement
                self.assertLess(abs(u - U[k]).max() / abs(U[k]).max(), 1e-6)
        finally:
            newton.solve = original


if __name__ == "__main__":
    unittest.main()
