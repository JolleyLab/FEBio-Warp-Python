"""Run path template: field table, terms, Newton step, step driver, adjoint.

The inputs and the stored FEBio results are those of the single-element and
small-mesh references (see references/single_element/README.md). FEBio is not
needed to run these.
"""

import os
import unittest

import numpy as np
from support import DEVICE, full_tier, setUpModule, tearDownModule  # noqa: F401  (device for the module)

from warp_febio.io import read_feb

ROOT = os.path.join(os.path.dirname(__file__), "..")
SINGLE = os.path.join(ROOT, "references", "single_element")
SMALL = os.path.join(ROOT, "references", "small_meshes")
CASES = (
    (SINGLE, "hex8_neo_hookean"),
    (SINGLE, "quad4_shell_neo_hookean"),
    (SMALL, "hex_block"),
    (SMALL, "shell_patch"),
)

try:
    import warp as wp

    wp.config.quiet = True
    HAVE_WARP = True
except ImportError:
    HAVE_WARP = False


def febio_log(path, states, columns):
    with open(path) as f:
        rows = [line.split() for line in f if not line.startswith("*")]
    return np.array(rows, dtype=float)[:, 1:].reshape(states, -1, columns)


def build(folder, name):
    from warp_febio.analysis.steps import build_system

    return build_system(read_feb(os.path.join(folder, name + ".feb")), device=DEVICE)


@unittest.skipUnless(HAVE_WARP, "Warp is not installed (pip install 'warp_febio[warp]')")
class RunPath(unittest.TestCase):
    def test_field_table(self):
        """One 3-vector entry per (node, variable); FEBio's three states."""
        from warp_febio.discretization.fields import FIXED, FREE, PRESCRIBED

        f = build(SINGLE, "hex8_neo_hookean").fields
        self.assertEqual(f.count, 8)  # 8 nodes, displacement only
        self.assertEqual((f.bc == FIXED).sum(), 12)  # bottom face, x y z
        self.assertEqual((f.bc == PRESCRIBED).sum(), 4)  # top face, z
        self.assertEqual((f.bc == FREE).sum(), 8)  # top face, x y
        f = build(SINGLE, "quad4_shell_neo_hookean").fields
        self.assertEqual(f.count, 8)  # 4 nodes, displacement and shell displacement
        self.assertEqual((f.bc == FREE).sum(), 8)

    def test_agrees_with_febio(self):
        from warp_febio.analysis.steps import run

        for folder, name in CASES:
            with self.subTest(name):
                results = run(build(folder, name))
                n = len(results.states)
                u_ref = febio_log(os.path.join(folder, "febio", name + "_nodes.txt"), n, 3)
                s_ref = febio_log(os.path.join(folder, "febio", name + "_elements.txt"), n, 6)
                u = np.stack([s.displacement for s in results.states])
                s = np.stack([s.stress["Part1"] for s in results.states])
                # FEBio prints 12 significant digits; 1e-10 leaves margin.
                self.assertLess(abs(u - u_ref).max() / abs(u_ref).max(), 1e-10)
                self.assertLess(abs(s - s_ref).max() / abs(s_ref).max(), 1e-10)

    def test_agrees_with_febio_using_cg(self):
        """Quick tier: Warp's CG on the device for one case. The full tier runs
        every iterative solver on every case."""
        from warp_febio.analysis.steps import run

        folder, name = CASES[0]
        results = run(build(folder, name), linear_solver="cg")
        n = len(results.states)
        u_ref = febio_log(os.path.join(folder, "febio", name + "_nodes.txt"), n, 3)
        u = np.stack([s.displacement for s in results.states])
        self.assertLess(abs(u - u_ref).max() / abs(u_ref).max(), 1e-9)

    def test_agrees_with_febio_using_sparse_direct_solvers(self):
        """The sparse LU on the host ("sparse", SciPy) and, where nvmath-python
        is installed and the device is a GPU, cuDSS on the device ("cudss")."""
        import importlib.util

        from warp_febio.analysis.steps import run

        methods = ["sparse"]
        if wp.get_device(DEVICE).is_cuda and importlib.util.find_spec("nvmath") and importlib.util.find_spec("cupy"):
            methods.append("cudss")
        for method in methods:
            for folder, name in CASES:
                with self.subTest(method=method, name=name):
                    results = run(build(folder, name), linear_solver=method)
                    n = len(results.states)
                    u_ref = febio_log(os.path.join(folder, "febio", name + "_nodes.txt"), n, 3)
                    u = np.stack([s.displacement for s in results.states])
                    self.assertLess(abs(u - u_ref).max() / abs(u_ref).max(), 1e-10)

    def test_cudss_says_what_is_missing(self):
        import importlib.util

        from warp_febio.analysis.steps import run
        from warp_febio.assembly.system import SolveError
        from warp_febio.solvers.linear.cudss import CudssUnavailable

        if importlib.util.find_spec("nvmath") and wp.get_device(DEVICE).is_cuda:
            self.skipTest("nvmath-python is installed and the device is a GPU")
        folder, name = CASES[0]
        # Not a SolveError: a method that is not installed is not a failed
        # step, and must not be retried with a smaller one.
        with self.assertRaises(CudssUnavailable) as caught:
            run(build(folder, name), linear_solver="cudss")
        self.assertNotIsInstance(caught.exception, SolveError)
        self.assertRegex(str(caught.exception), "nvmath-python|CUDA device")

    @full_tier("CG, BiCGSTAB and GMRES on every reference case, about 4 minutes on CPU")
    def test_agrees_with_febio_using_warp_linear_solvers(self):
        from warp_febio.analysis.steps import run

        for method in ("cg", "bicgstab", "gmres"):
            for folder, name in CASES:
                with self.subTest(method=method, name=name):
                    results = run(build(folder, name), linear_solver=method)
                    n = len(results.states)
                    u_ref = febio_log(os.path.join(folder, "febio", name + "_nodes.txt"), n, 3)
                    u = np.stack([s.displacement for s in results.states])
                    self.assertLess(abs(u - u_ref).max() / abs(u_ref).max(), 1e-9)

    def test_same_states_and_iterations_as_the_fixture(self):
        """The template reproduces the fixture of pull request 5."""
        from warp_febio.analysis.static import run as fixture_run
        from warp_febio.analysis.steps import run
        from warp_febio.compiled import compile

        for folder, name in CASES:
            with self.subTest(name):
                model = read_feb(os.path.join(folder, name + ".feb"))
                old = fixture_run(compile(model, device=DEVICE), linear_solver="direct")
                new = run(build(folder, name))
                self.assertEqual(
                    [s.iterations for s in old.states],
                    [s.iterations for s in new.states],
                )
                for a, b in zip(old.states, new.states):
                    self.assertEqual(a.time, b.time)
                    scale = max(abs(a.displacement).max(), 1e-30)
                    self.assertLess(abs(a.displacement - b.displacement).max() / scale, 1e-11)

    def test_tangent_is_minus_the_derivative_of_the_residual(self):
        from warp_febio.assembly.system import State
        from warp_febio.solvers.linear.warp_linear import dense

        rng = np.random.default_rng(0)
        for folder, name in CASES[:2]:
            with self.subTest(name):
                system = build(folder, name)
                u0 = 0.05 * rng.standard_normal((system.n, 3))

                def residual(u):
                    state = State(0.0, wp.array(u, dtype=wp.vec3d, device=DEVICE))
                    system.update(state)
                    return state, system.residual(state).numpy().reshape(-1)

                state, _ = residual(u0)
                system.tangent(state)
                K = dense(system.matrix())
                h, fd = 1e-6, np.zeros_like(K)
                for j in range(3 * system.n):
                    e = np.zeros(3 * system.n)
                    e[j] = h
                    e = e.reshape(-1, 3)
                    fd[:, j] = -(residual(u0 + e)[1] - residual(u0 - e)[1]) / (2 * h)
                self.assertLess(abs(K - fd).max() / abs(K).max(), 1e-8)
                self.assertLess(abs(K - K.T).max() / abs(K).max(), 1e-12)

    def test_adjoint_gradient_matches_finite_differences(self):
        """dJ/d(material parameters) from one transpose solve and one tape
        pass, against central differences of complete runs."""
        from warp_febio.analysis.steps import run
        from warp_febio.sensitivity.adjoint import parameter_gradient

        for folder, name in CASES[:3]:
            with self.subTest(name):
                system = build(folder, name)
                domain = system.terms[0]
                free = system.fields.free
                p0 = np.array(domain.material.param_values(0.0))

                def output(p):
                    domain.set_parameters(p)
                    _, states = run(system, keep_states=True)
                    u = states[-1].u.numpy()
                    return 0.5 * (u * u * free).sum(), states[-1]

                _, state = output(p0)
                gradients, _ = parameter_gradient(system, state, state.u.numpy() * free)
                adjoint = gradients[domain.name + ".material"][0]
                fd = np.zeros(len(p0))
                for k in range(len(p0)):
                    h = 1e-5 * max(abs(p0[k]), 1e-3)
                    dp = np.zeros(len(p0))
                    dp[k] = h
                    fd[k] = (output(p0 + dp)[0] - output(p0 - dp)[0]) / (2 * h)
                domain.set_parameters(p0)
                self.assertEqual(domain.parameter_names, ("E", "v"))
                # The displacements are prescribed, so the modulus has no
                # effect on them and Poisson's ratio does.
                self.assertLess(abs(adjoint[0]), 1e-10)
                self.assertLess(abs(fd[0]), 1e-10)
                self.assertGreater(abs(fd[1]), 1e-3)
                self.assertLess(abs(adjoint[1] - fd[1]) / abs(fd[1]), 1e-6)

    def test_unsupported_model_is_refused_with_the_report(self):
        from warp_febio.analysis.steps import build_system
        from warp_febio.coverage import CoverageError

        # A prescribed deformation is not supported.
        path = os.path.join(ROOT, "references", "small_problems", "inputs", "bk_ring_in_tube.feb")
        with self.assertRaises(CoverageError):
            build_system(read_feb(path), device=DEVICE)

    def test_open_mitral_input_is_reported_runnable(self):
        """With springs, a pressure in the step, dynamics, mass damping and
        the contact potential templated, the coverage report accepts the open
        mitral input. That is a statement about features, not about results:
        the run does not yet get through leaflet closure (work plan)."""
        from warp_febio.analysis.steps import inspect_model

        path = os.path.join(
            ROOT, "references", "febio", "originals", "cardiofem", "valve_pressurization", "mitral_p2prolapse.feb"
        )
        report = inspect_model(read_feb(path))
        self.assertTrue(report.runnable)
        notes = " ".join(e.note for e in report.entries)
        self.assertIn("w_tol is applied per point pair", notes)
        self.assertIn("scales the Newton tangent only", notes)


if __name__ == "__main__":
    unittest.main()
