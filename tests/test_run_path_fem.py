"""Solid domain on warp.fem (hex8, tet4), against the dedicated brick domain,
FEBio results, and the gradient checker."""

import os
import unittest

import numpy as np

from warp_febio.io import read_feb

ROOT = os.path.join(os.path.dirname(__file__), "..")
SINGLE = os.path.join(ROOT, "references", "single_element")
SMALL = os.path.join(ROOT, "references", "small_meshes")

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


def build(folder, name, solids="dedicated"):
    from warp_febio.analysis.steps import build_system

    return build_system(read_feb(os.path.join(folder, name + ".feb")), device="cpu", solids=solids)


def compare_with_febio(test, folder, name, system, tol=1e-10):
    from warp_febio.analysis.steps import run

    results = run(system)
    n = len(results.states)
    u_ref = febio_log(os.path.join(folder, "febio", name + "_nodes.txt"), n, 3)
    s_ref = febio_log(os.path.join(folder, "febio", name + "_elements.txt"), n, 6)
    u = np.stack([s.displacement for s in results.states])
    s = np.stack([s.stress["Part1"] for s in results.states])
    test.assertLess(abs(u - u_ref).max() / abs(u_ref).max(), tol)
    test.assertLess(abs(s - s_ref).max() / abs(s_ref).max(), tol)


@unittest.skipUnless(HAVE_WARP, "Warp is not installed (pip install 'warp_febio[warp]')")
class FemSolidDomainTests(unittest.TestCase):
    def test_hex8_matches_the_dedicated_domain(self):
        """Two implementations of the same element: residual and tangent at a
        random state agree to rounding."""
        from warp_febio.assembly.system import State
        from warp_febio.elements.brick_domain import BrickDomain
        from warp_febio.elements.fem_domain import FemSolidDomain
        from warp_febio.solvers.linear.warp_linear import dense

        a = build(SMALL, "hex_block")
        b = build(SMALL, "hex_block", solids="warp.fem")
        self.assertIsInstance(a.terms[0], BrickDomain)
        self.assertIsInstance(b.terms[0], FemSolidDomain)
        u = 0.03 * np.random.default_rng(0).standard_normal((a.n, 3))
        out = []
        for system in (a, b):
            state = State(0.0, wp.array(u, dtype=wp.vec3d, device="cpu"))
            system.update(state)
            R = system.residual(state).numpy()
            system.tangent(state)
            out.append((R, dense(system.matrix())))
        (Ra, Ka), (Rb, Kb) = out
        self.assertLess(abs(Ra - Rb).max() / abs(Ra).max(), 1e-12)
        self.assertLess(abs(Ka - Kb).max() / abs(Ka).max(), 1e-12)

    def test_hex8_agrees_with_febio(self):
        for folder, name in ((SINGLE, "hex8_neo_hookean"), (SMALL, "hex_block")):
            with self.subTest(name):
                compare_with_febio(self, folder, name, build(folder, name, solids="warp.fem"))

    def test_tet4_agrees_with_febio(self):
        """tet4 comes from warp.fem alone; there is no hand-written tet."""
        from warp_febio.elements.fem_domain import FemSolidDomain

        system = build(SMALL, "tet4_block")
        self.assertIsInstance(system.terms[0], FemSolidDomain)
        self.assertEqual(system.terms[0].element_type, "tet4")
        compare_with_febio(self, SMALL, "tet4_block", system)

    def test_gradients_through_warp_fem(self):
        """The adjoint's reverse pass goes through warp.fem's integration."""
        from warp_febio.sensitivity.check import check_parameter_gradients

        system = build(SINGLE, "hex8_pressure", solids="warp.fem")
        top = system.fields.entries_of("displacement", [4, 5, 6, 7])

        def output(u):
            dJ_du = np.zeros_like(u)
            dJ_du[top, 2] = 0.25
            return u[top, 2].mean(), dJ_du

        rows = check_parameter_gradients(system, output)
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertGreater(abs(r.finite_difference), 1e-2)
            self.assertLess(r.relative_error, 1e-6)


if __name__ == "__main__":
    unittest.main()
