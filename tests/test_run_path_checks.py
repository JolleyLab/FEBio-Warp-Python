"""Checks a reviewer would ask for beyond the single-domain cases: two
materials in two domains with reaction forces, FEBio's automatic time
stepper (step sequence, must points, iteration counts), and what happens when
a step does not converge. All against FEBio results for the same inputs."""

import os
import unittest

import numpy as np
from support import DEVICE, setUpModule, tearDownModule  # noqa: F401  (device for the module)

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


@unittest.skipUnless(HAVE_WARP, "Warp is not installed (pip install 'warp_febio[warp]')")
class TwoMaterialTests(unittest.TestCase):
    """references/small_meshes/two_material_pressure.feb: two hex8 domains
    (neo-Hookean and isotropic elastic), pressure on the top."""

    def check(self, solids):
        from warp_febio.analysis.steps import build_system, run

        model = read_feb(os.path.join(SMALL, "two_material_pressure.feb"))
        results = run(build_system(model, device=DEVICE, solids=solids))
        n = len(results.states)
        ref = os.path.join(SMALL, "febio", "two_material_pressure_")
        u_ref = febio_log(ref + "nodes.txt", n, 3)
        r_ref = febio_log(ref + "reactions.txt", n, 3)
        s_ref = febio_log(ref + "elements.txt", n, 6)
        u = np.stack([s.displacement for s in results.states])
        r = np.stack([s.reaction_force for s in results.states])
        s = np.stack([np.concatenate([s.stress[d.name] for d in model.domains]) for s in results.states])
        self.assertLess(abs(u - u_ref).max() / abs(u_ref).max(), 1e-9)
        self.assertLess(abs(s - s_ref).max() / abs(s_ref).max(), 1e-9)
        # FEBio's nodal R: minus the internal force, zero at fixed components.
        self.assertLess(abs(r - r_ref).max() / abs(r_ref).max(), 1e-9)
        self.assertEqual([s.iterations for s in results.states[1:]], [7, 9, 12, 15])

    def test_dedicated_domain(self):
        self.check("dedicated")

    def test_warp_fem_domain(self):
        # Two materials on warp.fem: each material needs its own integrands,
        # because warp.fem caches generated kernels by integrand name.
        self.check("warp.fem")


@unittest.skipUnless(HAVE_WARP, "Warp is not installed (pip install 'warp_febio[warp]')")
class AutoStepTests(unittest.TestCase):
    """references/single_element/hex8_pressure_autostep.feb."""

    CASE = os.path.join(SINGLE, "hex8_pressure_autostep.feb")

    def test_step_sequence_matches_febio(self):
        from warp_febio.analysis.steps import build_system, run

        results = run(build_system(read_feb(self.CASE), device=DEVICE))
        ref = np.loadtxt(os.path.join(SINGLE, "febio", "hex8_pressure_autostep_steps.txt"))
        states = results.states[1:]
        self.assertEqual(len(states), len(ref))
        np.testing.assert_allclose([s.time for s in states], ref[:, 0], rtol=2e-6)
        self.assertEqual([s.iterations for s in states], list(ref[:, 1].astype(int)))
        n = len(results.states)
        u_ref = febio_log(os.path.join(SINGLE, "febio", "hex8_pressure_autostep_nodes.txt"), n, 3)
        u = np.stack([s.displacement for s in results.states])
        self.assertLess(abs(u - u_ref).max() / abs(u_ref).max(), 1e-6)

    def test_failed_steps_cut_back_and_stop_like_febio(self):
        # With max_refs 8 this input needs more iterations than allowed late
        # in the step. FEBio 4 (commit e4df3f8) converges the ten steps below,
        # fails 13 attempts in total and stops after five retries in a row.
        from warp_febio.analysis.steps import build_system, run
        from warp_febio.assembly.system import SolveError

        model = read_feb(self.CASE)
        model.steps[0].control.child("solver").set("max_refs", 8)
        log = []
        with self.assertRaises(SolveError):
            run(build_system(model, device=DEVICE), log=log.append)
        times = [float(line.split()[3]) for line in log if line.startswith("converged")]
        febio = [
            0.1,
            0.24,
            0.35,
            0.505271,
            0.649024,
            0.782114,
            0.8,
            0.899817,
            0.943039,
            0.955516,
        ]
        np.testing.assert_allclose(times, febio, rtol=2e-6)
        retried = sum("retrying" in line for line in log)
        self.assertEqual(retried + 1, 13)


if __name__ == "__main__":
    unittest.main()
