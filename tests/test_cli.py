"""The command line: `python -m warp_febio inspect` and `run`.

Both go through the run path, so an input the run path can run is reported
as runnable and runs from the command line, on the device the tests use.
"""

import contextlib
import io
import os
import tempfile
import unittest

import numpy as np
from support import DEVICE, setUpModule, tearDownModule  # noqa: F401  (device for the module)

ROOT = os.path.join(os.path.dirname(__file__), "..", "references")
PRESSURE = os.path.join(ROOT, "single_element", "hex8_pressure.feb")
MITRAL = os.path.join(ROOT, "febio", "originals", "cardiofem", "valve_pressurization", "mitral_p2prolapse.feb")

try:
    import warp as wp

    from warp_febio.__main__ import main

    wp.config.quiet = True
    HAVE_WARP = True
except ImportError:
    HAVE_WARP = False


def call(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(argv)
    return code, out.getvalue()


@unittest.skipUnless(HAVE_WARP, "Warp is not installed (pip install 'warp_febio[warp]')")
class CommandLine(unittest.TestCase):
    def test_inspect_reports_what_the_run_path_can_run(self):
        """Pressure is a load of the run path, so the pressure case is runnable."""
        code, text = call(["inspect", PRESSURE])
        self.assertEqual(code, 0)
        self.assertIn("Verdict: runnable", text)

    def test_inspect_reports_an_input_that_cannot_run(self):
        code, text = call(["inspect", MITRAL])
        self.assertEqual(code, 1)
        self.assertIn("Verdict: not yet runnable", text)

    def test_run_refuses_an_input_that_cannot_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "mitral.xplt")
            code, text = call(["run", MITRAL, "-o", out, "--device", DEVICE])
            self.assertEqual(code, 1)
            self.assertIn("Verdict: not yet runnable", text)
            self.assertFalse(os.path.exists(out))

    def test_run_writes_a_plot_file_that_agrees_with_febio(self):
        from warp_febio.io import read_xplt

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "hex8_pressure.xplt")
            code, _ = call(["run", PRESSURE, "-o", out, "--device", DEVICE])
            self.assertEqual(code, 0)
            ours = read_xplt(out)
        ref = read_xplt(os.path.join(ROOT, "single_element", "febio", "hex8_pressure.xplt"))
        self.assertEqual(len(ours.states), len(ref.states))
        self.assertEqual([i.name for i in ours.dictionary["nodal"]], ["displacement"])
        u = np.asarray(ours.states[-1].data["nodal"][0][0])
        u_ref = np.asarray(ref.states[-1].data["nodal"][0][0])
        # Plot files hold single precision.
        self.assertLess(abs(u - u_ref).max() / abs(u_ref).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
