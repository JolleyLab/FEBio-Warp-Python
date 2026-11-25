"""Model, FEBio input and output, and the single-element end-to-end path.

The FEBio results used here are stored under references/ (see the README
there for how they were produced). FEBio itself is not needed to run these.
"""

import glob
import os
import tempfile
import unittest

import numpy as np

from warp_febio.io import read_feb, read_feb_string, read_xplt, write_feb, write_xplt

ROOT = os.path.join(os.path.dirname(__file__), "..")
SINGLE = os.path.join(ROOT, "references", "single_element")
ORIGINALS = os.path.join(ROOT, "references", "febio", "originals")
SMALL = os.path.join(ROOT, "references", "small_meshes")
CASES = ("hex8_neo_hookean", "quad4_shell_neo_hookean")


def febio_log(path, states, columns):
    with open(path) as f:
        rows = [line.split() for line in f if not line.startswith("*")]
    return np.array(rows, dtype=float)[:, 1:].reshape(states, -1, columns)


class FebReadWrite(unittest.TestCase):
    def test_round_trip_of_every_reference_input(self):
        files = glob.glob(
            os.path.join(ROOT, "references", "**", "*.feb"), recursive=True
        )
        self.assertGreaterEqual(len(files), 11)
        for path in files:
            with (
                self.subTest(os.path.basename(path)),
                tempfile.TemporaryDirectory() as tmp,
            ):
                model = read_feb(path)
                self.assertEqual(model.issues, [])
                out = os.path.join(tmp, "out.feb")
                write_feb(model, out)
                self.assertTrue(model.equals(read_feb(out)))

    def test_ids_are_kept(self):
        model = read_feb(
            os.path.join(
                ORIGINALS, "cardiofem", "valve_pressurization", "mitral_p2prolapse.feb"
            )
        )
        mesh = model.mesh
        self.assertEqual((len(mesh.nodes), mesh.blocks[0].count), (7442, 7200))
        # Connectivity is zero-based; the file's IDs come back through node_ids.
        self.assertEqual(mesh.blocks[0].connectivity.min(), 0)
        self.assertTrue(
            np.array_equal(mesh.node_ids[mesh.node_index(mesh.node_ids)], mesh.node_ids)
        )

    def test_only_spec_4_is_read(self):
        with self.assertRaises(ValueError):
            read_feb_string(
                '<febio_spec version="3.0"><Module type="solid"/></febio_spec>'
            )

    def test_unknown_section_is_reported_and_blocks_the_run(self):
        from warp_febio.compiled import inspect

        with open(os.path.join(SINGLE, "hex8_neo_hookean.feb")) as f:
            text = f.read()
        model = read_feb_string(
            text.replace("</febio_spec>", "<Mystery><a>1</a></Mystery></febio_spec>")
        )
        self.assertEqual([i.where for i in model.issues], ["Mystery"])
        self.assertFalse(inspect(model).runnable)


class PlotFiles(unittest.TestCase):
    def test_febio_files_rewrite_byte_for_byte(self):
        """FEBio 4.9 plot files, read and written again, are unchanged."""
        files = glob.glob(os.path.join(ORIGINALS, "**", "*.xplt"), recursive=True)
        self.assertEqual(len(files), 5)
        for path in files:
            with (
                self.subTest(os.path.basename(path)),
                tempfile.TemporaryDirectory() as tmp,
            ):
                out = os.path.join(tmp, "out.xplt")
                write_xplt(read_xplt(path), out)
                with open(path, "rb") as f, open(out, "rb") as g:
                    self.assertEqual(f.read(), g.read())

    def test_values_match_febio_log_data(self):
        for name in CASES:
            with self.subTest(name):
                pf = read_xplt(os.path.join(SINGLE, "febio", name + ".xplt"))
                u = febio_log(
                    os.path.join(SINGLE, "febio", name + "_nodes.txt"),
                    len(pf.states),
                    3,
                )
                s = febio_log(
                    os.path.join(SINGLE, "febio", name + "_elements.txt"),
                    len(pf.states),
                    6,
                )
                self.assertLess(abs(pf.variable("displacement") - u).max(), 1e-7)
                self.assertLess(abs(pf.variable("stress")[0] - s).max(), 1e-7)


class Coverage(unittest.TestCase):
    def test_mitral_input_is_reported_not_runnable(self):
        from warp_febio.compiled import compile, inspect
        from warp_febio.coverage import CoverageError

        model = read_feb(
            os.path.join(
                ORIGINALS, "cardiofem", "valve_pressurization", "mitral_p2prolapse.feb"
            )
        )
        report = inspect(model)
        missing = {e.febio_name for e in report.missing()}
        for name in (
            "isotropic Lee-Sacks",
            "contact potential",
            "nonlinear spring",
            "pressure",
            "mass damping",
            "DYNAMIC",
        ):
            self.assertIn(name, missing)
        with self.assertRaises(CoverageError):
            compile(model, device="cpu")


class SingleElement(unittest.TestCase):
    """Read a FEBio input, solve in Warp, write results, compare with FEBio."""

    def solve(self, name, backend="minimal", folder=SINGLE):
        from warp_febio.analysis.static import run
        from warp_febio.compiled import compile

        model = read_feb(os.path.join(folder, name + ".feb"))
        return model, run(compile(model, device="cpu", material_backend=backend))

    def compare(self, name, backend, folder=SINGLE):
        model, results = self.solve(name, backend, folder)
        n = len(results.states)
        u_ref = febio_log(os.path.join(folder, "febio", name + "_nodes.txt"), n, 3)
        s_ref = febio_log(os.path.join(folder, "febio", name + "_elements.txt"), n, 6)
        u = np.stack([s.displacement for s in results.states])
        s = np.stack([s.stress["Part1"] for s in results.states])
        # FEBio prints 12 significant digits; 1e-10 relative leaves margin.
        self.assertLess(abs(u - u_ref).max() / abs(u_ref).max(), 1e-10)
        self.assertLess(abs(s - s_ref).max() / abs(s_ref).max(), 1e-10)
        return model, results

    def test_agrees_with_febio(self):
        for name in CASES:
            with self.subTest(name):
                self.compare(name, "minimal")

    def test_small_meshes_agree_with_febio(self):
        """Eight hexahedra, and nine shells on a curved surface."""
        for name in ("hex_block", "shell_patch"):
            with self.subTest(name):
                self.compare(name, "minimal", SMALL)

    def test_agrees_with_febio_using_the_material_layer(self):
        from warp_febio.materials.minimal import interface_kernel

        if interface_kernel("neo-Hookean") is None:
            self.skipTest(
                "the material layer of draft pull request #2 is not installed"
            )
        for name in CASES:
            with self.subTest(name):
                self.compare(name, "interface")

    def test_plot_file_has_febio_structure(self):
        """Our plot file matches FEBio's for the same input in everything but
        the software name and float rounding."""
        for name in CASES:
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                model, results = self.solve(name)
                out = os.path.join(tmp, name + ".xplt")
                write_xplt(model, results, out)
                a = read_xplt(os.path.join(SINGLE, "febio", name + ".xplt"))
                b = read_xplt(out)
                self.assertEqual(a.version, b.version)
                self.assertEqual(a.dictionary, b.dictionary)
                self.assertEqual(a.parts, b.parts)
                self.assertTrue(np.array_equal(a.node_ids, b.node_ids))
                self.assertTrue(np.array_equal(a.nodes, b.nodes))
                self.assertTrue(np.array_equal(a.times, b.times))
                for x, y in zip(a.domains, b.domains):
                    self.assertEqual(
                        (x.elem_type, x.part_id, x.name),
                        (y.elem_type, y.part_id, y.name),
                    )
                    self.assertTrue(np.array_equal(x.elem_ids, y.elem_ids))
                    self.assertTrue(np.array_equal(x.connectivity, y.connectivity))
                for p, q in (
                    (a.node_sets, b.node_sets),
                    (a.element_sets, b.element_sets),
                ):
                    self.assertEqual(
                        [(i, n, list(v)) for i, n, v in p],
                        [(i, n, list(v)) for i, n, v in q],
                    )
                for p, q in ((a.surfaces, b.surfaces), (a.facet_sets, b.facet_sets)):
                    # Compare the used entries; FEBio leaves the padding of a facet uninitialized.
                    faces = lambda L: [
                        (
                            s.id,
                            s.name,
                            s.max_nodes,
                            [list(f[: 2 + f[1]]) for f in s.faces],
                        )
                        for s in L
                    ]
                    self.assertEqual(faces(p), faces(q))
                for x, y in zip(a.states, b.states):
                    self.assertTrue(np.array_equal(x.element_state, y.element_state))
                self.assertLess(
                    abs(a.variable("displacement") - b.variable("displacement")).max(),
                    1e-7,
                )
                self.assertLess(
                    abs(a.variable("stress")[0] - b.variable("stress")[0]).max(), 1e-7
                )
                # Same size once the software name is accounted for.
                size = lambda p: os.path.getsize(p)
                self.assertEqual(
                    size(out) - len(b.software),
                    size(os.path.join(SINGLE, "febio", name + ".xplt"))
                    - len(a.software),
                )

    def test_tangent_is_the_derivative_of_the_residual(self):
        import warp as wp

        from warp_febio.analysis.static import _dense, evaluate
        from warp_febio.compiled import State, compile

        rng = np.random.default_rng(0)
        for name in CASES:
            with self.subTest(name):
                cm = compile(
                    read_feb(os.path.join(SINGLE, name + ".feb")), device="cpu"
                )
                u = 0.05 * rng.standard_normal(cm.ndof)
                at = lambda v: evaluate(
                    cm, State(0.0, wp.array(v, dtype=wp.float64, device="cpu"))
                )
                R, K = at(u)
                K = _dense(K)
                h = 1e-6
                fd = np.zeros_like(K)
                for j in range(cm.ndof):
                    e = np.zeros(cm.ndof)
                    e[j] = h
                    fd[:, j] = -(at(u + e)[0].numpy() - at(u - e)[0].numpy()) / (2 * h)
                self.assertLess(abs(K - fd).max() / abs(K).max(), 1e-8)
                self.assertLess(abs(K - K.T).max() / abs(K).max(), 1e-12)


class VtkOutput(unittest.TestCase):
    def setUp(self):
        try:
            import h5py  # noqa: F401
            import vtk  # noqa: F401
        except ImportError:
            self.skipTest("vtk and h5py are needed for these checks")

    def test_vtkhdf_reads_back_in_double(self):
        import vtk
        from vtk.util import numpy_support as ns

        from warp_febio.io import write_vtkhdf

        for name in CASES:
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                model, results = SingleElement.solve(self, name)
                out = os.path.join(tmp, name + ".vtkhdf")
                write_vtkhdf(model, results, out)
                reader = vtk.vtkHDFReader()
                reader.SetFileName(out)
                reader.Update()
                self.assertEqual(reader.GetNumberOfSteps(), len(results.states))
                for k, s in enumerate(results.states):
                    reader.SetStep(k)
                    reader.Update()
                    grid = reader.GetOutput()
                    d = ns.vtk_to_numpy(grid.GetPointData().GetArray("displacement"))
                    st = ns.vtk_to_numpy(grid.GetCellData().GetArray("stress"))
                    self.assertTrue(np.array_equal(d, s.displacement))
                    self.assertTrue(np.array_equal(st, s.stress["Part1"]))

    def test_vtk_grid_to_model_to_feb(self):
        """The Slicer direction: a VTK mesh becomes a Model and a .feb that reads back."""
        from warp_febio.io.vtk_adapter import model_from_vtk, model_to_vtk

        for name in CASES:
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                source = read_feb(os.path.join(SINGLE, name + ".feb"))
                grid = model_to_vtk(source)
                model = model_from_vtk(
                    grid,
                    "neo-Hookean",
                    {"density": 1, "E": 1.0, "v": 0.3},
                    shell_thickness=0.1,
                )
                self.assertTrue(np.array_equal(model.mesh.nodes, source.mesh.nodes))
                self.assertTrue(
                    np.array_equal(
                        model.mesh.blocks[0].connectivity,
                        source.mesh.blocks[0].connectivity,
                    )
                )
                out = os.path.join(tmp, "from_vtk.feb")
                write_feb(model, out)
                self.assertTrue(model.equals(read_feb(out)))


if __name__ == "__main__":
    unittest.main()
