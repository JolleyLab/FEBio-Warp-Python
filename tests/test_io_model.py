"""Model, FEBio input and output, and the single-element end-to-end path.

The FEBio results used here are stored under references/ (see the README
there for how they were produced). FEBio itself is not needed to run these.
"""

import glob
import os
import tempfile
import unittest

import numpy as np
from support import DEVICE, full_tier, setUpModule, tearDownModule  # noqa: F401  (device for the module)

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


try:
    import warp  # noqa: F401

    HAVE_WARP = True
except ImportError:
    HAVE_WARP = False


needs_warp = unittest.skipUnless(HAVE_WARP, "Warp is not installed (pip install 'warp_febio[warp]')")


class WithoutWarp(unittest.TestCase):
    def test_model_and_io_import_without_warp(self):
        """The model and the file layers must load where Warp is absent, as in
        3D Slicer. Run in a fresh interpreter with the import of warp blocked."""
        import subprocess
        import sys

        code = (
            "import sys; sys.modules['warp'] = None\n"
            "import warp_febio, warp_febio.model\n"
            "from warp_febio.io import read_feb, write_feb, read_xplt, write_xplt\n"
            f"m = read_feb({os.path.join(SINGLE, 'hex8_neo_hookean.feb')!r})\n"
            "assert len(m.mesh.nodes) == 8\n"
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)


class FebReadWrite(unittest.TestCase):
    def test_round_trip_of_every_reference_input(self):
        files = glob.glob(os.path.join(ROOT, "references", "**", "*.feb"), recursive=True)
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
        model = read_feb(os.path.join(ORIGINALS, "cardiofem", "valve_pressurization", "mitral_p2prolapse.feb"))
        mesh = model.mesh
        self.assertEqual((len(mesh.nodes), mesh.blocks[0].count), (7442, 7200))
        # Connectivity is zero-based; the file's IDs come back through node_ids.
        self.assertEqual(mesh.blocks[0].connectivity.min(), 0)
        self.assertTrue(np.array_equal(mesh.node_ids[mesh.node_index(mesh.node_ids)], mesh.node_ids))

    def test_original_text_is_written_back(self):
        """Values are written as they stood in the file unless they were changed."""
        with open(os.path.join(SINGLE, "hex8_neo_hookean.feb")) as f:
            text = f.read().replace("<E>1</E>", "<E>1.00</E>").replace("<v>0.3</v>", "<v>3.0E-1</v>")
        model = read_feb_string(text)
        self.assertEqual(model.materials[0].value("E"), 1.0)
        model.materials[0].params.set("v", 0.25)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "o.feb")
            write_feb(model, out)
            with open(out) as f:
                written = f.read()
        self.assertIn("<E>1.00</E>", written)
        self.assertIn("<v>0.25</v>", written)

    def test_reference_to_another_file_is_reported(self):
        model = read_feb_string(
            '<febio_spec version="4.0"><Module type="solid"/><Include>mesh.feb</Include></febio_spec>'
        )
        self.assertEqual([i.where for i in model.issues], ["Include"])

    def test_only_spec_4_is_read(self):
        with self.assertRaises(ValueError):
            read_feb_string('<febio_spec version="3.0"><Module type="solid"/></febio_spec>')

    @needs_warp
    def test_unknown_section_is_reported_and_blocks_the_run(self):
        from warp_febio.compiled import inspect

        with open(os.path.join(SINGLE, "hex8_neo_hookean.feb")) as f:
            text = f.read()
        model = read_feb_string(text.replace("</febio_spec>", "<Mystery><a>1</a></Mystery></febio_spec>"))
        self.assertEqual([i.where for i in model.issues], ["Mystery"])
        self.assertFalse(inspect(model).runnable)


# A small input with what a well-formed FEBio file rarely has: a section that
# appears twice, carried sections out of FEBio Studio's order, attributes the
# reader has no field for, and a set name used twice.
UNUSUAL = """<febio_spec version="4.0">
<Module type="solid"/>
<Material><material id="1" name="m" type="neo-Hookean"><E>1</E><v>0.3</v></material></Material>
<Mesh>
<Nodes name="Obj">
<node id="1">0,0,0</node><node id="2">1,0,0</node><node id="3">1,1,0</node><node id="4">0,1,0</node>
<node id="5">0,0,1</node><node id="6">1,0,1</node><node id="7">1,1,1</node><node id="8">0,1,1</node>
</Nodes>
<Elements type="hex8" name="Part1" extra="kept"><elem id="1">1,2,3,4,5,6,7,8</elem></Elements>
{sets}
</Mesh>
<MeshDomains><SolidDomain name="Part1" mat="m"/></MeshDomains>
<MeshAdaptor><mesh_adaptor type="a"/></MeshAdaptor>
<Constraints><constraint type="x"/></Constraints>
<Rigid><rigid_bc type="r"/></Rigid>
<Constraints><constraint type="y"/></Constraints>
<Step><step id="1" name="s1" extra="kept"><Control><time_steps>1</time_steps></Control></step></Step>
</febio_spec>"""


class NothingDroppedUnseen(unittest.TestCase):
    """What the reader has no field for is either written back or reported."""

    def round_trip(self, model):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.feb")
            write_feb(model, out)
            with open(out, encoding="ISO-8859-1") as f:
                return read_feb(out), f.read()

    def test_repeated_sections_and_extra_attributes_are_written_back(self):
        model = read_feb_string(UNUSUAL.format(sets=""))
        self.assertEqual(model.issues, [])
        again, text = self.round_trip(model)
        self.assertTrue(model.equals(again))
        # Both Constraints sections, in their order, and the extra attributes.
        self.assertLess(text.index('type="x"'), text.index('type="y"'))
        self.assertEqual(text.count("<Constraints>"), 2)
        self.assertIn('<Elements type="hex8" name="Part1" extra="kept">', text)
        self.assertIn('<step id="1" name="s1" extra="kept">', text)
        self.assertEqual(again.mesh.blocks[0].attrs, {"extra": "kept"})
        self.assertEqual(again.steps[0].attrs, {"extra": "kept"})

    def test_section_order_follows_febio_studio_and_content_is_equal(self):
        """The writer places carried sections where FEBio Studio writes them
        (Rigid before Constraints, others after), so their order can change;
        equality compares them tag by tag."""
        model = read_feb_string(UNUSUAL.format(sets=""))
        again, _ = self.round_trip(model)
        self.assertEqual(
            [r.tag for r in model.other_sections],
            ["MeshAdaptor", "Constraints", "Rigid", "Constraints"],
        )
        self.assertEqual(
            [r.tag for r in again.other_sections],
            ["Rigid", "Constraints", "Constraints", "MeshAdaptor"],
        )
        self.assertTrue(model.equals(again))
        # A change inside one of the repeated sections is still a difference.
        again.other_sections[1].children[0].attrs["type"] = "changed"
        self.assertFalse(model.equals(again))

    def test_attributes_and_names_the_reader_cannot_keep_are_reported(self):
        sets = '<NodeSet name="a" flag="1">1,2</NodeSet><NodeSet name="b">1,2</NodeSet><NodeSet name="b">3,4</NodeSet>'
        model = read_feb_string(UNUSUAL.format(sets=sets))
        found = {(i.where, i.message.split(";")[0]) for i in model.issues}
        self.assertIn(("Mesh/NodeSet 'a'", "attribute(s) flag not read"), found)
        self.assertIn(("Mesh/NodeSet 'b'", "name used more than once"), found)

    def test_malformed_input_is_named(self):
        from warp_febio.io import FebFormatError

        head = '<febio_spec version="4.0"><Mesh><Nodes>'
        for text, expected in (
            (
                head + "<node>0,0,0</node></Nodes></Mesh></febio_spec>",
                "without an 'id'",
            ),
            (
                head + '<node id="1">0,0,0</node></Nodes><Elements type="tri3" '
                'name="p"><elem id="1">1,2,3</elem></Elements></Mesh></febio_spec>',
                "Mesh/Elements 'p': refers to a node ID that is not in the mesh",
            ),
        ):
            with self.assertRaises(FebFormatError) as caught:
                read_feb_string(text)
            self.assertIn(expected, str(caught.exception))

    def test_truncated_plot_file_is_named(self):
        with open(os.path.join(SINGLE, "febio", CASES[0] + ".xplt"), "rb") as f:
            data = f.read()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cut.xplt")
            with open(path, "wb") as f:
                f.write(data[: len(data) // 2])
            with self.assertRaises(ValueError) as caught:
                read_xplt(path)
            self.assertIn("truncated or corrupt", str(caught.exception))


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


class LoadCurves(unittest.TestCase):
    def test_interpolation_follows_febio_point_curve(self):
        from warp_febio.model import LoadController, Record

        pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [4.0, 2.0]])

        def curve(mode):
            rec = Record("load_controller")
            rec.set("interpolate", mode)
            return LoadController(1, "loadcurve", rec, pts)

        self.assertAlmostEqual(curve("LINEAR").value(0.25), 0.25)
        self.assertAlmostEqual(curve("LINEAR").value(9.0), 2.0)  # constant extension
        # STEP takes the value of the first point later than the time. At an
        # interior point's own time that is the following point (PointCurve's
        # binarySearch); only the first and last points return their own value.
        self.assertEqual(curve("STEP").value(0.5), 1.0)
        self.assertEqual(curve("STEP").value(1.0), 0.0)
        self.assertEqual(curve("STEP").value(1.5), 0.0)
        self.assertEqual(curve("STEP").value(4.0), 2.0)
        # SMOOTH passes through the points and is quadratic in the first interval.
        smooth = curve("SMOOTH")
        for t, v in pts:
            self.assertAlmostEqual(smooth.value(t), v)
        self.assertAlmostEqual(smooth.value(0.5), 0.75)  # parabola through (0,0), (1,1), (2,0)


@needs_warp
class Coverage(unittest.TestCase):
    def test_shell_on_solid_is_refused(self):
        """Found by the FEBio suite (sh48, sh50): shells sharing nodes with solids
        need FEBio's shell-solid interface, which is not implemented."""
        from warp_febio.compiled import inspect
        from warp_febio.model import Domain, ElementBlock, Record

        model = read_feb(os.path.join(SINGLE, "hex8_neo_hookean.feb"))
        model.mesh.blocks.append(
            ElementBlock(
                "Skin",
                "quad4",
                np.array([[4, 5, 6, 7]], np.int32),
                np.array([2], np.int32),
            )
        )
        rec = Record("ShellDomain", {"name": "Skin", "mat": "Material1"})
        rec.set("shell_thickness", 0.1)
        model.domains.append(Domain("ShellDomain", "Skin", "Material1", rec))
        self.assertIn(
            "shell attached to solid elements",
            {e.febio_name for e in inspect(model).missing()},
        )

    def test_mitral_input_is_reported_not_runnable(self):
        from warp_febio.compiled import compile, inspect
        from warp_febio.coverage import CoverageError

        model = read_feb(os.path.join(ORIGINALS, "cardiofem", "valve_pressurization", "mitral_p2prolapse.feb"))
        report = inspect(model)
        missing = {e.febio_name for e in report.missing()}
        # This is the fixture's report, without what the run path adds
        # (springs, loads, dynamics, contact). FEBio's tangent_scale (10
        # here) multiplies c0 in the tangent only
        # (FEIsotropicLeeSacks::Tangent), so it is listed as having no effect
        # on the solution and not as missing.
        scale = next(e for e in report.entries if e.febio_name == "isotropic Lee-Sacks.tangent_scale")
        self.assertEqual(scale.status, "no effect here")
        for name in (
            "contact potential",
            "nonlinear spring",
            "pressure",
            "mass damping",
            "DYNAMIC",
        ):
            self.assertIn(name, missing)
        with self.assertRaises(CoverageError):
            compile(model, device=DEVICE)


@needs_warp
class SingleElement(unittest.TestCase):
    """Read a FEBio input, solve in Warp, write results, compare with FEBio."""

    def solve(self, name, folder=SINGLE, linear_solver="direct"):
        from warp_febio.analysis.static import run
        from warp_febio.compiled import compile

        model = read_feb(os.path.join(folder, name + ".feb"))
        compiled = compile(model, device=DEVICE)
        return model, run(compiled, linear_solver=linear_solver)

    def compare(self, name, folder=SINGLE, linear_solver="direct"):
        model, results = self.solve(name, folder, linear_solver)
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
                self.compare(name)

    @full_tier("the fixture with every iterative solver on every case, about 4 minutes on CPU")
    def test_agrees_with_febio_using_warp_linear_solvers(self):
        """The same comparison with Warp's iterative solvers on the device."""
        for method in ("cg", "bicgstab", "gmres"):
            for name, folder in (
                (CASES[0], SINGLE),
                (CASES[1], SINGLE),
                ("hex_block", SMALL),
                ("shell_patch", SMALL),
            ):
                with self.subTest(method=method, name=name):
                    self.compare(name, folder, method)

    def test_small_meshes_agree_with_febio(self):
        """Eight hexahedra, and nine shells on a curved surface."""
        for name in ("hex_block", "shell_patch"):
            with self.subTest(name):
                self.compare(name, SMALL)

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
                self.assertLess(abs(a.variable("stress")[0] - b.variable("stress")[0]).max(), 1e-7)
                # Same size once the software name is accounted for.
                size = lambda p: os.path.getsize(p)
                self.assertEqual(
                    size(out) - len(b.software),
                    size(os.path.join(SINGLE, "febio", name + ".xplt")) - len(a.software),
                )

    def test_tangent_is_the_derivative_of_the_residual(self):
        import warp as wp

        from warp_febio.analysis.static import dense, evaluate, tangent_matrix
        from warp_febio.compiled import State, compile

        rng = np.random.default_rng(0)
        for name in CASES:
            with self.subTest(name):
                cm = compile(read_feb(os.path.join(SINGLE, name + ".feb")), device=DEVICE)
                u = 0.05 * rng.standard_normal(cm.ndof)
                at = lambda v: evaluate(cm, State(0.0, wp.array(v, dtype=wp.float64, device=DEVICE)))
                K = dense(tangent_matrix(cm, at(u)[1]))
                h = 1e-6
                fd = np.zeros_like(K)
                for j in range(cm.ndof):
                    e = np.zeros(cm.ndof)
                    e[j] = h
                    fd[:, j] = -(at(u + e)[0].numpy() - at(u - e)[0].numpy()) / (2 * h)
                self.assertLess(abs(K - fd).max() / abs(K).max(), 1e-8)
                self.assertLess(abs(K - K.T).max() / abs(K).max(), 1e-12)


@needs_warp
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
