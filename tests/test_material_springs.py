"""FEBio's nonlinear spring (chordae) and linear truss against FEBio's source.

The chord parameters are read from the repository's mitral input, so the checks
run on the values the valve model uses. The comparison of the spring with a
FEBio run is in the verification draft (pull request 4), next to the reference
it uses.
"""

import os
import pathlib
import unittest

import febio_closed_forms
import numpy as np
import warp as wp

from warp_febio.materials.models.linear_truss import LINEAR_TRUSS, make_truss_function
from warp_febio.materials.models.nonlinear_spring import (
    make_point_function,
    make_spring_kernel,
    nonlinear_spring,
)
from warp_febio.materials.uniaxial import make_update_kernel

# WARP_FEBIO_TEST_DEVICE selects the device ("cpu" by default, "cuda:0" for a
# GPU). It becomes Warp's current device while this module's tests run, so
# calls that do not name a device follow it.
DEVICE = os.environ.get("WARP_FEBIO_TEST_DEVICE", "cpu")
_device_scope = wp.ScopedDevice(DEVICE)


def setUpModule():
    _device_scope.__enter__()


def tearDownModule():
    _device_scope.__exit__(None, None, None)


wp.config.quiet = True

ROOT = pathlib.Path(__file__).resolve().parents[1]
MITRAL = ROOT / "references/febio/originals/cardiofem/valve_pressurization/mitral_p2prolapse.feb"

# Slack of 4 with unit slope beyond it, as in the mitral input's group A1.
CHORD = dict(scale=60.0, points=[(0.0, 0.0), (4.0, 0.0), (5.0, 1.0)])
# A curve with distinct slopes on every segment, to exercise segment selection.
GENERAL = dict(scale=2.5, points=[(-1.0, -0.2), (0.5, 0.1), (2.0, 0.1), (3.0, 1.4), (6.0, 2.0)])


def params_of(model, scale, points):
    values = {"scale": scale}
    for k, (x, y) in enumerate(points):
        values[f"x{k}"] = x
        values[f"y{k}"] = y
    return values


def evaluate_springs(model, X, x, nodes, rows, region=None, dtype=wp.float64):
    """Force (n, 3), stiffness (n, 3, 3) and measure (n) for a batch of springs."""
    n = len(nodes)
    vec3 = wp.types.vector(length=3, dtype=dtype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=dtype)
    pvec = model.param_vector(dtype)
    force = wp.zeros(n, dtype=vec3)
    stiffness = wp.zeros(n, dtype=mat33)
    measure = wp.zeros(n, dtype=dtype)
    wp.launch(
        make_spring_kernel(model, dtype),
        dim=n,
        inputs=[
            wp.array(X, dtype=vec3),
            wp.array(x, dtype=vec3),
            wp.array(np.asarray(nodes), dtype=int),
            wp.array(np.zeros(n, int) if region is None else region, dtype=int),
            wp.array([model.pack(dtype, **r) for r in rows], dtype=pvec),
        ],
        outputs=[force, stiffness, measure],
    )
    return force.numpy(), stiffness.numpy(), measure.numpy()


def random_springs(n, seed=0, stretch=(0.6, 1.9)):
    """n springs of reference length 5 to 20 with random directions and stretches."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2 * n, 3)) * 10.0
    d0 = rng.normal(size=(n, 3))
    d0 *= (rng.uniform(5.0, 20.0, n) / np.linalg.norm(d0, axis=1))[:, None]
    X[1::2] = X[0::2] + d0
    d1 = rng.normal(size=(n, 3))
    d1 /= np.linalg.norm(d1, axis=1)[:, None]
    L0 = np.linalg.norm(d0, axis=1)
    x = X.copy()
    x[1::2] = x[0::2] + d1 * (L0 * rng.uniform(*stretch, n))[:, None]
    nodes = np.arange(2 * n).reshape(n, 2)
    return X, x, nodes


def read_mitral_chords():
    """Chord materials, node pairs and node coordinates from the mitral input.

    Read with the standard library so the test does not depend on a reader.
    """
    if not MITRAL.exists():
        return None
    import xml.etree.ElementTree as ET

    root = ET.parse(MITRAL).getroot()
    ids, coords = [], []
    for node in root.iter("node"):
        ids.append(int(node.get("id")))
        coords.append([float(v) for v in node.text.split(",")])
    index_of = {i: k for k, i in enumerate(ids)}
    sets = {s.get("name"): s for s in root.find("Mesh").findall("DiscreteSet")}
    discrete = root.find("Discrete")
    materials = {m.get("id"): m for m in discrete.findall("discrete_material")}
    groups = []
    for link in discrete.findall("discrete"):
        mat = materials[link.get("dmat")]
        force = mat.find("force")
        pairs = np.array(
            [[index_of[int(v)] for v in e.text.split(",")] for e in sets[link.get("discrete_set")].findall("delem")]
        )
        groups.append(
            dict(
                name=mat.get("name"),
                type=mat.get("type"),
                scale=float(mat.findtext("scale")),
                measure=mat.findtext("measure", "elongation"),
                function=force.get("type"),
                interpolate=force.findtext("interpolate", "linear"),
                extend=force.findtext("extend", "constant"),
                points=[tuple(float(v) for v in pt.text.split(",")) for pt in force.find("points").findall("pt")],
                pairs=pairs,
            )
        )
    return np.array(coords), groups


class PointFunctionTests(unittest.TestCase):
    def test_value_and_slope_match_febio_point_curve(self):
        for extend in ("extrapolate", "constant"):
            pts = GENERAL["points"]
            n = len(pts)
            f = make_point_function(n, extend)
            pvec = wp.types.vector(length=2 * n, dtype=wp.float64)

            @wp.kernel(module="unique")
            def k(
                s: wp.array(dtype=wp.float64),
                p: wp.array(dtype=pvec),
                v: wp.array(dtype=wp.float64),
                m: wp.array(dtype=wp.float64),
            ):
                i = wp.tid()
                a, b = f(s[i], p[0])
                v[i] = a
                m[i] = b

            # Samples inside every segment, beyond both ends, and on every knot.
            s = np.concatenate([np.linspace(-3.0, 9.0, 121) + 0.013, [x for x, _ in pts]])
            v = wp.zeros(len(s), dtype=wp.float64)
            m = wp.zeros(len(s), dtype=wp.float64)
            flat = [c for pt in pts for c in pt]
            wp.launch(
                k,
                dim=len(s),
                inputs=[
                    wp.array(s, dtype=wp.float64),
                    wp.array([pvec(*flat)], dtype=pvec),
                ],
                outputs=[v, m],
            )
            ref_v = [febio_closed_forms.point_curve_linear(t, pts, extend) for t in s]
            ref_m = [febio_closed_forms.point_curve_linear_slope(t, pts, extend) for t in s]
            np.testing.assert_allclose(v.numpy(), ref_v, rtol=0, atol=1e-14)
            np.testing.assert_allclose(m.numpy(), ref_m, rtol=0, atol=1e-14)

    def test_exact_slope_against_febio_finite_difference(self):
        """FEBio differentiates the curve numerically; the two agree away from knots."""
        pts = GENERAL["points"]
        knots = np.array([x for x, _ in pts])
        worst = 0.0
        for t in np.linspace(-3.0, 9.0, 121) + 0.013:
            if np.abs(knots - t).min() < 1e-6:
                continue
            exact = febio_closed_forms.point_curve_linear_slope(t, pts)
            febio = febio_closed_forms.febio_point_curve_derive(t, pts)
            worst = max(worst, abs(exact - febio))
        self.assertLess(worst, 1e-6)
        # On an interior knot FEBio's central difference returns the mean slope.
        t = pts[1][0]
        left = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
        right = (pts[2][1] - pts[1][1]) / (pts[2][0] - pts[1][0])
        self.assertAlmostEqual(
            febio_closed_forms.febio_point_curve_derive(t, pts),
            0.5 * (left + right),
            places=6,
        )
        self.assertAlmostEqual(febio_closed_forms.point_curve_linear_slope(t, pts), right, places=14)

    def test_rejects_unsupported_choices(self):
        with self.assertRaises(ValueError):
            make_point_function(1)
        with self.assertRaises(ValueError):
            make_point_function(3, extend="repeat")
        with self.assertRaises(ValueError):
            nonlinear_spring(3, measure="log strain")


class NonlinearSpringTests(unittest.TestCase):
    def check_against_febio(self, spec, measure, extend, seed):
        model = nonlinear_spring(len(spec["points"]), measure, extend)
        row = params_of(model, spec["scale"], spec["points"])
        stretch = (0.6, 1.9) if measure != "elongation" else (0.4, 2.2)
        X, x, nodes = random_springs(200, seed, stretch)
        force, stiffness, s = evaluate_springs(model, X, x, nodes, [row])
        worst_f = worst_k = 0.0
        scale_f = scale_k = 1e-300
        for i, (a, b) in enumerate(nodes):
            f, K, sm = febio_closed_forms.nonlinear_spring(
                X[b] - X[a], x[b] - x[a], spec["scale"], spec["points"], measure, extend
            )
            worst_f = max(worst_f, np.abs(force[i] - f).max())
            worst_k = max(worst_k, np.abs(stiffness[i] - K).max())
            scale_f = max(scale_f, np.abs(f).max())
            scale_k = max(scale_k, np.abs(K).max())
            self.assertAlmostEqual(s[i], sm, places=12)
        self.assertLess(worst_f / scale_f, 1e-13)
        self.assertLess(worst_k / scale_k, 1e-13)

    def test_force_and_stiffness_match_febio_source(self):
        seed = 0
        for spec in (CHORD, GENERAL):
            for measure in ("elongation", "strain", "stretch"):
                for extend in ("extrapolate", "constant"):
                    with self.subTest(measure=measure, extend=extend):
                        seed += 1
                        self.check_against_febio(spec, measure, extend, seed)

    def test_stiffness_is_derivative_of_force(self):
        for measure in ("elongation", "strain", "stretch"):
            model = nonlinear_spring(len(GENERAL["points"]), measure)
            row = params_of(model, GENERAL["scale"], GENERAL["points"])
            X, x, nodes = random_springs(60, 7)
            _, K, _ = evaluate_springs(model, X, x, nodes, [row])
            h = 1e-6
            worst = 0.0
            for j in range(3):
                xp, xm = x.copy(), x.copy()
                xp[1::2, j] += h
                xm[1::2, j] -= h
                fp, _, sp = evaluate_springs(model, X, xp, nodes, [row])
                fm, _, sm = evaluate_springs(model, X, xm, nodes, [row])
                # Skip springs whose measure crossed a knot between the two samples.
                knots = np.array([p[0] for p in GENERAL["points"]])
                same = np.array([np.searchsorted(knots, a) == np.searchsorted(knots, b) for a, b in zip(sp, sm)])
                fd = (fp - fm) / (2 * h)
                worst = max(worst, np.abs(K[same][:, :, j] - fd[same]).max())
                self.assertGreater(same.sum(), 50)
            self.assertLess(worst / np.abs(K).max(), 1e-7)

    def test_chord_carries_tension_only(self):
        model = nonlinear_spring(3)
        row = params_of(model, CHORD["scale"], CHORD["points"])
        X = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        for length, expected in (
            (6.0, 0.0),
            (10.0, 0.0),
            (13.9, 0.0),
            (14.5, 30.0),
            (16.0, 120.0),
        ):
            x = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]])
            force, K, _ = evaluate_springs(model, X, x, [[0, 1]], [row])
            self.assertAlmostEqual(force[0][0], expected, places=10)
            if expected == 0.0:
                self.assertEqual(np.abs(K).max(), 0.0)

    def test_zero_length_spring_is_finite(self):
        model = nonlinear_spring(3)
        row = params_of(model, CHORD["scale"], CHORD["points"])
        X = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        x = np.zeros((2, 3))
        force, K, _ = evaluate_springs(model, X, x, [[0, 1]], [row])
        self.assertTrue(np.isfinite(force).all() and np.isfinite(K).all())
        self.assertEqual(np.abs(force).max(), 0.0)

    def test_single_precision(self):
        model = nonlinear_spring(len(GENERAL["points"]))
        row = params_of(model, GENERAL["scale"], GENERAL["points"])
        X, x, nodes = random_springs(100, 3)
        f64, K64, _ = evaluate_springs(model, X, x, nodes, [row])
        f32, K32, _ = evaluate_springs(model, X, x, nodes, [row], dtype=wp.float32)
        self.assertLess(np.abs(f32 - f64).max() / np.abs(f64).max(), 1e-4)
        self.assertLess(np.abs(K32 - K64).max() / np.abs(K64).max(), 1e-4)

    def test_uniaxial_update_kernel(self):
        """The same law through the uniaxial kind's batch kernel."""
        model = nonlinear_spring(3)
        pvec = model.param_vector()
        svec = model.state_vector()
        s = np.linspace(-1.0, 8.0, 19) + 0.01
        n = len(s)
        stress = wp.zeros(n, dtype=wp.float64)
        tangent = wp.zeros(n, dtype=wp.float64)
        trial = wp.zeros(n, dtype=svec)
        wp.launch(
            make_update_kernel(model),
            dim=n,
            inputs=[
                wp.array(s, dtype=wp.float64),
                wp.zeros(n, dtype=int),
                wp.array([model.pack(**params_of(model, **CHORD))], dtype=pvec),
                wp.zeros(n, dtype=svec),
            ],
            outputs=[stress, tangent, trial],
        )
        np.testing.assert_allclose(stress.numpy(), 60.0 * np.maximum(s - 4.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(tangent.numpy(), np.where(s >= 4.0, 60.0, 0.0), atol=1e-12)

    def test_gradients_with_respect_to_parameters(self):
        """Tape gradient of the summed force magnitude w.r.t. scale and the points."""
        model = nonlinear_spring(3)
        pvec = model.param_vector()
        vec3 = wp.vec3d
        X, x, nodes = random_springs(80, 11, stretch=(1.1, 1.9))
        kernel = make_spring_kernel(model)

        @wp.kernel(module="unique")
        def total(force: wp.array(dtype=vec3), out: wp.array(dtype=wp.float64)):
            i = wp.tid()
            wp.atomic_add(out, 0, wp.length(force[i]))

        def run(values, grad):
            p = wp.array([pvec(*values)], dtype=pvec, requires_grad=grad)
            n = len(nodes)
            force = wp.zeros(n, dtype=vec3, requires_grad=grad)
            out = wp.zeros(1, dtype=wp.float64, requires_grad=grad)
            tape = wp.Tape()
            with tape:
                wp.launch(
                    kernel,
                    dim=n,
                    inputs=[
                        wp.array(X, dtype=vec3),
                        wp.array(x, dtype=vec3),
                        wp.array(nodes, dtype=int),
                        wp.zeros(n, dtype=int),
                        p,
                    ],
                    outputs=[
                        force,
                        wp.zeros(n, dtype=wp.mat33d),
                        wp.zeros(n, dtype=wp.float64),
                    ],
                )
                wp.launch(total, dim=n, inputs=[force], outputs=[out])
            if grad:
                tape.backward(loss=out)
                return out.numpy()[0], p.grad.numpy()[0].copy()
            return out.numpy()[0], None

        # Slack 2.3 so that springs fall on both sides of it.
        base = np.array([60.0, 0.0, 0.0, 2.3, 0.0, 3.3, 1.0])
        value, grad = run(base, True)
        self.assertGreater(value, 0.0)
        # scale, slack abscissa x1, and the last point (x2, y2) set the active branch.
        for index in (0, 3, 5, 6):
            h = 1e-6 * max(1.0, abs(base[index]))
            up, down = base.copy(), base.copy()
            up[index] += h
            down[index] -= h
            fd = (run(up, False)[0] - run(down, False)[0]) / (2 * h)
            self.assertLess(abs(grad[index] - fd), 1e-5 * max(1.0, abs(fd)), index)


class MitralChordTests(unittest.TestCase):
    """The chord groups of the repository's mitral input."""

    @classmethod
    def setUpClass(cls):
        cls.data = read_mitral_chords()

    def setUp(self):
        if self.data is None:
            self.skipTest("the mitral input is not present")

    def test_every_chord_group_is_covered_by_one_kernel(self):
        _, groups = self.data
        self.assertEqual(len(groups), 8)
        self.assertEqual(sum(len(g["pairs"]) for g in groups), 100)
        signatures = {
            (
                g["type"],
                g["function"],
                g["interpolate"],
                g["measure"],
                g["extend"],
                len(g["points"]),
            )
            for g in groups
        }
        self.assertEqual(
            signatures,
            {("nonlinear spring", "point", "linear", "elongation", "extrapolate", 3)},
        )

    def test_mitral_chords_match_febio_source(self):
        X, groups = self.data
        spring = nonlinear_spring(3, "elongation", "extrapolate")
        rows = [params_of(spring, g["scale"], g["points"]) for g in groups]
        pairs = np.concatenate([g["pairs"] for g in groups])
        region = np.concatenate([np.full(len(g["pairs"]), k) for k, g in enumerate(groups)])
        # Move the leaflet end of every chord away from its papillary end so that
        # elongations span the slack and loaded ranges of every group.
        rng = np.random.default_rng(5)
        x = X.copy()
        d = X[pairs[:, 0]] - X[pairs[:, 1]]
        L0 = np.linalg.norm(d, axis=1)
        x[pairs[:, 0]] = X[pairs[:, 1]] + d * ((L0 + rng.uniform(-2.0, 16.0, len(L0))) / L0)[:, None]
        force, stiffness, s = evaluate_springs(spring, X, x, pairs, rows, region)
        loaded = 0
        for i, (a, b) in enumerate(pairs):
            g = groups[region[i]]
            f, K, _ = febio_closed_forms.nonlinear_spring(
                X[b] - X[a],
                x[b] - x[a],
                g["scale"],
                g["points"],
                g["measure"],
                g["extend"],
            )
            np.testing.assert_allclose(force[i], f, rtol=0, atol=1e-10)
            np.testing.assert_allclose(stiffness[i], K, rtol=0, atol=1e-10)
            loaded += np.abs(f).max() > 0
        self.assertGreater(loaded, 20)
        self.assertLess(loaded, 100)


class LinearTrussTests(unittest.TestCase):
    """FEBio's "linear truss": tau = E ln(lambda), and the linear-truss domain's use of it."""

    E, V_POISSON, AREA = (
        10000.0,
        0.5,
        0.001,
    )  # the values of suite problems dy16 and dy17

    def evaluate(self, X, x, nodes):
        fn = make_truss_function(LINEAR_TRUSS)
        pvec = LINEAR_TRUSS.param_vector()

        @wp.kernel(module="unique")
        def k(
            X: wp.array(dtype=wp.vec3d),
            x: wp.array(dtype=wp.vec3d),
            nodes: wp.array2d(dtype=int),
            area: wp.float64,
            p: wp.array(dtype=pvec),
            force: wp.array(dtype=wp.vec3d),
            stiffness: wp.array(dtype=wp.mat33d),
            tau: wp.array(dtype=wp.float64),
        ):
            i = wp.tid()
            a = nodes[i, 0]
            b = nodes[i, 1]
            f, K, t = fn(X[b] - X[a], x[b] - x[a], area, p[0])
            force[i] = f
            stiffness[i] = K
            tau[i] = t

        n = len(nodes)
        force = wp.zeros(n, dtype=wp.vec3d)
        stiffness = wp.zeros(n, dtype=wp.mat33d)
        tau = wp.zeros(n, dtype=wp.float64)
        wp.launch(
            k,
            dim=n,
            inputs=[
                wp.array(X, dtype=wp.vec3d),
                wp.array(x, dtype=wp.vec3d),
                wp.array(nodes, dtype=int),
                self.AREA,
                wp.array([LINEAR_TRUSS.pack(E=self.E, v=self.V_POISSON)], dtype=pvec),
            ],
            outputs=[force, stiffness, tau],
        )
        return force.numpy(), stiffness.numpy(), tau.numpy()

    def test_matches_febio_source(self):
        X, x, nodes = random_springs(100, 21)
        force, stiffness, tau = self.evaluate(X, x, nodes)
        for i, (a, b) in enumerate(nodes):
            r0, rt = X[b] - X[a], x[b] - x[a]
            L, l = np.linalg.norm(r0), np.linalg.norm(rt)
            n = rt / l
            t = self.E * np.log(l / L)  # FELinearTrussMaterial::Stress
            V = L * self.AREA
            T = t * V / l
            k = V / (l * l) * (self.E - 2 * t)  # FELinearTrussDomain::ElementStiffness
            self.assertAlmostEqual(tau[i], t, delta=1e-9 * self.E)
            np.testing.assert_allclose(force[i], T * n, rtol=0, atol=1e-12 * self.E)
            np.testing.assert_allclose(
                stiffness[i],
                k * np.outer(n, n) + T / l * np.eye(3),
                rtol=0,
                atol=1e-12 * self.E,
            )

    def test_stiffness_is_derivative_of_force(self):
        X, x, nodes = random_springs(60, 22)
        _, K, _ = self.evaluate(X, x, nodes)
        h = 1e-6
        for j in range(3):
            xp, xm = x.copy(), x.copy()
            xp[1::2, j] += h
            xm[1::2, j] -= h
            fd = (self.evaluate(X, xp, nodes)[0] - self.evaluate(X, xm, nodes)[0]) / (2 * h)
            self.assertLess(np.abs(K[:, :, j] - fd).max() / np.abs(K).max(), 1e-7)


if __name__ == "__main__":
    unittest.main()
