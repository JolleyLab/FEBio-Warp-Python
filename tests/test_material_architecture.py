"""Architecture checks for draft v0.3: geometry fields, composition, parameter
schedules, volumetric functions and the uniaxial history contract.

`FIBER_TEST_MODEL` below is test-only. It exists to prove that fiber directions
travel through the geometry vector, independently of any material axes, and that
a mixture hands each child its own slice. It is not a tissue model.
"""

import unittest
from typing import Any

import numpy as np
import warp as wp
from support import DEVICE, setUpModule, tearDownModule  # noqa: F401  (device for the module)

from warp_febio.materials import model as model_api
from warp_febio.materials import models  # noqa: F401
from warp_febio.materials import status as material_status
from warp_febio.materials.compose import solid_mixture
from warp_febio.materials.conformance import check_model, sample_states
from warp_febio.materials.exchange import evaluate_exchange
from warp_febio.materials.geometry import axes_from_vectors, check_axes
from warp_febio.materials.model import GeometryField, Material, from_spatial
from warp_febio.materials.models.nitinol import (
    NITINOL_SUPERELASTIC_UNIAXIAL,
    effective_modulus_nitinol,
    published_tpvr_frame,
)
from warp_febio.materials.schedule import LoadCurve
from warp_febio.materials.uniaxial import integrate, make_update_kernel
from warp_febio.materials.volumetric import PRESSURE_MODELS, make_volumetric


# Test-only fiber model:  W = (ksi / 4) (I4 - 1)^2,  I4 = a0 . C a0.
#   sigma = (2 / J) W'(I4) a (x) a,           a = F a0
#   c : d = (4 / J) W''(I4) (a . d a) a (x) a
@wp.func
def fiber_energy(F: Any, g: Any, p: Any):
    a = F * wp.vector(g[0], g[1], g[2])
    s = wp.dot(a, a) - F.dtype(1.0)
    return F.dtype(0.25) * p[0] * s * s


@wp.func
def fiber_cauchy(F: Any, g: Any, p: Any):
    a = F * wp.vector(g[0], g[1], g[2])
    s = wp.dot(a, a) - F.dtype(1.0)
    return (p[0] * s / wp.determinant(F)) * wp.outer(a, a)


@wp.func
def fiber_tangent_action(F: Any, d: Any, g: Any, p: Any):
    a = F * wp.vector(g[0], g[1], g[2])
    return (F.dtype(2.0) * p[0] * wp.dot(a, d * a) / wp.determinant(F)) * wp.outer(a, a)


FIBER_TEST_MODEL = from_spatial(
    "test_fiber",
    "",
    ("ksi",),
    fiber_energy,
    fiber_cauchy,
    fiber_tangent_action,
    geometry=(GeometryField("fiber", 3),),
)

LIMITS = {
    "rest_stress": 1e-12,
    "cauchy_asymmetry": 1e-12,
    "objectivity_stress": 1e-12,
    "stress_vs_energy_fd": 1e-7,
    "tangent_vs_stress_fd": 1e-7,
    "pk1_vs_cauchy": 1e-12,
    "tangent_asymmetry": 1e-12,
    "parameter_gradient_vs_fd": 1e-6,
}

NITINOL = dict(
    E=50000.0, eps_L=0.06, s_AM_s=500.0, s_AM_f=600.0, s_MA_s=300.0, s_MA_f=200.0
)  # illustrative values for testing the algorithm; not a characterized alloy


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class GeometryAndCompositionTests(unittest.TestCase):
    def test_fiber_model_passes_conformance(self):
        metrics = check_model(FIBER_TEST_MODEL, {"ksi": 800.0}, geometry={"fiber": unit([1.0, 2.0, 0.5])})
        for name, limit in LIMITS.items():
            self.assertLess(metrics[name], limit, name)

    def test_mixture_with_two_independent_fibers(self):
        neo = model_api.get("neo_hookean")
        mix = solid_mixture(
            "test_ground_two_fibers",
            [
                ("ground", neo),
                ("fiber1", FIBER_TEST_MODEL),
                ("fiber2", FIBER_TEST_MODEL),
            ],
        )
        self.assertEqual(mix.param_names, ("ground.E", "ground.v", "fiber1.ksi", "fiber2.ksi"))
        self.assertEqual(
            [(f.name, f.size) for f in mix.geometry],
            [("fiber1.fiber", 3), ("fiber2.fiber", 3)],
        )
        a1, a2 = unit([1.0, 0.3, 0.0]), unit([0.2, 1.0, 0.4])
        params = {
            "ground.E": 1000.0,
            "ground.v": 0.3,
            "fiber1.ksi": 800.0,
            "fiber2.ksi": 300.0,
        }
        geometry = {"fiber1.fiber": a1, "fiber2.fiber": a2}
        F = sample_states()
        H = 0.2 * np.random.default_rng(3).normal(size=F.shape)
        whole = model_api.evaluate(mix, F, params, H, geometry=geometry)
        parts = [
            model_api.evaluate(neo, F, {"E": 1000.0, "v": 0.3}, H),
            model_api.evaluate(FIBER_TEST_MODEL, F, {"ksi": 800.0}, H, geometry={"fiber": a1}),
            model_api.evaluate(FIBER_TEST_MODEL, F, {"ksi": 300.0}, H, geometry={"fiber": a2}),
        ]
        for key in ("energy", "cauchy", "spatial_tangent_action"):
            total = sum(part[key] for part in parts)
            self.assertLess(np.abs(whole[key] - total).max() / np.abs(total).max(), 1e-13)
        # The composed model is a model: it passes the same conformance checks,
        # including gradients with respect to every child's parameters.
        metrics = check_model(mix, params, geometry=geometry)
        for name, limit in LIMITS.items():
            self.assertLess(metrics[name], limit, name)

    def test_mixtures_nest_and_compile_in_single_precision(self):
        neo = model_api.get("neo_hookean")
        inner = solid_mixture("test_inner", [("a", neo), ("b", model_api.get("isotropic_elastic"))])
        outer = solid_mixture("test_outer", [("in", inner), ("f", FIBER_TEST_MODEL)])
        self.assertEqual(outer.param_names, ("in.a.E", "in.a.v", "in.b.E", "in.b.v", "f.ksi"))
        params = {
            "in.a.E": 1000.0,
            "in.a.v": 0.3,
            "in.b.E": 500.0,
            "in.b.v": 0.2,
            "f.ksi": 400.0,
        }
        F = sample_states()
        g = {"f.fiber": unit([0.0, 1.0, 1.0])}
        double = model_api.evaluate(outer, F, params, geometry=g)
        single = model_api.evaluate(outer, F, params, geometry=g, dtype=wp.float32)
        self.assertEqual(single["cauchy"].dtype, np.float32)
        self.assertLess(
            np.abs(single["cauchy"] - double["cauchy"]).max() / np.abs(double["cauchy"]).max(),
            1e-4,
        )

    def test_missing_fiber_field_is_an_error(self):
        with self.assertRaises(ValueError):
            FIBER_TEST_MODEL.pack_geometry(4)
        with self.assertRaises(ValueError):
            model_api.get("neo_hookean").pack_geometry(4, fiber=[1.0, 0.0, 0.0])


class DifferentiabilityTests(unittest.TestCase):
    """Gradients from the Warp tape against central differences, for the two
    inputs the conformance checks do not cover: geometry and history."""

    def test_gradient_with_respect_to_fiber_direction(self):
        mix = solid_mixture(
            "test_gradient_mix",
            [("ground", model_api.get("neo_hookean")), ("f", FIBER_TEST_MODEL)],
        )
        params = {"ground.E": 1000.0, "ground.v": 0.3, "f.ksi": 800.0}
        F = sample_states()[:20]
        n = len(F)
        direction = unit([0.6, 0.7, 0.3])

        def total_energy(a0, tape):
            g = wp.array(
                mix.pack_geometry(n, **{"f.fiber": a0}),
                dtype=mix.geometry_vector(),
                requires_grad=tape,
            )
            W = wp.zeros(n, dtype=wp.float64, requires_grad=tape)
            inputs = [
                wp.array(F, dtype=wp.mat33d),
                wp.zeros(n, dtype=wp.mat33d),
                g,
                wp.zeros(n, dtype=int),
                wp.array([mix.pack(**params)], dtype=mix.param_vector()),
            ]
            outputs = [W] + [wp.zeros(n, dtype=wp.mat33d) for _ in range(4)]
            outputs.append(wp.zeros(n, dtype=int))  # status
            kernel = model_api.make_point_kernel(mix)
            if not tape:
                wp.launch(kernel, dim=n, inputs=inputs, outputs=outputs)
                return W.numpy().sum()
            recording = wp.Tape()
            with recording:
                wp.launch(kernel, dim=n, inputs=inputs, outputs=outputs)
            W.grad.fill_(1.0)
            recording.backward()
            return g.grad.numpy().sum(axis=0)

        from_tape = total_energy(direction, True)
        h = 1e-6
        numeric = np.array(
            [
                (total_energy(direction + h * e, False) - total_energy(direction - h * e, False)) / (2 * h)
                for e in np.eye(3)
            ]
        )
        np.testing.assert_allclose(from_tape, numeric, rtol=1e-8)

    def test_gradient_through_superelastic_history(self):
        model = NITINOL_SUPERELASTIC_UNIAXIAL
        kernel = make_update_kernel(model)
        # Load into the forward plateau, then unload partway down the reverse one,
        # so the final stress depends on the history.
        path = np.concatenate([np.linspace(0, 0.05, 30), np.linspace(0.05, 0.02, 20)[1:]])

        def final_stress(values, tape):
            p = wp.array([model.pack(**values)], dtype=model.param_vector(), requires_grad=tape)
            state = wp.zeros(1, dtype=model.state_vector(), requires_grad=tape)
            region = wp.zeros(1, dtype=int)
            stress = None
            for strain in path:
                stress = wp.zeros(1, dtype=wp.float64, requires_grad=tape)
                tangent = wp.zeros(1, dtype=wp.float64, requires_grad=tape)
                trial = wp.zeros(1, dtype=model.state_vector(), requires_grad=tape)
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array([strain], dtype=wp.float64), region, p, state],
                    outputs=[stress, tangent, trial],
                )
                state = trial  # commit by handing the new array to the next step
            return stress, p

        recording = wp.Tape()
        with recording:
            stress, p = final_stress(NITINOL, True)
        stress.grad.fill_(1.0)
        recording.backward()
        from_tape = p.grad.numpy()[0]
        for index, name in enumerate(model.param_names):
            h = 1e-6 * NITINOL[name]
            up = final_stress({**NITINOL, name: NITINOL[name] + h}, False)[0].numpy()[0]
            down = final_stress({**NITINOL, name: NITINOL[name] - h}, False)[0].numpy()[0]
            numeric = (up - down) / (2 * h)
            self.assertAlmostEqual(from_tape[index], numeric, delta=1e-5 * abs(numeric) + 1e-12, msg=name)


class CoreRobustnessTests(unittest.TestCase):
    def test_models_sharing_a_name_do_not_share_a_kernel(self):
        neo = model_api.get("neo_hookean")
        first = solid_mixture("test_same_name", [("a", neo), ("b", FIBER_TEST_MODEL)])
        second = solid_mixture("test_same_name", [("a", neo), ("b", model_api.get("isotropic_elastic"))])
        F = sample_states()[:8]
        one = model_api.evaluate(
            first,
            F,
            {"a.E": 1000.0, "a.v": 0.3, "b.ksi": 500.0},
            geometry={"b.fiber": [1.0, 0.0, 0.0]},
        )
        two = model_api.evaluate(second, F, {"a.E": 1000.0, "a.v": 0.3, "b.E": 500.0, "b.v": 0.2})
        self.assertGreater(np.abs(one["cauchy"] - two["cauchy"]).max(), 1.0)

    def test_one_pass_tangent_survives_composition(self):
        neo, lee = model_api.get("neo_hookean"), model_api.get("lee_sacks_isotropic")
        mix = solid_mixture("test_combined_mix", [("n", neo), ("l", lee)])
        self.assertIsNotNone(mix.stress_and_tangent)
        params = {
            "n.E": 1000.0,
            "n.v": 0.3,
            "l.c0": 200.0,
            "l.c1": 2968.4,
            "l.c2": 0.2661,
            "l.k": 5000.0,
        }
        F = sample_states()
        fast = evaluate_exchange(mix, F, params)
        slow = evaluate_exchange(mix, F, params, combined=False)
        self.assertLess(
            np.abs(fast["tangent"] - slow["tangent"]).max() / np.abs(slow["tangent"]).max(),
            1e-13,
        )
        # A child without a one-pass function leaves the mixture without one.
        partial = solid_mixture("test_partial_mix", [("n", neo), ("f", FIBER_TEST_MODEL)])
        self.assertIsNone(partial.stress_and_tangent)

    def test_axes_from_vectors_follows_febio(self):
        Q = axes_from_vectors([2.0, 0.0, 0.0], [1.0, 1.0, 0.0])
        np.testing.assert_allclose(Q, np.eye(3), atol=1e-15)
        Q = axes_from_vectors([1.0, 1.0, 0.0], [0.0, 0.0, 3.0])
        check_axes(Q)
        np.testing.assert_allclose(Q[:, 0], np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0))
        # Nearly parallel vectors: FEBio replaces d by the x axis (or y).
        check_axes(axes_from_vectors([0.0, 0.0, 1.0], [0.0, 0.0, 2.0]))
        check_axes(axes_from_vectors([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
        with self.assertRaises(ValueError):
            axes_from_vectors([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            check_axes(np.diag([1.0, 1.0, 1.1]))
        with self.assertRaises(ValueError):
            check_axes(np.diag([1.0, 1.0, -1.0]))


class ParameterRangeTests(unittest.TestCase):
    """FEBio rejects out-of-range parameters at initialization. So does pack()."""

    def test_ranges_follow_febio(self):
        neo = model_api.get("neo_hookean")
        neo.pack(E=1.0, v=-1.0)  # FE_RANGE_RIGHT_OPEN(-1, 0.5): -1 is allowed
        for bad in (
            dict(E=0.0, v=0.3),
            dict(E=1.0, v=0.5),
            dict(E=1.0, v=0.6),
            dict(E=float("nan"), v=0.3),
        ):
            with self.assertRaises(ValueError):
                neo.pack(**bad)
        hgo = model_api.get("hgo_unconstrained")
        good = dict(c=200.0, k1=13480.0, k2=1.06, kappa=1.0 / 3.0, gamma=18.85, k=1500.0)
        hgo.pack(**good)  # FE_RANGE_CLOSED(0, 1/3): 1/3 is allowed
        with self.assertRaises(ValueError) as caught:
            hgo.pack(**{**good, "kappa": 0.4})
        self.assertIn("kappa", str(caught.exception))

    def test_ranges_carry_into_mixtures(self):
        mix = solid_mixture(
            "test_range_mix",
            [("ground", model_api.get("neo_hookean")), ("f", FIBER_TEST_MODEL)],
        )
        mix.pack(**{"ground.E": 1.0, "ground.v": 0.3, "f.ksi": 1.0})
        with self.assertRaises(ValueError) as caught:
            mix.pack(**{"ground.E": 1.0, "ground.v": 0.5, "f.ksi": 1.0})
        self.assertIn("ground.v", str(caught.exception))


class FailureStatusTests(unittest.TestCase):
    """The rule of materials/status.py: bad points are flagged and zeroed, good
    points in the same launch are unaffected, and nothing non-finite is written."""

    def setUp(self):
        self.model = model_api.get("lee_sacks_isotropic")
        self.params = dict(c0=200.0, c1=2968.4, c2=0.2661, k=5000.0)
        good = np.diag([1.1, 0.95, 1.0])
        inverted = np.diag([1.0, 1.0, -0.2])  # J < 0
        collapsed = np.diag([1.0, 1.0, 0.0])  # J = 0
        overflow = np.diag([40.0, 40.0, 1.0 / 1600.0])  # exp() overflows in FP64
        self.F = np.array([good, inverted, collapsed, overflow, good])

    def test_point_kernel_reports_and_isolates_failures(self):
        with np.errstate(all="ignore"):
            out = model_api.evaluate(self.model, self.F, self.params)
        self.assertEqual(
            list(out["status"]),
            [
                material_status.OK,
                material_status.NEGATIVE_JACOBIAN,
                material_status.NEGATIVE_JACOBIAN,
                material_status.NON_FINITE,
                material_status.OK,
            ],
        )
        for key in (
            "energy",
            "cauchy",
            "spatial_tangent_action",
            "pk1",
            "tangent_action",
        ):
            self.assertTrue(np.all(np.isfinite(out[key])), key)
            self.assertEqual(np.abs(out[key][1:4]).max(), 0.0, key)
        # The good points match a launch that contains only good points.
        clean = model_api.evaluate(self.model, self.F[[0, 4]], self.params)
        np.testing.assert_array_equal(out["cauchy"][[0, 4]], clean["cauchy"])
        self.assertEqual(material_status.worst(out["status"]), material_status.NEGATIVE_JACOBIAN)
        with self.assertRaises(material_status.MaterialFailure) as caught:
            material_status.raise_on_failure(out["status"])
        self.assertIn("negative Jacobian at 2", str(caught.exception))
        material_status.raise_on_failure(clean["status"])  # no failure, no exception

    def test_exchange_kernel_follows_the_same_rule(self):
        with np.errstate(all="ignore"):
            out = evaluate_exchange(self.model, self.F, self.params)
        self.assertEqual(list(out["status"] != material_status.OK), [False, True, True, True, False])
        self.assertTrue(np.all(np.isfinite(out["tangent"])))
        self.assertEqual(np.abs(out["tangent"][1:4]).max(), 0.0)
        self.assertGreater(np.abs(out["tangent"][0]).max(), 0.0)


class HolzapfelGasserOgdenTests(unittest.TestCase):
    """Behaviour specific to the fiber model, beyond the shared conformance checks."""

    def setUp(self):
        self.model = model_api.get("hgo_unconstrained")
        self.axes = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]])
        self.base = dict(c=200.0, k1=13480.0, k2=1.06, kappa=0.1, gamma=35.0, k=1500.0)

    def evaluate(self, F, params, axes=None):
        Q = self.axes if axes is None else axes
        return model_api.evaluate(self.model, F, params, geometry={"mat_axis": Q.reshape(9)})

    def test_parameter_sets_from_the_literature_and_the_febio_suite(self):
        small = np.eye(3) + 0.02 * np.random.default_rng(5).normal(size=(40, 3, 3))
        cases = {
            # Coupled model at lower dispersion, so the fibers act.
            "lower dispersion": (self.base, None),
            # Values of FEBio test suite problem mp10 ("HGO unconstrained").
            # k2 = 524.6 makes the response explode beyond small strain, so
            # this set is checked on small deformations.
            "febio suite mp10": (
                dict(c=0.00764, k1=0.9966, k2=524.6, kappa=0.226, gamma=49.98, k=0.764),
                small,
            ),
        }
        import febio_closed_forms

        limits = {
            "cauchy_vs_febio": 1e-12,
            "tangent_vs_febio": 1e-12,
            "stress_vs_energy_fd": 1e-6,
            "tangent_vs_stress_fd": 1e-6,
            "objectivity_stress": 1e-12,
            "parameter_gradient_vs_fd": 1e-5,
            "geometry_gradient_vs_fd": 1e-5,
        }
        for label, (params, states) in cases.items():
            with self.subTest(case=label):
                metrics = check_model(
                    self.model,
                    params,
                    lambda F, **p: febio_closed_forms.hgo_unconstrained(F, self.axes, **p),
                    geometry={"mat_axis": self.axes.reshape(9)},
                    states=states,
                )
                for name, limit in limits.items():
                    self.assertLess(metrics[name], limit, f"{label}: {name}")

    def test_fibers_carry_no_compression(self):
        # One family along the first axis (gamma = 0), no dispersion. Shortening
        # along that axis leaves only the ground matrix.
        axes = np.eye(3)
        fibers = {**self.base, "kappa": 0.0, "gamma": 0.0}
        ground_only = {**fibers, "k1": 0.0}
        F = np.array([np.diag([0.9, 1.05, 1.05])])
        np.testing.assert_allclose(
            self.evaluate(F, fibers, axes)["cauchy"],
            self.evaluate(F, ground_only, axes)["cauchy"],
            atol=1e-12,
        )
        stretched = np.array([np.diag([1.1, 0.97, 0.97])])
        self.assertGreater(
            self.evaluate(stretched, fibers, axes)["cauchy"][0, 0, 0],
            10.0 * self.evaluate(stretched, ground_only, axes)["cauchy"][0, 0, 0],
        )

    def test_direction_dependence_and_its_limits(self):
        axes = np.eye(3)
        params = {**self.base, "kappa": 0.0, "gamma": 0.0}
        along = self.evaluate(np.array([np.diag([1.1, 1.0, 1.0])]), params, axes)["cauchy"][0, 0, 0]
        across = self.evaluate(np.array([np.diag([1.0, 1.0, 1.1])]), params, axes)["cauchy"][0, 2, 2]
        self.assertGreater(along, 5.0 * across)  # stiffer along the fibers
        F = sample_states()[:12]
        # The two families are symmetric about the first axis: gamma and -gamma agree.
        np.testing.assert_allclose(
            self.evaluate(F, {**self.base, "gamma": 35.0})["cauchy"],
            self.evaluate(F, {**self.base, "gamma": -35.0})["cauchy"],
            rtol=1e-12,
            atol=1e-9,
        )
        # At kappa = 1/3 the model is isotropic: the axes do not matter.
        isotropic = {**self.base, "kappa": 1.0 / 3.0}
        np.testing.assert_allclose(
            self.evaluate(F, isotropic, self.axes)["cauchy"],
            self.evaluate(F, isotropic, np.eye(3))["cauchy"],
            rtol=1e-10,
            atol=1e-7,
        )
        # At the published kappa = 0.33 the model is close to isotropic but not
        # isotropic: over these states the axes change the stress by a few
        # percent (about 4.5 percent at most), against roughly 90 percent at
        # kappa = 0.1.
        published = {**self.base, "kappa": 0.33, "gamma": 18.85}

        def axis_effect(params):
            a = self.evaluate(F, params, self.axes)["cauchy"]
            b = self.evaluate(F, params, np.eye(3))["cauchy"]
            return np.abs(a - b).max() / np.abs(a).max()

        self.assertLess(axis_effect(published), 0.06)
        self.assertGreater(axis_effect(published), 0.01)
        self.assertGreater(axis_effect(self.base), 3.0 * axis_effect(published))

    def test_energy_and_its_k2_derivative_at_zero_exponent(self):
        F = sample_states()[:20]
        at_zero = {**self.base, "k2": 0.0}
        tiny = {**self.base, "k2": 1.0e-9}
        e0 = self.evaluate(F, at_zero)["energy"]
        self.assertTrue(np.all(np.isfinite(e0)))
        np.testing.assert_allclose(e0, self.evaluate(F, tiny)["energy"], rtol=1e-7)
        # d(energy)/d(k2) at k2 = 0 from the tape against a one-sided difference.
        n = len(F)
        p = wp.array(
            [self.model.pack(**at_zero)],
            dtype=self.model.param_vector(),
            requires_grad=True,
        )
        W = wp.zeros(n, dtype=wp.float64, requires_grad=True)
        outputs = [W] + [wp.zeros(n, dtype=wp.mat33d) for _ in range(4)] + [wp.zeros(n, dtype=int)]
        recording = wp.Tape()
        with recording:
            wp.launch(
                model_api.make_point_kernel(self.model),
                dim=n,
                inputs=[
                    wp.array(F, dtype=wp.mat33d),
                    wp.zeros(n, dtype=wp.mat33d),
                    wp.array(
                        self.model.pack_geometry(n, mat_axis=self.axes.reshape(9)),
                        dtype=self.model.geometry_vector(),
                    ),
                    wp.zeros(n, dtype=int),
                    p,
                ],
                outputs=outputs,
            )
        W.grad.fill_(1.0)
        recording.backward()
        h = 1.0e-6
        one_sided = (self.evaluate(F, {**self.base, "k2": h})["energy"].sum() - e0.sum()) / h
        self.assertAlmostEqual(p.grad.numpy()[0][2], one_sided, delta=1e-4 * abs(one_sided))


class VolumetricTests(unittest.TestCase):
    def test_derivatives_and_rest_state(self):
        J = np.linspace(0.6, 1.6, 41)
        k = 1500.0
        h = 1e-5
        for name in PRESSURE_MODELS:
            U, dU, d2U = make_volumetric(name)

            @wp.kernel
            def evaluate(
                J: wp.array(dtype=wp.float64),
                k: wp.float64,
                out: wp.array(dtype=wp.vec3d),
            ):
                i = wp.tid()
                out[i] = wp.vec3d(U(J[i], k), dU(J[i], k), d2U(J[i], k))

            def run(values):
                out = wp.zeros(len(values), dtype=wp.vec3d)
                wp.launch(
                    evaluate,
                    dim=len(values),
                    inputs=[wp.array(values, dtype=wp.float64), k],
                    outputs=[out],
                )
                return out.numpy()

            mid, up, down, rest = run(J), run(J + h), run(J - h), run(np.array([1.0]))
            with self.subTest(pressure_model=name):
                np.testing.assert_allclose(mid[:, 1], (up[:, 0] - down[:, 0]) / (2 * h), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(mid[:, 2], (up[:, 1] - down[:, 1]) / (2 * h), rtol=1e-6, atol=1e-6)
                # No energy and no pressure at J = 1; stiffness there is the bulk modulus.
                np.testing.assert_allclose(rest[0], [0.0, 0.0, k], atol=1e-9)


class UncoupledModelTests(unittest.TestCase):
    def test_uncoupled_model_is_not_accepted_by_solid_mixture(self):
        """Deviatoric-only models must not be summed like coupled ones."""
        from warp_febio.materials.compose import solid_mixture

        uncoupled = model_api.get("hgo_uncoupled")
        self.assertEqual(uncoupled.formulation, "uncoupled")
        self.assertEqual(dict(uncoupled.options)["pressure_model"], "Abaqus (GOH)")
        with self.assertRaises(ValueError):
            solid_mixture("bad", [("a", model_api.get("neo_hookean")), ("b", uncoupled)])


class ScheduleTests(unittest.TestCase):
    def test_load_curves(self):
        ramp = LoadCurve((0.0, 2.0), (1.0, 3.0))
        self.assertAlmostEqual(ramp(1.0), 2.0)
        self.assertAlmostEqual(ramp(5.0), 3.0)
        step = LoadCurve((0.0, 1.0), (1.0, 0.25), interpolate="step")
        # FEBio's STEP rule: between two points the later point's value applies.
        self.assertEqual([step(t) for t in (0.0, 0.99, 1.0, 2.0)], [1.0, 0.25, 0.25, 0.25])
        # Values from FEBio 4.13 runs of STEP curves through these points, at
        # output times that fall on the points and between them.
        two = LoadCurve((0.35, 0.65), (1.0, 0.75), interpolate="step")
        self.assertEqual(
            [two(t) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)],
            [1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.75, 0.75],
        )
        three = LoadCurve((0.25, 0.5, 0.75), (1.0, 0.6, 0.3), interpolate="step")
        self.assertEqual(
            [three(t) for t in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)],
            [1.0, 1.0, 0.6, 0.3, 0.3, 0.3, 0.3, 0.3],
        )
        with self.assertRaises(ValueError):
            LoadCurve((0.0, 0.0), (1.0, 2.0))

    def test_level_1_nitinol_is_an_elastic_law_with_a_scheduled_modulus(self):
        for law in ("neo-Hookean", "isotropic elastic"):
            frame = effective_modulus_nitinol(
                E=4.0e7,
                v=0.33,
                density=6.45e-6,
                release_time=1.0,
                release_factor=0.5,
                law=law,
            )
            self.assertIsInstance(frame, Material)
            self.assertEqual(frame.model.febio_type, law)
            self.assertEqual(frame.density, 6.45e-6)
            self.assertEqual(frame.params_at(0.5)["E"], 4.0e7)
            self.assertEqual(frame.params_at(1.5)["E"], 2.0e7)
            self.assertEqual(frame.params_at(1.5)["v"], 0.33)
            F = sample_states()[:6]
            before = model_api.evaluate(frame.model, F, frame.params_at(0.5))
            after = model_api.evaluate(frame.model, F, frame.params_at(1.5))
            # Both laws are linear in E at fixed Poisson ratio.
            np.testing.assert_allclose(after["cauchy"], 0.5 * before["cauchy"], rtol=1e-12, atol=1e-6)

    def test_published_tpvr_frame(self):
        """The frame material and load curve of the six open TPVR inputs."""
        frame = published_tpvr_frame()
        self.assertEqual(frame.model.febio_type, "isotropic elastic")
        self.assertEqual(frame.params_at(0.5)["E"], 4.0e7)
        self.assertEqual(frame.params_at(0.998063)["E"], 4.0e7)
        self.assertEqual(frame.params_at(0.999)["E"], 3.0e7)
        self.assertEqual(frame.params_at(3.0)["E"], 3.0e7)


class SuperelasticUniaxialTests(unittest.TestCase):
    """Algorithmic checks of the draft law. They show the update is consistent
    with its own equations. They do not show agreement with the published model
    or with test data."""

    def cycle(self, steps, top=0.09):
        return np.concatenate([np.linspace(0.0, top, steps), np.linspace(top, 0.0, steps)[1:]])

    def test_full_cycle_closes_with_expected_plateaus(self):
        steps = 4001
        path = self.cycle(steps)
        stress, tangent, state = integrate(NITINOL_SUPERELASTIC_UNIAXIAL, path, NITINOL)
        stress, xi = stress[:, 0], state[:, 0, 0]
        E, eps_L = NITINOL["E"], NITINOL["eps_L"]
        loading = slice(0, steps)
        start = path[loading][np.argmax(xi[loading] > 1e-12)]
        finish = path[loading][np.argmax(xi[loading] >= 1.0 - 1e-12)]
        self.assertAlmostEqual(start, NITINOL["s_AM_s"] / E, delta=2 * 0.09 / steps)
        self.assertAlmostEqual(finish, NITINOL["s_AM_f"] / E + eps_L, delta=2 * 0.09 / steps)
        self.assertEqual(xi.max(), 1.0)
        self.assertAlmostEqual(stress[-1], 0.0, places=9)  # full strain recovery
        self.assertAlmostEqual(xi[-1], 0.0, places=12)
        # Dissipated energy per cycle is the area between the plateaus:
        # eps_L times the difference of their mean stresses.
        expected = eps_L * (
            0.5 * (NITINOL["s_AM_s"] + NITINOL["s_AM_f"]) - 0.5 * (NITINOL["s_MA_s"] + NITINOL["s_MA_f"])
        )
        self.assertAlmostEqual(np.trapezoid(stress, path), expected, delta=1e-3 * expected)

    def test_update_does_not_depend_on_step_size(self):
        target = 0.05  # mid-plateau
        coarse, _, state_coarse = integrate(NITINOL_SUPERELASTIC_UNIAXIAL, np.array([0.0, target]), NITINOL)
        fine, _, state_fine = integrate(NITINOL_SUPERELASTIC_UNIAXIAL, np.linspace(0.0, target, 1000), NITINOL)
        self.assertAlmostEqual(coarse[-1, 0], fine[-1, 0], delta=1e-9 * NITINOL["s_AM_f"])
        self.assertAlmostEqual(state_coarse[-1, 0, 0], state_fine[-1, 0, 0], places=12)

    def test_tangent_is_consistent(self):
        path = self.cycle(4001)
        stress, tangent, _ = integrate(NITINOL_SUPERELASTIC_UNIAXIAL, path, NITINOL)
        stress, tangent = stress[:, 0], tangent[:, 0]
        numeric = np.gradient(stress, path)
        # Skip the points next to a change of branch, where the slope jumps.
        jump = np.abs(np.diff(tangent, prepend=tangent[0])) > 1.0
        near = jump | np.roll(jump, 1) | np.roll(jump, -1)
        near[[0, 4000, -1]] = True
        self.assertLess(np.abs((tangent - numeric)[~near]).max() / NITINOL["E"], 1e-9)

    def test_compression_mirrors_tension_and_inner_loops_recover(self):
        path = self.cycle(801)
        tension, _, _ = integrate(NITINOL_SUPERELASTIC_UNIAXIAL, path, NITINOL)
        compression, _, _ = integrate(NITINOL_SUPERELASTIC_UNIAXIAL, -path, NITINOL)
        np.testing.assert_allclose(compression, -tension, atol=1e-9)
        inner = np.concatenate(
            [
                np.linspace(0, 0.045, 400),
                np.linspace(0.045, 0.03, 200),
                np.linspace(0.03, 0.09, 600),
                np.linspace(0.09, 0, 800),
            ]
        )
        stress, _, state = integrate(NITINOL_SUPERELASTIC_UNIAXIAL, inner, NITINOL)
        xi = state[:, 0, 0]
        self.assertTrue(0.0 < xi[599] < xi[399])  # partial reverse transformation
        self.assertTrue(np.all((xi >= 0.0) & (xi <= 1.0)))
        self.assertAlmostEqual(stress[-1, 0], 0.0, places=9)
        self.assertAlmostEqual(xi[-1], 0.0, places=12)

    def test_many_fibers_with_different_histories_in_one_launch(self):
        # A beam cross-section in bending: fibers at different heights see
        # different strains, tension on one side and compression on the other.
        heights = np.linspace(-1.0, 1.0, 9)
        curvature = np.concatenate([np.linspace(0, 0.08, 200), np.linspace(0.08, 0, 200)])
        path = curvature[:, None] * heights[None, :]
        stress, _, state = integrate(NITINOL_SUPERELASTIC_UNIAXIAL, path, NITINOL)
        np.testing.assert_allclose(stress[:, ::-1], -stress, atol=1e-9)  # antisymmetric
        self.assertEqual(state[199, 4, 0], 0.0)  # neutral axis never transforms
        self.assertGreater(state[199, -1, 0], state[199, -3, 0])  # outer fiber transforms most
        np.testing.assert_allclose(stress[-1], 0.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
