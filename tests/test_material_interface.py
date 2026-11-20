"""Conformance of every registered material model, and agreement with the
implementations that predate the common interface.

These are material-point checks. They establish that the equations match the
FEBio source; they do not establish agreement of a full FEBio solve.
"""

import unittest

import numpy as np
import warp as wp

from warp_febio.materials import model as model_api
from warp_febio.materials import models  # noqa: F401  (registers the models)
from warp_febio.materials import neo_hookean as legacy_nh
from warp_febio.materials import stvk as legacy_stvk
from warp_febio.materials.conformance import check_model, sample_states
from warp_febio.materials.lee_sacks import evaluate as legacy_lee_sacks

# Parameter sets. Lee-Sacks: adult mitral values (kPa) from the valve deck in
# references/febio/originals/cardiofem/valve_pressurization. The others match
# the existing unit tests.
PARAMETERS = {
    "lee_sacks_isotropic": dict(c0=200.0, c1=2968.4, c2=0.2661, k=5000.0),
    "neo_hookean": dict(E=1.0e6, v=0.45),
    "isotropic_elastic": dict(E=1.0e6, v=0.3),
}

# Loosest acceptable value for each metric returned by check_model.
LIMITS = {
    "rest_energy": 1e-12,
    "rest_stress": 1e-12,
    "objectivity_energy": 1e-12,
    "objectivity_stress": 1e-12,
    "cauchy_asymmetry": 1e-12,
    "pk1_vs_cauchy": 1e-12,
    "pk1_tangent_vs_fd": 1e-7,
    "stress_vs_energy_fd": 1e-7,
    "tangent_vs_stress_fd": 1e-7,
    "fp32_vs_fp64_stress": 1e-4,
    "tangent_asymmetry": 1e-12,
    "cauchy_vs_febio": 1e-12,
    "tangent_vs_febio": 1e-12,
    "parameter_gradient_vs_fd": 1e-6,
}


@wp.kernel
def legacy_lame_response(
    which: int,
    F: wp.array(dtype=wp.mat33d),
    H: wp.array(dtype=wp.mat33d),
    lame: wp.float64,
    shear: wp.float64,
    W: wp.array(dtype=wp.float64),
    P: wp.array(dtype=wp.mat33d),
    dP: wp.array(dtype=wp.mat33d),
):
    i = wp.tid()
    if which == 0:
        W[i] = legacy_nh.febio_energy(F[i], lame, shear)
        P[i] = legacy_nh.febio_pk1(F[i], lame, shear)
        dP[i] = legacy_nh.febio_tangent_action(F[i], H[i], lame, shear)
    else:
        W[i] = legacy_stvk.energy(F[i], lame, shear)
        P[i] = legacy_stvk.pk1(F[i], lame, shear)
        dP[i] = legacy_stvk.tangent_action(F[i], H[i], lame, shear)


def legacy(name, F, H, params):
    """The pre-interface implementation of `name`, evaluated on the same states."""
    if name == "lee_sacks_isotropic":
        out = legacy_lee_sacks(F, params, H)
        return out["energy"], out["pk1"], out["tangent_action"]
    lame, shear = legacy_nh.lame_from_young_poisson(params["E"], params["v"])
    n = len(F)
    W = wp.zeros(n, dtype=wp.float64)
    P = wp.zeros(n, dtype=wp.mat33d)
    dP = wp.zeros(n, dtype=wp.mat33d)
    wp.launch(
        legacy_lame_response,
        dim=n,
        inputs=[
            0 if name == "neo_hookean" else 1,
            wp.array(F, dtype=wp.mat33d),
            wp.array(H, dtype=wp.mat33d),
            lame,
            shear,
        ],
        outputs=[W, P, dP],
    )
    return W.numpy(), P.numpy(), dP.numpy()


class MaterialInterfaceTests(unittest.TestCase):
    def test_every_model_has_parameters_for_the_test(self):
        self.assertEqual({m.name for m in model_api.all_models()}, set(PARAMETERS))

    def test_conformance(self):
        for model in model_api.all_models():
            with self.subTest(model=model.name):
                metrics = check_model(model, PARAMETERS[model.name])
                for name, limit in LIMITS.items():
                    self.assertLess(metrics[name], limit, f"{model.name}: {name}")

    def test_matches_previous_implementation(self):
        F = sample_states()
        H = 0.2 * np.random.default_rng(11).normal(size=F.shape)
        for model in model_api.all_models():
            with self.subTest(model=model.name):
                params = PARAMETERS[model.name]
                new = model_api.evaluate(model, F, params, H)
                W, P, dP = legacy(model.name, F, H, params)
                for value, reference in (
                    (new["energy"], W),
                    (new["pk1"], P),
                    (new["tangent_action"], dP),
                ):
                    self.assertLess(
                        np.abs(value - reference).max() / np.abs(reference).max(), 1e-12
                    )

    def test_lookup_by_febio_type(self):
        self.assertEqual(model_api.by_febio_type("isotropic Lee-Sacks").name, "lee_sacks_isotropic")
        self.assertEqual(model_api.by_febio_type("neo-Hookean").name, "neo_hookean")
        self.assertEqual(model_api.by_febio_type("isotropic elastic").name, "isotropic_elastic")
        with self.assertRaises(KeyError):
            model_api.by_febio_type("not a material")

    def test_parameter_names_are_checked(self):
        model = model_api.get("neo_hookean")
        with self.assertRaises(ValueError):
            model.pack(E=1.0)
        with self.assertRaises(ValueError):
            model.pack(E=1.0, v=0.3, k=2.0)

    def test_cuda_matches_cpu_when_available(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA device unavailable; GPU execution is unverified")
        F = sample_states()
        for model in model_api.all_models():
            cpu = model_api.evaluate(model, F, PARAMETERS[model.name])
            gpu = model_api.evaluate(model, F, PARAMETERS[model.name], device=devices[0])
            np.testing.assert_allclose(gpu["cauchy"], cpu["cauchy"], rtol=1e-11, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
