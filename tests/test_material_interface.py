"""Conformance of every registered material model, and agreement with the
implementations that predate the common interface.

These are material-point checks. They establish that the equations match the
FEBio source; they do not establish agreement of a full FEBio solve.
"""

import json
import pathlib
import unittest

import numpy as np
import warp as wp

import febio_closed_forms
from warp_febio.materials import model as model_api
from warp_febio.materials import models  # noqa: F401  (registers the models)
from warp_febio.materials.conformance import check_model, sample_states

# The pilot implementations are an independent comparison. The input and output
# draft (pull request 5) moves them from warp_febio.materials to the
# warp_febio_pilot package under pilots/; support both locations.
try:
    from warp_febio.materials import neo_hookean as legacy_nh
    from warp_febio.materials import stvk as legacy_stvk
    from warp_febio.materials.lee_sacks import evaluate as legacy_lee_sacks
except ImportError:
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilots"))
    from warp_febio_pilot.materials import neo_hookean as legacy_nh
    from warp_febio_pilot.materials import stvk as legacy_stvk
    from warp_febio_pilot.materials.lee_sacks import evaluate as legacy_lee_sacks

# Parameter sets. Lee-Sacks: adult mitral values (kPa) from the valve deck in
# references/febio/originals/cardiofem/valve_pressurization. The others match
# the existing unit tests.
PARAMETERS = {
    "lee_sacks_isotropic": dict(c0=200.0, c1=2968.4, c2=0.2661, k=5000.0),
    "neo_hookean": dict(E=1.0e6, v=0.45),
    "isotropic_elastic": dict(E=1.0e6, v=0.3),
    # Right ventricular outflow tract values of Zelonis et al., Table 1 (kPa),
    # used here with the coupled form of the model.
    "hgo_unconstrained": dict(
        c=200.0, k1=13480.0, k2=1.06, kappa=0.33, gamma=18.85, k=1500.0
    ),
    # The same values with the uncoupled form the paper names. Only the
    # deviatoric part is checked here; k does not enter it.
    "hgo_uncoupled": dict(
        c=200.0, k1=13480.0, k2=1.06, kappa=0.33, gamma=18.85, k=1500.0
    ),
}

# Material axes for models that need them: an arbitrary rotation, so that no
# test passes only because the axes line up with the global ones.
AXES = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]])
GEOMETRY = {
    "hgo_unconstrained": {"mat_axis": AXES.reshape(9)},
    "hgo_uncoupled": {"mat_axis": AXES.reshape(9)},
}

# Models compared with FEBio through stored FEBio runs
# (tests/test_febio_agreement_material_point.py) and not through a formula
# transcribed from the FEBio source.
CHECKED_AGAINST_FEBIO_RUNS = ("hgo_uncoupled",)

# Models that have a pilot implementation to compare against.
PILOT = ("lee_sacks_isotropic", "neo_hookean", "isotropic_elastic")


def closed_form_for(name):
    """FEBio closed form for a model, with its geometry bound, or None."""
    if name == "hgo_unconstrained":
        return lambda F, **params: febio_closed_forms.hgo_unconstrained(
            F, AXES, **params
        )
    return febio_closed_forms.CLOSED_FORMS.get(name)


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
    "geometry_gradient_vs_fd": 1e-6,
    "combined_vs_actions": 1e-12,
    "stress_parameter_gradient_vs_fd": 1e-6,
    "stress_deformation_gradient_vs_fd": 1e-6,
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
                states = None
                if model.name == "hgo_uncoupled":
                    # Under a pure volume change the deviatoric deformation is a
                    # rotation, so both fiber strains are exactly zero: the
                    # point where the fibers switch on. The stress is continuous
                    # there and the tangent is not, so a finite difference
                    # across it is not a test of the tangent. Those states are
                    # left out for this model.
                    F = sample_states()
                    Ft = F * np.linalg.det(F)[:, None, None] ** (-1.0 / 3.0)
                    C = np.einsum("nki,nkj->nij", Ft, Ft)
                    states = F[np.abs(C - np.eye(3)).max(axis=(1, 2)) > 1e-9]
                metrics = check_model(
                    model,
                    PARAMETERS[model.name],
                    closed_form_for(model.name),
                    geometry=GEOMETRY.get(model.name),
                    states=states,
                )
                for name, limit in LIMITS.items():
                    if name in metrics:
                        self.assertLess(metrics[name], limit, f"{model.name}: {name}")
                # Every model must be checked against FEBio where FEBio has it.
                if model.febio_type and model.name not in CHECKED_AGAINST_FEBIO_RUNS:
                    self.assertIn("cauchy_vs_febio", metrics, model.name)
                    self.assertIn("tangent_vs_febio", metrics, model.name)

    def test_matches_previous_implementation(self):
        F = sample_states()
        H = 0.2 * np.random.default_rng(11).normal(size=F.shape)
        for name in PILOT:
            model = model_api.get(name)
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
        self.assertEqual(
            model_api.by_febio_type("isotropic Lee-Sacks").name, "lee_sacks_isotropic"
        )
        self.assertEqual(model_api.by_febio_type("neo-Hookean").name, "neo_hookean")
        self.assertEqual(
            model_api.by_febio_type("isotropic elastic").name, "isotropic_elastic"
        )
        self.assertEqual(
            model_api.by_febio_type("HGO unconstrained").name, "hgo_unconstrained"
        )
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
            geometry = GEOMETRY.get(model.name)
            cpu = model_api.evaluate(
                model, F, PARAMETERS[model.name], geometry=geometry
            )
            gpu = model_api.evaluate(
                model, F, PARAMETERS[model.name], geometry=geometry, device=devices[0]
            )
            np.testing.assert_allclose(
                gpu["cauchy"], cpu["cauchy"], rtol=1e-11, atol=1e-8
            )


class FebioSuiteParameterTests(unittest.TestCase):
    """Every parameter set the FEBio test suite uses for a material we implement.

    tests/data/febio_suite_materials.json lists them, with the suite commit and
    the problems each comes from (regenerate with
    tests/data/extract_febio_suite_materials.py). The checks here are at the
    material point: FEBio's closed forms for stress and tangent, and internal
    consistency. Running the suite problems themselves needs elements and a
    solver, and is the job of the verification harness.
    """

    LIMITS = {
        "cauchy_vs_febio": 1e-10,
        "tangent_vs_febio": 1e-10,
        "stress_vs_energy_fd": 1e-6,
        "tangent_vs_stress_fd": 1e-6,
        "objectivity_stress": 1e-10,
        "cauchy_asymmetry": 1e-12,
    }

    def test_all_suite_parameter_sets(self):
        record = json.loads(
            (
                pathlib.Path(__file__).parent / "data" / "febio_suite_materials.json"
            ).read_text()
        )
        # Small deformations for sets whose exponent makes large ones meaningless.
        small = np.eye(3) + 0.02 * np.random.default_rng(5).normal(size=(40, 3, 3))
        checked = 0
        for name, info in record["models"].items():
            model = model_api.get(name)
            for entry in info["sets"]:
                params = dict(zip(info["parameters"], entry["values"]))
                label = f"{name} {params} from {entry['problems'][:3]}"
                with self.subTest(case=label):
                    # FEBio accepts these values, so our range checks must too.
                    model.pack(**params)
                    metrics = check_model(
                        model,
                        params,
                        closed_form_for(name),
                        geometry=GEOMETRY.get(name),
                        states=small if name == "hgo_unconstrained" else None,
                    )
                    for metric, limit in self.LIMITS.items():
                        self.assertLess(metrics[metric], limit, f"{label}: {metric}")
                    checked += 1
        # Guard against the data file going missing or empty without notice.
        self.assertGreaterEqual(checked, 70)


if __name__ == "__main__":
    unittest.main()
