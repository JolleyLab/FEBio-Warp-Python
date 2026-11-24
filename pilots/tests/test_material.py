"""Equation verification only: these tests do not establish FEBio agreement."""

import unittest
import numpy as np
import warp as wp

from warp_febio_pilot.materials.lee_sacks import evaluate

PARAMETERS = dict(c0=200.0, c1=2968.4, c2=0.2661, k=5000.0, tangent_scale=10.0)


class MaterialTests(unittest.TestCase):
    def setUp(self):
        self.f = np.array(
            [
                [[1.12, 0.08, 0.0], [0.02, 0.96, 0.04], [0.0, 0.01, 0.97]],
                [[0.95, 0.0, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 0.95]],
            ]
        )
        self.h = np.random.default_rng(31).normal(size=self.f.shape) * 0.2

    def test_reference_and_rigid_rotation(self):
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = evaluate(np.array([np.eye(3), rotation]), PARAMETERS)
        for key in ("energy", "pk1", "cauchy"):
            np.testing.assert_allclose(result[key], 0, atol=1e-12)
        np.testing.assert_array_equal(result["J"], [1, 1])

    def test_cauchy_source_equation_and_objectivity(self):
        result = evaluate(self.f, PARAMETERS)
        b = self.f @ self.f.transpose(0, 2, 1)
        j = np.linalg.det(self.f)
        s = np.trace(b, axis1=1, axis2=2) - 3
        stress = (
            (5000 * np.log(j) / j)[:, None, None] * np.eye(3)
            + (200 / j)[:, None, None] * (b - np.eye(3))
            + (2 * 2968.4 * 0.2661 * s * np.exp(0.2661 * s**2) / j)[:, None, None] * b
        )
        np.testing.assert_allclose(result["cauchy"], stress, rtol=2e-13, atol=2e-12)
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotated = evaluate(rotation @ self.f, PARAMETERS)
        np.testing.assert_allclose(rotated["energy"], result["energy"], rtol=2e-13)
        np.testing.assert_allclose(
            rotated["cauchy"], rotation @ stress @ rotation.T, atol=2e-12
        )

    def test_energy_gradient_and_pk1_derivative(self):
        result = evaluate(self.f, PARAMETERS, self.h)
        eps = 1e-6
        plus = evaluate(self.f + eps * self.h, PARAMETERS)
        minus = evaluate(self.f - eps * self.h, PARAMETERS)
        np.testing.assert_allclose(
            (plus["energy"] - minus["energy"]) / (2 * eps),
            np.einsum("nij,nij->n", result["pk1"], self.h),
            rtol=1e-8,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            (plus["pk1"] - minus["pk1"]) / (2 * eps),
            result["tangent_action"],
            rtol=2e-8,
            atol=2e-6,
        )

    def test_tangent_scale_does_not_change_physics(self):
        one = evaluate(self.f, PARAMETERS, self.h)
        two = evaluate(self.f, {**PARAMETERS, "tangent_scale": 1}, self.h)
        for name in one:
            np.testing.assert_array_equal(one[name], two[name])
            self.assertEqual(one[name].dtype, np.float64)

    def test_inversion_and_nonfinite_rejected(self):
        for bad in (
            np.diag([-1.0, 1.0, 1.0]),
            np.zeros((3, 3)),
            np.full((3, 3), np.nan),
        ):
            with self.assertRaises(ValueError):
                evaluate([bad], PARAMETERS)

    def test_cuda_matches_cpu_when_available(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA device unavailable; GPU execution is unverified")
        cpu = evaluate(self.f, PARAMETERS, self.h)
        gpu = evaluate(self.f, PARAMETERS, self.h, device=devices[0])
        for name in cpu:
            np.testing.assert_allclose(gpu[name], cpu[name], rtol=1e-11, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
