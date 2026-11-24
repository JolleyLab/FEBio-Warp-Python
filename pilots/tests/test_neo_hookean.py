"""Verify both laws through their actual Warp device functions."""

import unittest

import numpy as np
import warp as wp

from warp_febio.materials import neo_hookean as nh


@wp.kernel
def febio_response(
    F: wp.mat33d,
    H: wp.mat33d,
    lame: wp.float64,
    shear: wp.float64,
    W: wp.array(dtype=wp.float64),
    response: wp.array(dtype=wp.mat33d),
):
    W[0] = nh.febio_energy(F, lame, shear)
    response[0] = nh.febio_pk1(F, lame, shear)
    response[1] = nh.febio_tangent_action(F, H, lame, shear)


@wp.kernel
def stable_response(
    F: wp.mat22d,
    H: wp.mat22d,
    lame: wp.float64,
    shear: wp.float64,
    W: wp.array(dtype=wp.float64),
    response: wp.array(dtype=wp.mat22d),
):
    W[0] = nh.stable_energy_2d(F, lame, shear)
    response[0] = nh.stable_pk1_2d(F, lame, shear)
    response[1] = nh.stable_tangent_action_2d(F, H, lame, shear)
    response[2] = nh.stable_gauss_newton_action_2d(F, H, lame, shear)


class NeoHookeanTests(unittest.TestCase):
    def check_response(self, kernel, dtype, F, H, count, device):
        lame, shear = nh.lame_from_young_poisson(1.0e6, 0.45)
        energy = wp.zeros(1, dtype=wp.float64, device=device)
        response = wp.zeros(count, dtype=dtype, device=device)

        def evaluate(matrix):
            wp.launch(
                kernel,
                dim=1,
                inputs=[dtype(matrix), dtype(H), lame, shear],
                outputs=[energy, response],
                device=device,
            )
            return energy.numpy()[0], response.numpy().copy()

        reference_energy, reference = evaluate(np.eye(len(F)))
        np.testing.assert_allclose(reference_energy, 0.0)
        np.testing.assert_allclose(reference[0], 0.0)
        _, values = evaluate(F)
        step = 1.0e-6
        plus_e, plus = evaluate(F + step * H)
        minus_e, minus = evaluate(F - step * H)
        np.testing.assert_allclose(
            (plus_e - minus_e) / (2 * step), np.sum(values[0] * H), rtol=1.0e-7
        )
        np.testing.assert_allclose(
            (plus[0] - minus[0]) / (2 * step), values[1], rtol=1.0e-7, atol=1.0e-3
        )
        return values, lame, shear

    def test_material_laws(self):
        for device in ["cpu", *wp.get_cuda_devices()[:1]]:
            with self.subTest(device=str(device)):
                F = np.array([[1.12, 0.07, 0.0], [0.02, 0.96, 0.03], [0.01, 0.0, 1.04]])
                H = np.array(
                    [[0.2, -0.12, 0.03], [0.1, -0.03, 0.04], [-0.1, 0.02, 0.05]]
                )
                values, lame, shear = self.check_response(
                    febio_response, wp.mat33d, F, H, 2, device
                )
                J = np.linalg.det(F)
                expected_cauchy = (
                    shear * (F @ F.T - np.eye(3)) + lame * np.log(J) * np.eye(3)
                ) / J
                np.testing.assert_allclose(values[0] @ F.T / J, expected_cauchy)
                stable, _, _ = self.check_response(
                    stable_response, wp.mat22d, F[:2, :2], H[:2, :2], 3, device
                )
                self.assertGreater(np.linalg.norm(stable[1] - stable[2]), 1.0)
