"""Check the reusable StVK functions against the PatchWarp/FEBio equations."""

import unittest

import numpy as np
import warp as wp

from warp_febio_pilot.materials import stvk


@wp.kernel
def response(
    F: wp.mat33d,
    H: wp.mat33d,
    lame: wp.float64,
    shear: wp.float64,
    energy: wp.array(dtype=wp.float64),
    stress: wp.array(dtype=wp.mat33d),
):
    energy[0] = stvk.energy(F, lame, shear)
    stress[0] = stvk.pk1(F, lame, shear)
    stress[1] = stvk.tangent_action(F, H, lame, shear)


class StVKTests(unittest.TestCase):
    def test_stress_energy_and_tangent(self):
        young, poisson = 1e6, 0.3
        lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        shear = young / (2 * (1 + poisson))
        F = np.array([[1.08, 0.07, 0.0], [0.02, 0.96, 0.03], [0.01, 0.0, 1.04]])
        H = np.array([[0.2, -0.12, 0.03], [0.1, -0.03, 0.04], [-0.1, 0.02, 0.05]])
        B = F @ F.T
        J = np.linalg.det(F)
        expected = (B * (0.5 * lame * (np.trace(B) - 3) - shear) + shear * B @ B) / J
        for device in ["cpu", *wp.get_cuda_devices()[:1]]:
            with self.subTest(device=str(device)):
                energy = wp.zeros(1, dtype=wp.float64, device=device)
                stress = wp.zeros(2, dtype=wp.mat33d, device=device)

                def evaluate(matrix):
                    wp.launch(
                        response,
                        1,
                        inputs=[wp.mat33d(matrix), wp.mat33d(H), lame, shear],
                        outputs=[energy, stress],
                        device=device,
                    )
                    return energy.numpy()[0], stress.numpy().copy()

                rest_energy, rest_stress = evaluate(np.eye(3))
                self.assertEqual(rest_energy, 0.0)
                np.testing.assert_array_equal(rest_stress[0], np.zeros((3, 3)))
                _, stress_at_F = evaluate(F)
                np.testing.assert_allclose(stress_at_F[0] @ F.T / J, expected)
                step = 1e-6
                plus_e, plus = evaluate(F + step * H)
                minus_e, minus = evaluate(F - step * H)
                np.testing.assert_allclose(
                    (plus_e - minus_e) / (2 * step),
                    np.sum(stress_at_F[0] * H),
                    rtol=1e-7,
                )
                np.testing.assert_allclose(
                    (plus[0] - minus[0]) / (2 * step),
                    stress_at_F[1],
                    rtol=1e-7,
                    atol=1e-3,
                )
