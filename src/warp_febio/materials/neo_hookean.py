"""Two compressible Neo-Hookean laws, with FP64 Warp material-point functions.

F and its perturbation H are dimensionless. Lamé moduli, reference energy
density, and stresses use Pa. Callers must supply admissible states (det F > 0).
The functions can be called from a Warp kernel or a warp.fem integrand.
See docs/PHYSICS.md for the two source formulations.
"""

import math

import warp as wp


def lame_from_young_poisson(young_pa: float, poisson: float) -> tuple[float, float]:
    """Convert Young's modulus and Poisson's ratio to (lambda, mu)."""
    if not math.isfinite(young_pa) or young_pa <= 0 or not -1.0 < poisson < 0.5:
        raise ValueError("Require finite E > 0 and -1 < nu < 0.5")
    shear = young_pa / (2.0 * (1.0 + poisson))
    return 2.0 * shear * poisson / (1.0 - 2.0 * poisson), shear


@wp.func
def febio_energy(F: wp.mat33d, lame: wp.float64, shear: wp.float64):
    log_j = wp.log(wp.determinant(F))
    return shear * (wp.float64(0.5) * (wp.ddot(F, F) - wp.float64(3.0)) - log_j) + (
        wp.float64(0.5) * lame * log_j * log_j
    )


@wp.func
def febio_pk1(F: wp.mat33d, lame: wp.float64, shear: wp.float64):
    inverse_transpose = wp.transpose(wp.inverse(F))
    return shear * F + (lame * wp.log(wp.determinant(F)) - shear) * inverse_transpose


@wp.func
def febio_tangent_action(
    F: wp.mat33d, H: wp.mat33d, lame: wp.float64, shear: wp.float64
):
    """Exact directional derivative of PK1, expressed in reference coordinates."""
    inverse_transpose = wp.transpose(wp.inverse(F))
    coefficient = lame * wp.log(wp.determinant(F)) - shear
    # d(log J)[H] = F^-T : H; d(F^-T)[H] = -F^-T H^T F^-T.
    return (
        shear * H
        + lame * wp.ddot(inverse_transpose, H) * inverse_transpose
        - coefficient * (inverse_transpose * wp.transpose(H) * inverse_transpose)
    )


@wp.func
def _cofactor_2d(F: wp.mat22d):
    # The determinant gradient in two dimensions is linear in matrix entries.
    return wp.mat22d(F[1, 1], -F[1, 0], -F[0, 1], F[0, 0])


@wp.func
def stable_energy_2d(F: wp.mat22d, lame: wp.float64, shear: wp.float64):
    """Stable energy from the Warp mixed example's stress, normalized at F=I."""
    volume_change = wp.determinant(F) - wp.float64(1.0)
    return (
        wp.float64(0.5) * shear * (wp.ddot(F, F) - wp.float64(2.0))
        + wp.float64(0.5) * (lame + shear) * volume_change * volume_change
        - shear * volume_change
    )


@wp.func
def stable_pk1_2d(F: wp.mat22d, lame: wp.float64, shear: wp.float64):
    volume_stress = (lame + shear) * (wp.determinant(F) - wp.float64(1.0)) - shear
    return shear * F + volume_stress * _cofactor_2d(F)


@wp.func
def stable_gauss_newton_action_2d(
    F: wp.mat22d, H: wp.mat22d, lame: wp.float64, shear: wp.float64
):
    """Approximate tangent used by the linked Warp example."""
    determinant_gradient = _cofactor_2d(F)
    return (
        shear * H
        + (lame + shear) * wp.ddot(determinant_gradient, H) * determinant_gradient
    )


@wp.func
def stable_tangent_action_2d(
    F: wp.mat22d, H: wp.mat22d, lame: wp.float64, shear: wp.float64
):
    """Exact tangent, including the derivative of the determinant gradient."""
    volume_stress = (lame + shear) * (wp.determinant(F) - wp.float64(1.0)) - shear
    return stable_gauss_newton_action_2d(
        F, H, lame, shear
    ) + volume_stress * _cofactor_2d(H)
