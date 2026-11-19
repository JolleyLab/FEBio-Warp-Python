"""Full 3D Saint Venant–Kirchhoff law used by the PatchWarp director shell.

FEBio names this material "isotropic elastic". F and its perturbation H are
dimensionless. Supply Lamé moduli in Pa for stresses in Pa and energy per
reference volume in J/m³. The element owns kinematics and quadrature.
"""

import warp as wp


@wp.func
def green_lagrange_strain(F: wp.mat33d):
    identity = wp.identity(n=3, dtype=wp.float64)
    return wp.float64(0.5) * (wp.transpose(F) * F - identity)


@wp.func
def pk2(strain: wp.mat33d, lame: wp.float64, shear: wp.float64):
    """Second Piola stress given Green–Lagrange strain."""
    identity = wp.identity(n=3, dtype=wp.float64)
    return lame * wp.trace(strain) * identity + wp.float64(2.0) * shear * strain


@wp.func
def energy(F: wp.mat33d, lame: wp.float64, shear: wp.float64):
    strain = green_lagrange_strain(F)
    trace = wp.trace(strain)
    return wp.float64(0.5) * lame * trace * trace + shear * wp.ddot(strain, strain)


@wp.func
def pk1(F: wp.mat33d, lame: wp.float64, shear: wp.float64):
    return F * pk2(green_lagrange_strain(F), lame, shear)


@wp.func
def tangent_action(F: wp.mat33d, H: wp.mat33d, lame: wp.float64, shear: wp.float64):
    """Exact directional derivative of first Piola stress."""
    strain = green_lagrange_strain(F)
    increment = wp.float64(0.5) * (wp.transpose(H) * F + wp.transpose(F) * H)
    return H * pk2(strain, lame, shear) + F * pk2(increment, lame, shear)
