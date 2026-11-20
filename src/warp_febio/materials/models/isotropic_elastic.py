"""St. Venant-Kirchhoff, written directly in the reference form.

FEBio material type : "isotropic elastic"
FEBio source        : https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEIsotropicElastic.cpp
FEBio feature manual: https://febiosoftware.github.io/febio-feature-manual/features/solid_material_isotropic_elastic
Used in             : PatchWarp native-quad shell verification; Zelonis et al. (TPVR)
                      text, Eq. 4, for the nitinol device with a load curve on E.

This file is the example of writing a model in the reference form. FEBio writes
this model spatially (Cauchy stress and spatial tangent in terms of b). Here it
is written from its energy in the reference frame instead, and `from_reference`
produces the Cauchy stress and spatial tangent that elements consume. The test
compares those against the closed forms in FEIsotropicElastic.cpp, which checks
the reference-to-spatial direction on a model that was not written spatially.

Strain energy per reference volume (matches FEIsotropicElastic::StrainEnergyDensity):

    W = (lambda/2) (tr E)^2 + mu E:E,     E = (F^T F - I) / 2   (Green-Lagrange strain)

    S = lambda tr(E) I + 2 mu E           second Piola-Kirchhoff stress
    P = F S                               first Piola-Kirchhoff stress

Parameter vector p, in this order (same names as the FEBio XML tags):
    p[0] = E   Young's modulus   [pressure]
    p[1] = v   Poisson's ratio   [none], -1 < v < 0.5

Caution carried over from the existing stvk.py: this law softens and can lose
stability under large compression. It is a benchmark law, not a tissue model.

Q (the local material frame) is ignored; the model is isotropic.
"""

from typing import Any

import warp as wp

from ..model import from_reference, register
from .neo_hookean import lame_lambda, lame_mu  # same E, v -> Lame conversion as FEBio


@wp.func
def green_lagrange(F: Any):
    return (wp.transpose(F) * F - wp.identity(n=3, dtype=F.dtype)) * F.dtype(0.5)


@wp.func
def pk2_from_strain(strain: Any, p: Any):
    """S = lambda tr(E) I + 2 mu E. Linear in the strain, which the tangent uses."""
    two = strain.dtype(2.0)
    I = wp.identity(n=3, dtype=strain.dtype)
    return lame_lambda(p) * wp.trace(strain) * I + two * lame_mu(p) * strain


@wp.func
def energy(F: Any, Q: Any, p: Any):
    strain = green_lagrange(F)
    tr = wp.trace(strain)
    return F.dtype(0.5) * lame_lambda(p) * tr * tr + lame_mu(p) * wp.ddot(strain, strain)


@wp.func
def pk1(F: Any, Q: Any, p: Any):
    return F * pk2_from_strain(green_lagrange(F), p)


@wp.func
def tangent_action(F: Any, H: Any, Q: Any, p: Any):
    """dP[H] = H S(E) + F S(dE),  dE = (H^T F + F^T H) / 2.

    S is linear in E, so its increment is S evaluated at the strain increment.
    """
    dE = (wp.transpose(H) * F + wp.transpose(F) * H) * F.dtype(0.5)
    return H * pk2_from_strain(green_lagrange(F), p) + F * pk2_from_strain(dE, p)


ISOTROPIC_ELASTIC = register(
    from_reference(
        name="isotropic_elastic",
        febio_type="isotropic elastic",
        param_names=("E", "v"),
        energy=energy,
        pk1=pk1,
        tangent_action=tangent_action,
        source="https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEIsotropicElastic.cpp",
    )
)
