"""Compressible neo-Hookean, written in FEBio's spatial form.

FEBio material type : "neo-Hookean"
FEBio source        : https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FENeoHookean.cpp
FEBio feature manual: https://febiosoftware.github.io/febio-feature-manual/features/solid_material_neo-hookean
Used in             : Zelonis et al. (TPVR), Table 1, for the device and the compression tube.

Strain energy per reference volume (matches FENeoHookean::StrainEnergyDensity):

    W = (mu/2) (I1 - 3) - mu ln J + (lambda/2) (ln J)^2

Parameter vector p, in this order (same names as the FEBio XML tags):
    p[0] = E   Young's modulus   [pressure]
    p[1] = v   Poisson's ratio   [none], -1 < v < 0.5

FEBio takes E and v as input and converts to the Lame parameters inside each
call. We do the same, so a .feb value maps straight to p and a gradient with
respect to E or v is available directly.

The geometry vector g is ignored; the model is isotropic.
"""

from typing import Any

import warp as wp

from ..model import ParamRange, from_spatial, register
from ..voigt import add_identity4s, add_identity_dyad, zeros66

# Ranges FEBio enforces on Young's modulus and Poisson's ratio. Shared with
# isotropic_elastic.py, which declares the same two parameters.
ELASTIC_CONSTANT_RANGES = (
    ParamRange("E", low=0.0, low_closed=False),
    ParamRange("v", low=-1.0, high=0.5, low_closed=True, high_closed=False),
)


@wp.func
def lame_lambda(p: Any):
    """lambda = v E / ((1 + v)(1 - 2 v)), as in FENeoHookean::Stress."""
    E = p[0]
    v = p[1]
    one = E / E  # 1 in the same float type as the parameters
    return v * E / ((one + v) * (one - (one + one) * v))


@wp.func
def lame_mu(p: Any):
    """mu = E / (2 (1 + v))."""
    E = p[0]
    v = p[1]
    one = E / E
    return E / ((one + one) * (one + v))


@wp.func
def energy(F: Any, g: Any, p: Any):
    """Strain energy density. Mirrors FENeoHookean::StrainEnergyDensity."""
    lam = lame_lambda(p)
    mu = lame_mu(p)
    lnJ = wp.log(wp.determinant(F))
    half = F.dtype(0.5)
    I1 = wp.ddot(F, F)  # tr(b)
    return mu * (half * (I1 - F.dtype(3.0)) - lnJ) + half * lam * lnJ * lnJ


@wp.func
def cauchy(F: Any, g: Any, p: Any):
    """Cauchy stress. Mirrors FENeoHookean::Stress:

    sigma = (mu / J) (b - I) + (lambda ln J / J) I
    """
    lam = lame_lambda(p)
    mu = lame_mu(p)
    J = wp.determinant(F)
    b = F * wp.transpose(F)
    I = wp.identity(n=3, dtype=F.dtype)
    return (mu / J) * (b - I) + (lam * wp.log(J) / J) * I


@wp.func
def spatial_tangent_action(F: Any, d: Any, g: Any, p: Any):
    """c : d for symmetric d. FENeoHookean::Tangent returns

        c = lam1 (I (x) I) + 2 mu1 (I (o) I),   lam1 = lambda / J,   mu1 = (mu - lambda ln J) / J

    and (I (x) I) : d = tr(d) I,  (I (o) I) : d = d.
    """
    lam = lame_lambda(p)
    mu = lame_mu(p)
    J = wp.determinant(F)
    lam1 = lam / J
    mu1 = (mu - lam * wp.log(J)) / J
    I = wp.identity(n=3, dtype=F.dtype)
    return lam1 * wp.trace(d) * I + F.dtype(2.0) * mu1 * d


@wp.func
def stress_and_tangent(F: Any, g: Any, p: Any):
    """Cauchy stress and the full spatial tangent in one pass.

    The tangent is FENeoHookean::Tangent as written there:
        c = dyad1s(I) * lam1 + dyad4s(I) * (2 mu1)
    """
    lam = lame_lambda(p)
    mu = lame_mu(p)
    J = wp.determinant(F)
    lnJ = wp.log(J)
    b = F * wp.transpose(F)
    I = wp.identity(n=3, dtype=F.dtype)
    sigma = (mu / J) * (b - I) + (lam * lnJ / J) * I
    tangent = zeros66(F)
    tangent = add_identity_dyad(tangent, lam / J)
    tangent = add_identity4s(tangent, F.dtype(2.0) * (mu - lam * lnJ) / J)
    return sigma, tangent


NEO_HOOKEAN = register(
    from_spatial(
        name="neo_hookean",
        febio_type="neo-Hookean",
        param_names=("E", "v"),
        energy=energy,
        cauchy=cauchy,
        spatial_tangent_action=spatial_tangent_action,
        stress_and_tangent=stress_and_tangent,
        # FENeoHookean.cpp: E uses FE_RANGE_GREATER(0), v uses FE_RANGE_RIGHT_OPEN(-1, 0.5).
        param_ranges=ELASTIC_CONSTANT_RANGES,
        source="https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FENeoHookean.cpp",
    )
)
