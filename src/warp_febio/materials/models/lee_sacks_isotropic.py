"""Isotropic Lee-Sacks, written in FEBio's spatial form.

FEBio material type : "isotropic Lee-Sacks" (coupled; this is the one the valve decks use)
FEBio source        : https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEIsotropicLeeSacks.cpp
FEBio feature manual: https://febiosoftware.github.io/febio-feature-manual/features/solid_material_isotropic_lee-sacks
Related (uncoupled) : https://febiosoftware.github.io/febio-feature-manual/features/solid_material_uncoupled_isotropic_lee-sacks
Model reference     : Kamensky et al., CMAME 2018, Section 5.2.1 (isotropic Fung-type
                      leaflet model after Lee et al. 2014); used in Wu et al. 2022 and 2023.

Strain energy per reference volume (matches FEIsotropicLeeSacks::StrainEnergyDensity):

    W = c0 [ (I1 - 3)/2 - ln J ]  +  (c1/2) [ exp( c2 (I1 - 3)^2 ) - 1 ]  +  (k/2) (ln J)^2

    I1 = tr(b) = F:F,   J = det F

Parameter vector p, in this order (same names as the FEBio XML tags):
    p[0] = c0   matrix (neo-Hookean) modulus          [pressure]
    p[1] = c1   fiber (exponential) modulus           [pressure]
    p[2] = c2   exponential coefficient               [none]
    p[3] = k    bulk modulus, penalty on volume change [pressure]

Not carried over: FEBio's `tangent_scale`. In FEBio it multiplies c0 in the
tangent only (not in the stress) to make Newton more robust, following Remark 6
of Kamensky et al. It does not change the converged answer. Here the tangent is
exact, i.e. tangent_scale = 1.

The geometry vector g is accepted for a uniform signature and ignored, because
the model is isotropic.
"""

from typing import Any

import warp as wp

from ..model import ParamRange, from_spatial, register
from ..voigt import add_dyad1s, add_identity4s, add_identity_dyad, zeros66


@wp.func
def energy(F: Any, g: Any, p: Any):
    """Strain energy density W(F). Mirrors FEIsotropicLeeSacks::StrainEnergyDensity."""
    c0 = p[0]
    c1 = p[1]
    c2 = p[2]
    k = p[3]
    I1 = wp.ddot(F, F)  # tr(b) = F:F
    lnJ = wp.log(wp.determinant(F))
    half = F.dtype(0.5)  # F.dtype keeps the function generic in FP32/FP64
    s = I1 - F.dtype(3.0)
    return (
        c0 * (half * s - lnJ)  # neo-Hookean matrix
        + half * c1 * (wp.exp(c2 * s * s) - F.dtype(1.0))  # Fung-type exponential term
        + half * k * lnJ * lnJ  # volumetric penalty U(J)
    )


@wp.func
def cauchy(F: Any, g: Any, p: Any):
    """Cauchy stress. Mirrors FEIsotropicLeeSacks::Stress line for line:

    sigma = (k ln J / J) I  +  (c0 / J) (b - I)  +  (2 c1 c2 (I1 - 3) e^Q / J) b
    """
    c0 = p[0]
    c1 = p[1]
    c2 = p[2]
    k = p[3]
    b = F * wp.transpose(F)  # left Cauchy-Green tensor
    J = wp.determinant(F)
    s = wp.trace(b) - F.dtype(3.0)  # I1 - 3
    I = wp.identity(n=3, dtype=F.dtype)
    eQ = wp.exp(c2 * s * s)  # FEBio calls the exponent Q
    return (k * wp.log(J) / J) * I + (c0 / J) * (b - I) + (F.dtype(2.0) * c1 * c2 * s * eQ / J) * b


@wp.func
def spatial_tangent_action(F: Any, d: Any, g: Any, p: Any):
    """Spatial tangent contracted with a symmetric d, i.e. c : d.

    FEIsotropicLeeSacks::Tangent builds c = cw + cp from three fourth-order pieces.
    Contracting each with a symmetric d gives:

        (b (x) b) : d = b (b : d)        FEBio dyad1s(b)
        (I (x) I) : d = I tr(d)          FEBio dyad1s(I)
        (I (o) I) : d = d                FEBio dyad4s(I), the symmetric identity

    so  c : d = cw b (b:d) + (p + J p_J) tr(d) I + (2 c0 / J - 2 p) d
    with p = k ln J / J and p_J = dp/dJ = k (1 - ln J) / J^2.
    """
    c0 = p[0]
    c1 = p[1]
    c2 = p[2]
    k = p[3]
    b = F * wp.transpose(F)
    J = wp.determinant(F)
    s = wp.trace(b) - F.dtype(3.0)
    I = wp.identity(n=3, dtype=F.dtype)
    eQ = wp.exp(c2 * s * s)
    lnJ = wp.log(J)
    pr = k * lnJ / J  # pressure p
    pJ = k * (F.dtype(1.0) - lnJ) / (J * J)  # dp/dJ
    two = F.dtype(2.0)
    cw = F.dtype(4.0) * c1 * c2 * eQ * (F.dtype(1.0) + two * c2 * s * s) / J
    return cw * wp.ddot(b, d) * b + (pr + J * pJ) * wp.trace(d) * I + (two * c0 / J - two * pr) * d


@wp.func
def stress_and_tangent(F: Any, g: Any, p: Any):
    """Cauchy stress and the full spatial tangent in one pass.

    The tangent is FEIsotropicLeeSacks::Tangent as written there (tangent_scale = 1):
        c = dyad1s(b) * cw + dyad1s(I) * (p + J p_J) + dyad4s(I) * (2 c0 / J - 2 p)
    The exponential is evaluated once and shared by stress and tangent.
    """
    c0 = p[0]
    c1 = p[1]
    c2 = p[2]
    k = p[3]
    b = F * wp.transpose(F)
    J = wp.determinant(F)
    s = wp.trace(b) - F.dtype(3.0)
    I = wp.identity(n=3, dtype=F.dtype)
    eQ = wp.exp(c2 * s * s)
    lnJ = wp.log(J)
    pr = k * lnJ / J
    pJ = k * (F.dtype(1.0) - lnJ) / (J * J)
    two = F.dtype(2.0)
    sigma = pr * I + (c0 / J) * (b - I) + (two * c1 * c2 * s * eQ / J) * b
    tangent = zeros66(F)
    tangent = add_dyad1s(tangent, b, F.dtype(4.0) * c1 * c2 * eQ * (F.dtype(1.0) + two * c2 * s * s) / J)
    tangent = add_identity_dyad(tangent, pr + J * pJ)
    tangent = add_identity4s(tangent, two * c0 / J - two * pr)
    return sigma, tangent


# Registered under the FEBio type string so a .feb reader can look it up.
LEE_SACKS_ISOTROPIC = register(
    from_spatial(
        name="lee_sacks_isotropic",
        febio_type="isotropic Lee-Sacks",
        param_names=("c0", "c1", "c2", "k"),
        energy=energy,
        cauchy=cauchy,
        spatial_tangent_action=spatial_tangent_action,
        stress_and_tangent=stress_and_tangent,
        # FEBio declares no ranges for this material. The ones here are the
        # conditions for the energy to be bounded below and are stricter than
        # FEBio: flagged as a deliberate difference.
        param_ranges=(
            ParamRange("c0", low=0.0),
            ParamRange("c1", low=0.0),
            ParamRange("c2", low=0.0),
            ParamRange("k", low=0.0),
        ),
        source="https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEIsotropicLeeSacks.cpp",
    )
)
