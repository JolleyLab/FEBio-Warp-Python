"""Holzapfel-Gasser-Ogden, uncoupled: the deviatoric part.

FEBio material type : "Holzapfel-Gasser-Ogden"
FEBio source        : https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEHolzapfelGasserOgden.cpp

This is the model the published outflow tract simulations name (Zelonis et
al., arXiv 2507.06337, Table 1). It is FEBio's uncoupled form:

    W = W_dev(F~) + U(J),      F~ = J^(-1/3) F

    W_dev = (c/2)(I1~ - 3) + (k1 / (2 k2)) sum_i (exp(k2 <E_i>^2) - 1)
    E_i   = kappa (I1~ - 3) + (1 - 3 kappa)(I4_i~ - 1)

with the same two fiber families as the unconstrained model
(hgo_unconstrained.py), evaluated on the deviatoric deformation.

What this file provides, and what it does not
---------------------------------------------
Only the deviatoric functions, as FEBio's DevStrainEnergyDensity, DevStress and
DevTangent. The volumetric part U(J) is in volumetric.py; FEBio's default for
this material is pressure model 3, "Abaqus (GOH)", recorded in `options`. An
input can override it: the six open TPVR inputs set pressure_model to
"default", so the pressure model must be taken from the input, not assumed.

Checked against FEBio runs (stress, tangent and energy to about 1e-12) with
the test suite's ma34 constants and with the TPVR vessel constants, including
material axes away from the global frame and both pressure models.

One property to know: under a pure volume change the deviatoric deformation is
a rotation, so both fiber strains are exactly zero, which is where the fibers
switch on. The stress is continuous there; the tangent is not. FEBio takes the
active branch at zero strain (E >= 0) and so does this file.

A model marked formulation="uncoupled" is not accepted by solid_mixture or by
the displacement element of the input and output draft. Following Jeff Weiss,
the two parts are to be combined by a multi-field element, which does not exist
yet, and the uncoupled form is for near-incompressibility only. The tests sum
the two parts as FEUncoupledMaterial::Stress and ::Tangent do, only to compare
with what FEBio logs at a material point.

Parameters follow FEBio: c, k1, k2, kappa, gamma [degrees], and the bulk
modulus k of the uncoupled base class. k does not enter the functions here.
"""

from typing import Any

import warp as wp

from ..model import GeometryField, ParamRange, from_spatial, register
from ..voigt import (
    add_dyad1s,
    add_dyad1s_pair,
    add_identity4s,
    add_identity_dyad,
    contract,
    zeros66,
)
from .hgo_unconstrained import (
    fiber_direction,
    fiber_energy,
    fiber_strain,
    structure_tensor,
)


@wp.func
def deviatoric(a: Any):
    """a - tr(a)/3 I, FEBio's mat3ds::dev."""
    return a - (wp.trace(a) / a.dtype(3.0)) * wp.identity(n=3, dtype=a.dtype)


@wp.func
def isochoric(F: Any):
    """F~ = J^(-1/3) F."""
    return wp.pow(wp.determinant(F), -F.dtype(1.0) / F.dtype(3.0)) * F


@wp.func
def dev_energy(F: Any, g: Any, p: Any):
    """Deviatoric strain energy. Mirrors FEHolzapfelGasserOgden::DevStrainEnergyDensity."""
    c = p[0]
    k1 = p[1]
    k2 = p[2]
    kappa = p[3]
    gamma = p[4]
    one = F.dtype(1.0)
    Ft = isochoric(F)
    I1 = wp.ddot(Ft, Ft)
    E1 = fiber_strain(Ft, fiber_direction(g, gamma, one), kappa)
    E2 = fiber_strain(Ft, fiber_direction(g, gamma, -one), kappa)
    return F.dtype(0.5) * c * (I1 - F.dtype(3.0)) + fiber_energy(E1, k1, k2) + fiber_energy(E2, k1, k2)


@wp.func
def dev_stress_and_tangent(F: Any, g: Any, p: Any):
    """Deviatoric Cauchy stress and spatial tangent (6x6, tens4ds order).

    Mirrors FEHolzapfelGasserOgden::DevStress and ::DevTangent. With
    s = c b~ + sum_i 2 k1 <E_i> exp(k2 E_i^2) h_i  and  sbar = dev(s):

        J sigma_dev = sbar
        J c_dev     = (2/3) tr(s) (dyad4s(I) - dyad1s(I)/3) - (2/3) dyad1s(sbar, I)
                      + sum_i 4 k1 (1 + 2 k2 E_i^2) exp(k2 E_i^2) dyad1s(dev h_i)
    """
    c = p[0]
    k1 = p[1]
    k2 = p[2]
    kappa = p[3]
    gamma = p[4]
    zero = F.dtype(0.0)
    one = F.dtype(1.0)
    two = F.dtype(2.0)
    three = F.dtype(3.0)
    four = F.dtype(4.0)
    J = wp.determinant(F)
    Ft = isochoric(F)
    I = wp.identity(n=3, dtype=F.dtype)
    s = c * (Ft * wp.transpose(Ft))
    fibers = zeros66(F)
    for family in range(2):
        sign = one - two * F.dtype(family)
        a_r = fiber_direction(g, gamma, sign)
        E = fiber_strain(Ft, a_r, kappa)
        h = structure_tensor(Ft, a_r, kappa)
        ex = wp.exp(k2 * E * E)
        active = E >= zero
        s = s + wp.where(active, two * k1 * E * ex, zero) * h
        fibers = add_dyad1s(
            fibers,
            deviatoric(h),
            wp.where(active, four * k1 * (one + two * k2 * E * E) * ex, zero),
        )
    sbar = deviatoric(s)
    trs = wp.trace(s)
    tangent = fibers
    tangent = add_identity4s(tangent, two / three * trs)
    tangent = add_identity_dyad(tangent, -two / three * trs / three)
    tangent = add_dyad1s_pair(tangent, sbar, I, -two / three)
    return sbar / J, tangent / J


@wp.func
def dev_cauchy(F: Any, g: Any, p: Any):
    sigma, tangent = dev_stress_and_tangent(F, g, p)
    return sigma


@wp.func
def dev_spatial_tangent_action(F: Any, d: Any, g: Any, p: Any):
    sigma, tangent = dev_stress_and_tangent(F, g, p)
    return contract(tangent, d)


HGO_UNCOUPLED = register(
    from_spatial(
        name="hgo_uncoupled",
        febio_type="Holzapfel-Gasser-Ogden",
        param_names=("c", "k1", "k2", "kappa", "gamma", "k"),
        energy=dev_energy,
        cauchy=dev_cauchy,
        spatial_tangent_action=dev_spatial_tangent_action,
        geometry=(GeometryField("mat_axis", 9),),
        stress_and_tangent=dev_stress_and_tangent,
        formulation="uncoupled",
        # FEHolzapfelGasserOgden's constructor sets m_npmodel = 3.
        options=(("pressure_model", "Abaqus (GOH)"),),
        # FEHolzapfelGasserOgden.cpp: c, k1, k2 use FE_RANGE_GREATER_OR_EQUAL(0);
        # kappa uses FE_RANGE_CLOSED(0, 1/3); gamma is unchecked.
        # FEUncoupledMaterial.cpp: k uses FE_RANGE_GREATER_OR_EQUAL(0).
        param_ranges=(
            ParamRange("c", low=0.0),
            ParamRange("k1", low=0.0),
            ParamRange("k2", low=0.0),
            ParamRange("kappa", low=0.0, high=1.0 / 3.0),
            ParamRange("k", low=0.0),
        ),
        source="https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEHolzapfelGasserOgden.cpp",
    )
)
