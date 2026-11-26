"""Holzapfel-Gasser-Ogden, unconstrained (coupled), written in FEBio's spatial form.

FEBio material type : "HGO unconstrained"
FEBio source        : https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEHolzapfelUnconstrained.cpp
FEBio feature manual: https://febiosoftware.github.io/febio-feature-manual/features/solid_material_hgo_unconstrained
Model reference     : Gasser, Ogden and Holzapfel, J. R. Soc. Interface 3 (2006) 15-35.
Used for            : arterial and outflow tract wall. This is the coupled
                      counterpart of the uncoupled "Holzapfel-Gasser-Ogden" used
                      for the right ventricular outflow tract in Zelonis et al.
                      (arXiv 2507.06337). Following FEBio practice (Jeff Weiss),
                      the coupled form with a displacement-based element is the
                      default unless near-incompressibility is required.

Two fiber families lie in the plane of the first two material axes, at +gamma
and -gamma from the first axis:

    a_r1 = cos(gamma) n1 + sin(gamma) n2,     a_r2 = cos(gamma) n1 - sin(gamma) n2

where n1 and n2 are the first two columns of the material axes Q.

Strain energy per reference volume (matches FEHolzapfelUnconstrained::StrainEnergyDensity):

    W = (c/2)(I1 - 3) - c ln J + (k/2)[(J^2 - 1)/2 - ln J]
        + (k1 / (2 k2)) sum_i [ exp(k2 <E_i>^2) - 1 ]

    E_i = kappa (I1 - 3) + (1 - 3 kappa)(I4_i - 1),     I4_i = a_ri . C a_ri

<E> means the fiber term is active only for E >= 0 (fibers carry no compression).

Parameter vector p, in this order (same names as the FEBio XML tags):
    p[0] = c      ground matrix shear modulus            [pressure], >= 0
    p[1] = k1     fiber modulus                          [pressure], >= 0
    p[2] = k2     fiber exponential coefficient          [none],     >= 0
    p[3] = kappa  fiber dispersion                       [none],     0 <= kappa <= 1/3
    p[4] = gamma  fiber angle from the first axis        [degrees]
    p[5] = k      bulk modulus-like penalty              [pressure], >= 0

Geometry vector g:
    "mat_axis"  9 floats, the material axes Q row by row. Its columns are the
                FEBio mat_axis vectors. FEBio reads the fiber plane from the
                first two columns, and so does this model.

Deliberate differences from FEBio:
- For k2 = 0 FEBio's energy divides by zero, although its stress and tangent
  are finite. Here the energy is written so that it takes its limit,
  (k1/2) E^2, smoothly, and its derivative with respect to k2 is correct there.
- The fiber switch is written with wp.where instead of an if-block. Values are
  identical; the form keeps both families in one branch-free code path.

Note on kappa: at kappa = 1/3 the fiber term depends on I1 only and the model is
isotropic. The published outflow tract value, 0.33, is close to that limit. In
the test states the choice of material axes changes the stress by up to about
4.5 percent at kappa = 0.33, against about 90 percent at kappa = 0.1. So fiber
angle and axes have a small but not negligible effect at the published value
(tests/test_material_architecture.py).
"""

from typing import Any

import warp as wp

from ..model import GeometryField, ParamRange, from_spatial, register
from ..voigt import add_dyad1s, add_identity4s, add_identity_dyad, zeros66

_DEGREES_TO_RADIANS = 0.017453292519943295


@wp.func
def fiber_direction(g: Any, gamma: Any, sign: Any):
    """Reference fiber direction cos(gamma) n1 + sign sin(gamma) n2.

    g holds Q row by row, so column 0 is (g[0], g[3], g[6]) and column 1 is
    (g[1], g[4], g[7]), as in FEHolzapfelUnconstrained::Stress.
    """
    n1 = wp.vector(g[0], g[3], g[6])
    n2 = wp.vector(g[1], g[4], g[7])
    angle = gamma * gamma.dtype(_DEGREES_TO_RADIANS)
    return wp.cos(angle) * n1 + sign * wp.sin(angle) * n2


@wp.func
def fiber_strain(F: Any, a_r: Any, kappa: Any):
    """E = kappa (I1 - 3) + (1 - 3 kappa)(I4 - 1), with I4 = |F a_r|^2."""
    a = F * a_r
    I1 = wp.ddot(F, F)
    one = F.dtype(1.0)
    three = F.dtype(3.0)
    return kappa * (I1 - three) + (one - three * kappa) * (wp.dot(a, a) - one)


@wp.func
def exp_minus_one_over(x: Any):
    """(exp(x) - 1) / x, continued smoothly through x = 0.

    Near zero the quotient loses digits and is undefined at zero, so a short
    series is used there. This keeps the energy finite and its derivative with
    respect to k2 correct at k2 = 0, where FEBio's expression divides by zero.
    """
    one = x.dtype(1.0)
    small = wp.abs(x) < x.dtype(1.0e-5)
    series = one + x * (x.dtype(0.5) + x * x.dtype(1.0 / 6.0))
    quotient = (wp.exp(x) - one) / wp.where(small, one, x)
    return wp.where(small, series, quotient)


@wp.func
def fiber_energy(E: Any, k1: Any, k2: Any):
    """(k1 / (2 k2)) (exp(k2 E^2) - 1) for E >= 0, else 0.

    Written as (k1/2) E^2 f(k2 E^2) with f(x) = (exp(x) - 1) / x, which is the
    same value and has the limit (k1/2) E^2 at k2 = 0.
    """
    zero = E.dtype(0.0)
    value = E.dtype(0.5) * k1 * E * E * exp_minus_one_over(k2 * E * E)
    return wp.where(E >= zero, value, zero)


@wp.func
def structure_tensor(F: Any, a_r: Any, kappa: Any):
    """h = kappa b + (1 - 3 kappa) a (x) a, with a = F a_r."""
    a = F * a_r
    b = F * wp.transpose(F)
    return kappa * b + (F.dtype(1.0) - F.dtype(3.0) * kappa) * wp.outer(a, a)


@wp.func
def energy(F: Any, g: Any, p: Any):
    """Strain energy density. Mirrors FEHolzapfelUnconstrained::StrainEnergyDensity."""
    c = p[0]
    k1 = p[1]
    k2 = p[2]
    kappa = p[3]
    gamma = p[4]
    k = p[5]
    one = F.dtype(1.0)
    half = F.dtype(0.5)
    J = wp.determinant(F)
    lnJ = wp.log(J)
    I1 = wp.ddot(F, F)
    ground = (
        half * c * (I1 - F.dtype(3.0))
        - c * lnJ
        + half * k * (half * (J * J - one) - lnJ)
    )
    E1 = fiber_strain(F, fiber_direction(g, gamma, one), kappa)
    E2 = fiber_strain(F, fiber_direction(g, gamma, -one), kappa)
    return ground + fiber_energy(E1, k1, k2) + fiber_energy(E2, k1, k2)


@wp.func
def cauchy(F: Any, g: Any, p: Any):
    """Cauchy stress. Mirrors FEHolzapfelUnconstrained::Stress:

    J sigma = c (b - I) + (k/2)(J^2 - 1) I + sum_i 2 k1 <E_i> exp(k2 E_i^2) h_i
    """
    c = p[0]
    k1 = p[1]
    k2 = p[2]
    kappa = p[3]
    gamma = p[4]
    k = p[5]
    zero = F.dtype(0.0)
    one = F.dtype(1.0)
    two = F.dtype(2.0)
    J = wp.determinant(F)
    b = F * wp.transpose(F)
    I = wp.identity(n=3, dtype=F.dtype)
    s = c * (b - I) + (F.dtype(0.5) * k * (J * J - one)) * I
    a1 = fiber_direction(g, gamma, one)
    a2 = fiber_direction(g, gamma, -one)
    E1 = fiber_strain(F, a1, kappa)
    E2 = fiber_strain(F, a2, kappa)
    # Macaulay bracket: a family contributes only when its strain is not negative.
    w1 = wp.where(E1 >= zero, two * k1 * E1 * wp.exp(k2 * E1 * E1), zero)
    w2 = wp.where(E2 >= zero, two * k1 * E2 * wp.exp(k2 * E2 * E2), zero)
    s = s + w1 * structure_tensor(F, a1, kappa) + w2 * structure_tensor(F, a2, kappa)
    return s / J


@wp.func
def spatial_tangent_action(F: Any, d: Any, g: Any, p: Any):
    """c : d for symmetric d. FEHolzapfelUnconstrained::Tangent returns

        J c = (2c - (J^2 - 1) k) (I (o) I) + J^2 k (I (x) I)
              + sum_i 4 k1 (1 + 2 k2 E_i^2) exp(k2 E_i^2) (h_i (x) h_i)     for E_i >= 0

    and (I (o) I) : d = d,  (I (x) I) : d = tr(d) I,  (h (x) h) : d = h (h : d).
    """
    c = p[0]
    k1 = p[1]
    k2 = p[2]
    kappa = p[3]
    gamma = p[4]
    k = p[5]
    zero = F.dtype(0.0)
    one = F.dtype(1.0)
    two = F.dtype(2.0)
    four = F.dtype(4.0)
    J = wp.determinant(F)
    I = wp.identity(n=3, dtype=F.dtype)
    out = (two * c - (J * J - one) * k) * d + (J * J * k * wp.trace(d)) * I
    a1 = fiber_direction(g, gamma, one)
    a2 = fiber_direction(g, gamma, -one)
    E1 = fiber_strain(F, a1, kappa)
    E2 = fiber_strain(F, a2, kappa)
    h1 = structure_tensor(F, a1, kappa)
    h2 = structure_tensor(F, a2, kappa)
    w1 = wp.where(
        E1 >= zero, four * k1 * (one + two * k2 * E1 * E1) * wp.exp(k2 * E1 * E1), zero
    )
    w2 = wp.where(
        E2 >= zero, four * k1 * (one + two * k2 * E2 * E2) * wp.exp(k2 * E2 * E2), zero
    )
    out = out + (w1 * wp.ddot(h1, d)) * h1 + (w2 * wp.ddot(h2, d)) * h2
    return out / J


@wp.func
def stress_and_tangent(F: Any, g: Any, p: Any):
    """Cauchy stress and the full spatial tangent (6x6, tens4ds order) in one pass.

    Same values as `cauchy` and six calls of `spatial_tangent_action`, but the
    fiber directions, strains, exponentials and structure tensors are computed
    once and shared. On CPU this was 2.2 times faster than stress plus six
    actions for one million points. Elements that assemble a stiffness matrix
    call this; matrix-free operators call the action.
    """
    c = p[0]
    k1 = p[1]
    k2 = p[2]
    kappa = p[3]
    gamma = p[4]
    k = p[5]
    zero = F.dtype(0.0)
    one = F.dtype(1.0)
    two = F.dtype(2.0)
    four = F.dtype(4.0)
    J = wp.determinant(F)
    b = F * wp.transpose(F)
    I = wp.identity(n=3, dtype=F.dtype)
    s = c * (b - I) + (F.dtype(0.5) * k * (J * J - one)) * I
    # J c = (2c - (J^2 - 1) k) dyad4s(I) + J^2 k dyad1s(I) + fiber terms
    tangent = zeros66(F)
    tangent = add_identity4s(tangent, two * c - (J * J - one) * k)
    tangent = add_identity_dyad(tangent, J * J * k)
    for family in range(2):
        sign = one - two * F.dtype(family)
        a_r = fiber_direction(g, gamma, sign)
        E = fiber_strain(F, a_r, kappa)
        h = structure_tensor(F, a_r, kappa)
        ex = wp.exp(k2 * E * E)
        active = E >= zero
        s = s + wp.where(active, two * k1 * E * ex, zero) * h
        tangent = add_dyad1s(
            tangent,
            h,
            wp.where(active, four * k1 * (one + two * k2 * E * E) * ex, zero),
        )
    return s / J, tangent / J


HGO_UNCONSTRAINED = register(
    from_spatial(
        name="hgo_unconstrained",
        febio_type="HGO unconstrained",
        param_names=("c", "k1", "k2", "kappa", "gamma", "k"),
        energy=energy,
        cauchy=cauchy,
        spatial_tangent_action=spatial_tangent_action,
        geometry=(GeometryField("mat_axis", 9),),
        stress_and_tangent=stress_and_tangent,
        # FEHolzapfelUnconstrained.cpp: c, k1, k2, k use FE_RANGE_GREATER_OR_EQUAL(0);
        # kappa uses FE_RANGE_CLOSED(0, 1/3); gamma is unchecked.
        param_ranges=(
            ParamRange("c", low=0.0),
            ParamRange("k1", low=0.0),
            ParamRange("k2", low=0.0),
            ParamRange("kappa", low=0.0, high=1.0 / 3.0),
            ParamRange("k", low=0.0),
        ),
        source="https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEHolzapfelUnconstrained.cpp",
    )
)
