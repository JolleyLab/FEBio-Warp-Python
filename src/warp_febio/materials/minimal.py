"""A self-contained neo-Hookean, for the first end-to-end example.

This file exists so the input, model, solve and output path can run without
the material layer proposed in draft pull request #2, which is still under
review. It follows the same convention as that layer, so the element does not
change when the material layer is used instead:

    stress_and_tangent(F, g, p) -> (Cauchy stress 3x3, spatial tangent 6x6)

with the 6x6 in FEBio's tens4ds order (11, 22, 33, 12, 23, 13) and no factors
of two on shear. It is to be removed once the material layer is merged.

FEBio material type : "neo-Hookean"
FEBio source        : https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FENeoHookean.cpp
"""

from dataclasses import dataclass
from typing import Any, Callable

import warp as wp

mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
vec2d = wp.types.vector(length=2, dtype=wp.float64)
vec1d = wp.types.vector(length=1, dtype=wp.float64)


@dataclass(frozen=True)
class MaterialKernel:
    """What an element kernel needs from a material: functions and layouts."""

    febio_type: str
    param_names: tuple  # FEBio tag names, in the order of the parameter vector
    param_vector: Any  # Warp vector type
    geometry_vector: Any  # Warp vector type (material axes, fibers); unused here
    stress_and_tangent: Callable
    backend: str  # "minimal" or "interface"


@wp.func
def neo_hookean_stress_and_tangent(F: wp.mat33d, g: vec1d, p: vec2d):
    """FENeoHookean::Stress and ::Tangent.

    sigma = (mu / J) (b - I) + (lambda ln J / J) I
    c     = lam1 (I (x) I) + 2 mu1 (I (o) I),  lam1 = lambda / J,  mu1 = (mu - lambda ln J) / J
    """
    E = p[0]
    v = p[1]
    one = wp.float64(1.0)
    two = wp.float64(2.0)
    lam = v * E / ((one + v) * (one - two * v))
    mu = E / (two * (one + v))
    J = wp.determinant(F)
    lnJ = wp.log(J)
    b = F * wp.transpose(F)
    I = wp.identity(n=3, dtype=wp.float64)
    sigma = (mu / J) * (b - I) + (lam * lnJ / J) * I
    lam1 = lam / J
    mu1 = (mu - lam * lnJ) / J
    c = mat66d()
    for r in range(3):
        for q in range(3):
            c[r, q] = lam1
        c[r, r] = c[r, r] + two * mu1
        c[r + 3, r + 3] = mu1
    return sigma, c


MINIMAL = {
    "neo-Hookean": MaterialKernel(
        febio_type="neo-Hookean",
        param_names=("E", "v"),
        param_vector=vec2d,
        geometry_vector=vec1d,
        stress_and_tangent=neo_hookean_stress_and_tangent,
        backend="minimal",
    )
}


def minimal_kernel(febio_type: str):
    return MINIMAL.get(febio_type)


def interface_kernel(febio_type: str):
    """The same thing from the material layer of draft pull request #2, if present.

    Returns None when that layer is not installed or has no model of this type
    with a one-pass stress and tangent.
    """
    try:
        from . import model as material_model
        from . import models  # noqa: F401  (registers the models)
    except ImportError:
        return None
    try:
        m = material_model.by_febio_type(febio_type)
    except KeyError:
        return None
    if m.stress_and_tangent is None or m.formulation != "coupled":
        return None
    return MaterialKernel(
        febio_type=febio_type,
        param_names=tuple(m.param_names),
        param_vector=m.param_vector(wp.float64),
        geometry_vector=m.geometry_vector(wp.float64),
        stress_and_tangent=m.stress_and_tangent,
        backend="interface",
    )
