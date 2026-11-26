"""Volumetric energy U(J) for uncoupled materials.

FEBio's uncoupled materials add a volumetric term U(J) to a deviatoric energy,
with four choices selected by the `pressure_model` parameter
(https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEUncoupledMaterial.h):

    0  default        U = (k/2) (ln J)^2
    1  NIKE3D         U = (k/4) (J^2 - 2 ln J - 1)
    2  Abaqus         U = (k/2) (J - 1)^2
    3  Abaqus (GOH)   U = (k/2) ((J^2 - 1)/2 - ln J)

Models 1 and 3 are the same function. FEBio lists them separately, so they are
kept separate here. The MFEM immersed FSI code uses the same energies under the
names QUAD (model 2) and ST91 (models 1 and 3).

`pressure_model` is a build-time option, not an entry of the parameter vector:
it selects code. The bulk modulus k is a float parameter.

An uncoupled model here supplies only its deviatoric functions. These functions
are for the multi-field element that combines the two parts (FEBio's
three-field domain uses U'(J) as the pressure and U''(J) in the stiffness).
Following FEBio practice they are not meant to be summed with the deviatoric
part in a displacement-only element, which locks near incompressibility.
"""

from typing import Any

import warp as wp

PRESSURE_MODELS = {"default": 0, "NIKE3D": 1, "Abaqus": 2, "Abaqus (GOH)": 3}


def make_volumetric(pressure_model="default"):
    """Return (U, dU, d2U) Warp functions of (J, k) for one pressure model."""
    which = PRESSURE_MODELS[pressure_model]

    if which == 0:

        @wp.func
        def U(J: Any, k: Any):
            lnJ = wp.log(J)
            return J.dtype(0.5) * k * lnJ * lnJ

        @wp.func
        def dU(J: Any, k: Any):
            return k * wp.log(J) / J

        @wp.func
        def d2U(J: Any, k: Any):
            return k * (J.dtype(1.0) - wp.log(J)) / (J * J)

    elif which in (1, 3):

        @wp.func
        def U(J: Any, k: Any):
            return J.dtype(0.25) * k * (J * J - J.dtype(2.0) * wp.log(J) - J.dtype(1.0))

        @wp.func
        def dU(J: Any, k: Any):
            return J.dtype(0.5) * k * (J - J.dtype(1.0) / J)

        @wp.func
        def d2U(J: Any, k: Any):
            return J.dtype(0.5) * k * (J.dtype(1.0) + J.dtype(1.0) / (J * J))

    else:

        @wp.func
        def U(J: Any, k: Any):
            return J.dtype(0.5) * k * (J - J.dtype(1.0)) * (J - J.dtype(1.0))

        @wp.func
        def dU(J: Any, k: Any):
            return k * (J - J.dtype(1.0))

        @wp.func
        def d2U(J: Any, k: Any):
            return k * (J / J)

    return U, dU, d2U
