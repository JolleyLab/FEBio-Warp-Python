"""FEBio's linear truss material.

FEBio material type : "linear truss"
FEBio source        : https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FETrussMaterial.cpp
Used by             : the "linear-truss" domain, FEBioMech/FELinearTrussDomain.cpp

The Kirchhoff stress of the truss is linear in the logarithmic stretch,

    tau = E ln(lambda),        lambda = l / L,        d(tau)/d(ln lambda) = E,

which is FELinearTrussMaterial::Stress and ::Tangent. As a uniaxial model its
"strain" is therefore ln(lambda), supplied by the element, and its "stress" is
the Kirchhoff stress.

Parameters: E (> 0) and v, Poisson's ratio in [-1, 0.5], FEBio default 0.5.
v does not enter the stress. The linear-truss domain uses it for the volume
ratio J = lambda^(1 - 2 v) when it reports Cauchy stress tau / J.

Scope
-----
This file holds the material law, and `make_truss_function`, which writes out
how FELinearTrussDomain turns that law into the axial force and 3x3 stiffness
block of one two-node element (static case). It is here so the law can be
checked against its own derivative; the truss element itself (mass, dynamics,
assembly) belongs to the element work and is not attempted.

FEBio also has an "elastic-truss" domain (FEElasticTrussDomain.cpp) that
evaluates any solid material at F = diag(lambda, lambda^-v, lambda^-v). It is
not implemented here: in FEBio develop at 552138d its stiffness uses k = E with
the expression used by the linear-truss domain commented out, so what the
consistent tangent should be is a question for the FEBio team first.
"""

from typing import Any

import warp as wp

from ..uniaxial import UniaxialModel

SOURCE = (
    "FEBioMech/FETrussMaterial.cpp, FEBioMech/FELinearTrussDomain.cpp (FEBio develop)"
)


@wp.func
def linear_truss_update(log_stretch: Any, p: Any, committed: Any):
    """tau = E ln(lambda). No history; the state is returned unchanged."""
    return p[0] * log_stretch, p[0], committed


LINEAR_TRUSS = UniaxialModel(
    name="linear_truss",
    param_names=("E", "v"),
    state_names=("unused",),
    update=linear_truss_update,
    initial_state=(0.0,),
    source=SOURCE,
    febio_type="linear truss",
)


def make_truss_function(model: UniaxialModel = LINEAR_TRUSS):
    """Axial force and stiffness of one truss, as FELinearTrussDomain forms them.

        force_and_stiffness(r0, rt, area, p) -> (force, stiffness block, tau)

    r0, rt : vector from the first to the second node, reference and current
    area   : reference cross-sectional area (the domain's cross_sectional_area)

    With V = L * area the reference volume, n the current unit axis and l the
    current length:

        force     = (tau V / l) n                       on the second node
        stiffness = k n (x) n + (T / l) I,   k = V / l^2 (E - 2 tau),   T = tau V / l
    """
    update = model.update

    @wp.func
    def force_and_stiffness(r0: Any, rt: Any, area: Any, p: Any):
        zero = r0.dtype(0.0)
        two = r0.dtype(2.0)
        L = wp.length(r0)
        l = wp.length(rt)
        n = rt / l
        tau, E, unused = update(wp.log(l / L), p, wp.vector(zero, length=1))
        V = L * area
        T = tau * V / l
        k = V / (l * l) * (E - two * tau)
        K = k * wp.outer(n, n) + (T / l) * wp.identity(n=3, dtype=r0.dtype)
        return T * n, K, tau

    return force_and_stiffness
