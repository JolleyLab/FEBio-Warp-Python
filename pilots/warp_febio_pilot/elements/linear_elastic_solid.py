"""Weak form for isotropic small-strain solids on a displacement space."""

import warp as wp
import warp.fem as fem


@fem.integrand
def linear_elastic_stiffness(
    s: fem.Sample, u: fem.Field, v: fem.Field, lame: wp.float64, shear: wp.float64
):
    # eps(u) is the symmetric small-strain gradient, not Green strain.
    return lame * fem.div(u, s) * fem.div(v, s) + wp.float64(2.0) * shear * wp.ddot(
        fem.D(u, s), fem.D(v, s)
    )
