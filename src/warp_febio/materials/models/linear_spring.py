"""FEBio's linear spring and tension-only linear spring.

DRAFT (chordae elements, November 2026). Structure for discussion, not final.

FEBio material types : "linear spring", "tension-only linear spring"
FEBio source         : https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FESpringMaterial.cpp

Both are discrete materials on the elongation dl = L - L0:

    linear spring                F = E dl
    tension-only linear spring   F = E (dl - slack_length)   for dl >= slack_length
                                 F = 0                        otherwise

which are FELinearSpring::force and FETensionOnlyLinearSpring::force, with
::stiffness their derivatives (E, or 0 below the slack length).

Why the second one is here. The chordae of the published valve models are
tension-only, written in the open mitral input as a "nonlinear spring" with a
three-point curve (zero up to a slack elongation, then a straight line). The
tension-only linear spring is FEBio's closed form of the same law: scale times
the slope of the curve is E, and the knot is slack_length. As a parameter set
it is the more direct one for gradients, since the slack length and the
stiffness are the parameters themselves and not coordinates of curve points.
The two are compared in tests/test_chordae_elements.py.

A third law here is not FEBio's: the "smooth tension-only linear spring",
the tension-only linear spring with its corner rounded over a band around the
slack length. It is for optimization, where chords that sit at their slack
length make the gradient of the FEBio law one-sided (decision C9 of
docs/design/chordae.md). See smooth_tension_only_linear_spring_update.

As uniaxial models their "strain" is the elongation and their "stress" is the
spring force, as for the nonlinear spring (nonlinear_spring.py), so the same
discrete domain evaluates all three.
"""

from typing import Any

import warp as wp

from ..uniaxial import UniaxialModel

SOURCE = "FEBioMech/FESpringMaterial.cpp (FEBio develop)"


@wp.func
def linear_spring_update(dl: Any, p: Any, committed: Any):
    """F = E dl. No history; the state is returned unchanged."""
    return p[0] * dl, p[0], committed


@wp.func
def tension_only_linear_spring_update(dl: Any, p: Any, committed: Any):
    """F = E (dl - slack_length) at or beyond the slack length, else zero.

    The branch is FEBio's (extension >= 0 is active). The force is continuous
    at the slack length and the stiffness is not, so a gradient with respect
    to the slack length is one-sided there.
    """
    zero = dl.dtype(0.0)
    extension = dl - p[1]
    active = extension >= zero
    return wp.where(active, p[0] * extension, zero), wp.where(active, p[0], zero), committed


@wp.func
def smooth_tension_only_linear_spring_update(dl: Any, p: Any, committed: Any):
    """The tension-only linear spring with a rounded corner. FEBio-Warp only.

    With x = dl - slack_length and w the smoothing length (p[2]):

        x <= -w          F = 0                      dF/dl = 0
        -w < x < w       F = E (x + w)^2 / (4 w)    dF/dl = E (x + w) / (2 w)
        x >= w           F = E x                    dF/dl = E

    The force and the stiffness are both continuous, the stiffness is the
    exact derivative of the force, and outside the band the law is FEBio's
    "tension-only linear spring". With w = 0 it is FEBio's law everywhere,
    with FEBio's branch at the slack length (active at x = 0).

    Relation to FEBio. FEBio has no such material. Its names, its parameters
    E and slack_length, its measure (elongation) and its branch at the slack
    length are those of FETensionOnlyLinearSpring, so the two differ only
    inside the band. FEBio's own way to a smooth force curve is a "nonlinear
    spring" with a point curve and `smooth` interpolation (a spline through
    the points, used by suite problem di04), which is not implemented here
    yet. A spline through the three points of the chord curve is not
    tension-only: it does not stay at zero below the knot. This closed form
    does, and its parameters are the design variables themselves.

    Written without a data-dependent branch (wp.where and wp.clamp), so one
    kernel serves slack, rounded and taut springs alike on the device.

    What the rounding costs: inside the band the force differs from FEBio's
    by at most E w / 4, reached at the slack length, where this spring pulls
    with E w / 4 and FEBio's with nothing. So w is chosen small against the
    elongations of interest, and results are not FEBio's to within that
    difference.
    """
    zero = dl.dtype(0.0)
    x = dl - p[1]
    w = p[2]
    w_safe = wp.where(w > zero, w, dl.dtype(1.0))
    t = wp.clamp(x + w, zero, dl.dtype(2.0) * w)
    beyond = x >= w
    force = wp.where(beyond, p[0] * x, p[0] * t * t / (dl.dtype(4.0) * w_safe))
    slope = wp.where(beyond, p[0], p[0] * t / (dl.dtype(2.0) * w_safe))
    return force, slope, committed


LINEAR_SPRING = UniaxialModel(
    name="linear_spring",
    param_names=("E",),
    state_names=("unused",),
    update=linear_spring_update,
    initial_state=(0.0,),
    source=SOURCE,
    febio_type="linear spring",
    options=(("measure", "elongation"),),
)

TENSION_ONLY_LINEAR_SPRING = UniaxialModel(
    name="tension_only_linear_spring",
    param_names=("E", "slack_length"),
    state_names=("unused",),
    update=tension_only_linear_spring_update,
    initial_state=(0.0,),
    source=SOURCE,
    febio_type="tension-only linear spring",
    options=(("measure", "elongation"),),
)

# FEBio-Warp only: FEBio has no smoothed tension-only spring. An input that
# uses this type is not a FEBio input.
SMOOTH_TENSION_ONLY_LINEAR_SPRING = UniaxialModel(
    name="smooth_tension_only_linear_spring",
    param_names=("E", "slack_length", "smoothing_length"),
    state_names=("unused",),
    update=smooth_tension_only_linear_spring_update,
    initial_state=(0.0,),
    source="FEBio-Warp (not a FEBio material); outside its band it is FETensionOnlyLinearSpring, " + SOURCE,
    febio_type="smooth tension-only linear spring",
    options=(("measure", "elongation"),),
)
