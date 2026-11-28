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
