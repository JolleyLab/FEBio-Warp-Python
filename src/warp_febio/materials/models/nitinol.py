"""Nitinol at two levels of fidelity (draft v0.3).

Level 1: effective modulus
--------------------------
The model used in the published transcatheter pulmonary valve simulations
(Zelonis et al., arXiv 2507.06337; sensitivity study arXiv 2601.05410). The
frame is an elastic solid, and a step load curve on Young's modulus stands in
for the change of effective stiffness between compression and release. It has
no hysteresis and no stress plateau. It is fast, differentiable in E, and
matches what the published FEBio input files do.

It needs no constitutive code of its own: it is an existing elastic model plus
a parameter schedule. `effective_modulus_nitinol` builds that pairing.

The paper's text describes the device law as FEBio "isotropic elastic" (its
Eq. 4) while its Table 1 lists neo-Hookean. The six open input files in the
FEBio model repository (files 3598 to 3603) settle it: the frame is
"isotropic elastic" with E = 4.0e7, v = 0.33, density = 6.45e-6 and a load
curve on E, and the two crimping tubes are "neo-Hookean" shells with the same
constants and no load curve. The load curve is the same in all six files:

    <interpolate>STEP</interpolate> <extend>CONSTANT</extend>
    <points> <pt>0.998063,1</pt> <pt>1,0.75</pt> </points>

so E is 4.0e7 through t = 0.998063 and 3.0e7 after it (see schedule.LoadCurve
for FEBio's step rule). `published_tpvr_frame` returns exactly that.

Level 2: superelastic, uniaxial, for beam cross-sections (SHELL)
----------------------------------------------------------------
This level is a shell. It fixes the interface a superelastic law plugs into
(a uniaxial model with committed and trial history, a consistent tangent, and
gradients through the history) and carries a simple law so that interface can
be exercised and tested. Refining the law itself is separate, dedicated work.
Do not use it for device results.

A stress-induced austenite to martensite transformation with a single internal
variable, the martensite fraction xi, written for the fibers of a beam
cross-section. Wire-frame devices deform mostly in bending, and a beam with a
fiber-integrated section needs only this uniaxial law.

Constitutive equations (isothermal, small axial strain, one elastic modulus):

    sigma = E (eps - eps_L xi sgn(eps)),      0 <= xi <= 1

    forward (austenite to martensite), while |sigma| rises between s_AM_s and s_AM_f:
        d(xi) = (1 - xi) d|sigma| / (s_AM_f - |sigma|)
    reverse (martensite to austenite), while |sigma| falls between s_MA_s and s_MA_f:
        d(xi) = xi d|sigma| / (|sigma| - s_MA_f)

with s_MA_f < s_MA_s < s_AM_s < s_AM_f. These are linear kinetic rules of the
kind introduced by Auricchio, Taylor and Lubliner for superelastic alloys
(Comput. Methods Appl. Mech. Engrg. 146 (1997) 281-312; the finite-strain
companion, Auricchio and Taylor, 143 (1997) 175-194, is reference 56 of Zelonis
et al.). THE EQUATIONS ABOVE WERE WRITTEN FROM THE GENERAL FORM OF THAT MODEL
AND HAVE NOT YET BEEN CHECKED LINE BY LINE AGAINST THE PAPERS. Treat this as a
draft of the structure and the update algorithm, to be verified against the
primary source and against uniaxial test data before any use.

Both rules integrate exactly over a step. xi is linear in |sigma| along each
branch, so with sigma linear in xi the update is one linear equation, with no
local Newton iteration and no dependence on the step size:

    forward:  xi = [1 - A (s_AM_f - E e)] / (1 + A E eps_L),  A = (1 - xi_n) / (s_AM_f - s_0)
    reverse:  xi = B (E e - s_MA_f) / (1 + B E eps_L),        B = xi_n / (s_0 - s_MA_f)

where e = |eps| and s_0 is the stress at which the branch becomes active in this
step: max(|sigma_n|, s_AM_s) on the forward branch, min(|sigma_n|, s_MA_s) on
the reverse branch.

Simplifications to remove later, in this order: equal moduli for austenite and
martensite; symmetric tension and compression; isothermal; small axial strain.
There is no FEBio counterpart to verify against. FEBio has no shape-memory
material, and adding one there is deferred.
"""

from typing import Any

import warp as wp

from ..model import Material
from ..schedule import LoadCurve
from ..uniaxial import UniaxialModel
from . import isotropic_elastic, neo_hookean


def effective_modulus_nitinol(
    E, v, density, release_time, release_factor, law="isotropic elastic"
):
    """Level 1 nitinol: an elastic law with a step load curve on E.

    E, v, density: as in the FEBio input.
    release_time: E keeps its value through this time.
    release_factor: multiplier applied to E after release_time.
    law: "isotropic elastic" (what the open TPVR inputs use for the frame) or
        "neo-Hookean".
    """
    model = {
        "neo-Hookean": neo_hookean.NEO_HOOKEAN,
        "isotropic elastic": isotropic_elastic.ISOTROPIC_ELASTIC,
    }[law]
    # A FEBio STEP curve takes the later point's value between two points, so
    # the second point only has to lie after release_time.
    curve = LoadCurve(
        times=(release_time, release_time + 1.0),
        values=(1.0, release_factor),
        interpolate="step",
    )
    return Material(
        model=model, params={"E": E, "v": v}, density=density, schedules={"E": curve}
    )


def published_tpvr_frame():
    """The device frame material of the six open TPVR inputs (kPa, mm, kg/mm^3)."""
    return Material(
        model=isotropic_elastic.ISOTROPIC_ELASTIC,
        params={"E": 4.0e7, "v": 0.33},
        density=6.45e-6,
        schedules={
            "E": LoadCurve(
                times=(0.998063, 1.0), values=(1.0, 0.75), interpolate="step"
            )
        },
    )


@wp.func
def superelastic_update(strain: Any, p: Any, committed: Any):
    """One step of the uniaxial superelastic law. See the module docstring.

    Two points matter for differentiation and are checked by a test of the
    gradient through a strain history against finite differences
    (tests/test_material_architecture.py):

    - The new state is built with a constructor. An earlier version copied the
      committed state and wrote its components (trial = committed;
      trial[0] = xi). That gave wrong reverse-mode gradients through the
      history in Warp 1.19, while forward values were correct.
    - Candidate values for each branch are computed and chosen with wp.where.
      Denominators of branches that are not taken are replaced by 1 so they
      cannot produce infinities in the adjoint.
    """
    E = p[0]
    eps_L = p[1]  # maximum transformation strain
    s_AM_s = p[2]  # forward transformation start stress
    s_AM_f = p[3]  # forward transformation finish stress
    s_MA_s = p[4]  # reverse transformation start stress
    s_MA_f = p[5]  # reverse transformation finish stress
    xi_n = committed[0]  # martensite fraction at the last converged step
    s_n = committed[1]  # |stress| at the last converged step

    zero = strain.dtype(0.0)
    one = strain.dtype(1.0)
    sign = wp.where(strain < zero, -one, one)
    e = wp.abs(strain)

    # Elastic trial with the martensite fraction frozen.
    s_trial = E * (e - eps_L * xi_n)

    # Forward transformation (austenite to martensite). s_0 is where the branch
    # becomes active in this step: the previous stress if already on the
    # plateau, otherwise s_AM_s.
    forward = (s_trial > s_n) and (s_trial > s_AM_s) and (xi_n < one)
    gap_f = s_AM_f - wp.max(s_n, s_AM_s)
    A = (one - xi_n) / wp.where(gap_f > zero, gap_f, one)
    den_f = one + A * E * eps_L
    xi_f = wp.where(gap_f > zero, (one - A * (s_AM_f - E * e)) / den_f, one)
    # Inside the plateau the fraction moves and the tangent is reduced. Once
    # the fraction reaches 1 the material is fully martensitic and elastic.
    moving_f = (xi_f < one) and (xi_f > xi_n)
    xi_forward = wp.where(xi_f >= one, one, wp.where(xi_f > xi_n, xi_f, xi_n))
    tangent_forward = wp.where(moving_f, E / den_f, E)

    # Reverse transformation (martensite to austenite).
    reverse = (s_trial < s_n) and (s_trial < s_MA_s) and (xi_n > zero)
    gap_r = wp.min(s_n, s_MA_s) - s_MA_f
    B = xi_n / wp.where(gap_r > zero, gap_r, one)
    den_r = one + B * E * eps_L
    xi_r = wp.where(gap_r > zero, B * (E * e - s_MA_f) / den_r, zero)
    moving_r = (xi_r > zero) and (xi_r < xi_n)
    xi_reverse = wp.where(xi_r <= zero, zero, wp.where(xi_r < xi_n, xi_r, xi_n))
    tangent_reverse = wp.where(moving_r, E / den_r, E)

    xi = wp.where(forward, xi_forward, wp.where(reverse, xi_reverse, xi_n))
    tangent = wp.where(forward, tangent_forward, wp.where(reverse, tangent_reverse, E))
    s = E * (e - eps_L * xi)
    # A new state vector is constructed; the committed one is never modified.
    return sign * s, tangent, wp.vector(xi, s)


NITINOL_SUPERELASTIC_UNIAXIAL = UniaxialModel(
    name="nitinol_superelastic_uniaxial",
    param_names=("E", "eps_L", "s_AM_s", "s_AM_f", "s_MA_s", "s_MA_f"),
    state_names=("martensite_fraction", "abs_stress"),
    update=superelastic_update,
    initial_state=(0.0, 0.0),
    source="Draft after Auricchio, Taylor and Lubliner, CMAME 146 (1997) 281-312; not yet verified against the paper",
)
