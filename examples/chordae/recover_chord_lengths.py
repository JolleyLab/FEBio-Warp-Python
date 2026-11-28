"""Recover chord rest lengths, and so chord tensions, from a target shape.

A small inverse problem of the kind FEBio FINESSE addresses (Laurence et al.,
"FEBio FINESSE: An Open-Source Finite Element Simulation Approach to Estimate
In Vivo Heart Valve Strains Using Shape Enforcement", Ann Biomed Eng 2025,
arXiv 2407.09629). There, the loaded shape of a valve is known from an image
and the chordae are not, so they are emulated by a force on the free edge or
by prescribing its motion. Here the chords are in the model, and the question
is put the other way round: which chord lengths make the loaded shape match
the target?

The model is references/small_meshes/shell_patch_truss_chords.feb: nine quad4
shells clamped on one edge, the opposite edge held by four truss chords.

    1. A target shape is made by running the model with the four rest lengths
       changed by a few percent. The tensions of that run are kept aside.
    2. Starting from the unchanged lengths, the rest lengths are fitted so
       that the nodal positions match the target. The gradient of the misfit
       with respect to the rest lengths comes from one adjoint solve per
       evaluation (sensitivity/adjoint.py).
    3. The recovered lengths and the chord tensions are compared with those
       of step 1.

The target here is the model's own result, so the fit can be exact. With an
image-derived target the misfit would be a distance to a surface, as in
FINESSE, and would not reach zero. That, and whether lengths are identifiable
on a real valve, is not examined here.

    uv run examples/chordae/recover_chord_lengths.py [device]
"""

import pathlib
import sys

import numpy as np
import warp as wp
from scipy.optimize import minimize

from warp_febio.analysis.steps import build_system, run
from warp_febio.io import read_feb
from warp_febio.sensitivity.adjoint import parameter_gradient

ROOT = pathlib.Path(__file__).resolve().parents[2]
INPUT = ROOT / "references" / "small_meshes" / "shell_patch_truss_chords.feb"
# Relative change of the four rest lengths that defines the target.
CHANGE = np.array([0.03, -0.02, 0.05, -0.03])


def recover(device=None, log=print):
    """Returns (true lengths, fitted lengths, true tensions, fitted tensions, evaluations)."""
    wp.config.quiet = True
    system = build_system(read_feb(INPUT), device=device)
    chords = next(t for t in system.terms if getattr(t, "element_type", "") == "line2")
    geometric = chords.rest_length.numpy().copy()

    def solve(lengths):
        chords.rest_length.assign(lengths)
        _, states = run(system, keep_states=True)
        return states[-1]

    # 1. The target, and what the fit should find.
    true_lengths = geometric * (1.0 + CHANGE)
    state = solve(true_lengths)
    target = state.u.numpy().copy()
    true_tension = chords.axial_force(state).numpy().copy()

    # 2. Fit. The unknowns are the lengths relative to the geometric ones.
    evaluations = [0]

    def misfit(x):
        state = solve(geometric * x)
        d = state.u.numpy() - target
        gradients, _ = parameter_gradient(system, state, d)
        evaluations[0] += 1
        return 0.5 * float((d * d).sum()), gradients["Chords.rest_length"] * geometric

    scale = 0.5 * float(((solve(geometric).u.numpy() - target) ** 2).sum())
    result = minimize(
        lambda x: tuple(v / scale for v in misfit(x)),
        np.ones(4),
        jac=True,
        method="L-BFGS-B",
        # Lengths within 10 percent of the geometric ones. Without bounds the
        # first trial step of the optimizer leaves the range in which the
        # forward solve converges; the run path has no line search yet.
        bounds=[(0.9, 1.1)] * 4,
        options=dict(ftol=1e-22, gtol=1e-10, maxiter=200),
    )
    fitted_lengths = geometric * result.x
    fitted_tension = chords.axial_force(solve(fitted_lengths)).numpy().copy()
    chords.rest_length.assign(geometric)

    log(f"misfit, relative to the start: {result.fun:.2e} after {evaluations[0]} evaluations")
    log("chord   length change, target   fitted      tension, target   fitted")
    for i in range(4):
        log(
            f"  {i + 1}       {CHANGE[i]:+.4f}              {result.x[i] - 1.0:+.4f}"
            f"      {true_tension[i]:.6e}      {fitted_tension[i]:.6e}"
        )
    return true_lengths, fitted_lengths, true_tension, fitted_tension, evaluations[0]


if __name__ == "__main__":
    recover(sys.argv[1] if len(sys.argv) > 1 else None)
