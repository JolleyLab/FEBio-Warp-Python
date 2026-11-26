"""Finite-difference check of adjoint gradients, for any system.

TEMPLATE (run path draft, November 2026).

Every term lists its differentiable parameter arrays (`Term.parameters()`).
This check walks all of them, so a new term is covered by the same call
with no change here: perturb each parameter, run the model again, and
compare the central difference with the adjoint gradient.

It runs the whole model twice per parameter, so it is for small cases. The
standing case is references/single_element/hex8_pressure.feb.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.steps import run
from .adjoint import parameter_gradient


@dataclass
class GradientRow:
    parameter: str  # key of System.parameters()
    index: tuple  # position within that array
    value: float
    adjoint: float
    finite_difference: float

    @property
    def relative_error(self) -> float:
        scale = max(abs(self.finite_difference), abs(self.adjoint), 1e-300)
        return abs(self.adjoint - self.finite_difference) / scale


def check_parameter_gradients(
    system, output, rel_step: float = 1e-5, linear_solver: str = "direct"
) -> list:
    """Adjoint against central differences for every parameter of `system`.

    output(u) -> (J, dJ_du), with u and dJ_du as (n_entries, 3) host arrays:
    a scalar output of the final state and its derivative.
    Returns one GradientRow per scalar parameter.
    """

    def final(_=None):
        _, states = run(system, linear_solver=linear_solver, keep_states=True)
        return states[-1]

    state = final()
    _, dJ_du = output(state.u.numpy())
    gradients, _ = parameter_gradient(system, state, dJ_du, linear_solver)

    rows = []
    for key, array in system.parameters().items():
        base = array.numpy().copy()
        for index in np.ndindex(base.shape):
            h = rel_step * max(abs(base[index]), 1e-3)
            values = []
            for sign in (1.0, -1.0):
                trial = base.copy()
                trial[index] += sign * h
                array.assign(trial)
                values.append(output(final().u.numpy())[0])
            array.assign(base)
            rows.append(
                GradientRow(
                    key,
                    index,
                    float(base[index]),
                    float(gradients[key][index]),
                    (values[0] - values[1]) / (2 * h),
                )
            )
    return rows
