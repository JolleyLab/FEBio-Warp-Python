"""Results of a run, on the host, keyed by FEBio's plot-variable names.

The output writers (XPLT, VTKHDF, VTK) take a Model and a Results; they do not
reach into the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StateResult:
    """One output state.

    displacement        (N, 3) nodal displacement; for shells, the front face
    shell_displacement  (N, 3) back-face displacement (sx, sy, sz), or None
    stress              part name -> (E, 6) element-average Cauchy stress in
                        FEBio's order xx, yy, zz, xy, yz, xz. The average is
                        the plain mean over integration points, as FEBio's
                        plot variable "stress" (FEPlotElementStress).
    stress_points       part name -> (E, Q, 3, 3) stress at integration points
    """

    time: float
    displacement: np.ndarray
    shell_displacement: np.ndarray | None = None
    stress: dict = field(default_factory=dict)
    stress_points: dict = field(default_factory=dict)
    iterations: int = 0
    #: FEBio's nodal reaction forces (N, 3); set by the run path template.
    reaction_force: np.ndarray | None = None


@dataclass
class Results:
    states: list = field(default_factory=list)
    software: str = "FEBio-Warp"

    @property
    def times(self):
        return np.array([s.time for s in self.states])
