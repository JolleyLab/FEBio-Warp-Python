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
    truss_stretch       truss part name -> (E,) FEBio's "truss stretch"
    axial_force         truss part name -> (E,) axial force, tension positive.
                        Not a FEBio plot variable; it is the chord tension
    discrete            discrete set name -> {FEBio plot variable name: array}
                        for the springs of that set
    """

    time: float
    displacement: np.ndarray
    shell_displacement: np.ndarray | None = None
    stress: dict = field(default_factory=dict)
    stress_points: dict = field(default_factory=dict)
    iterations: int = 0
    #: FEBio's nodal reaction forces (N, 3); set by the run path template.
    reaction_force: np.ndarray | None = None
    truss_stretch: dict = field(default_factory=dict)
    axial_force: dict = field(default_factory=dict)
    discrete: dict = field(default_factory=dict)


@dataclass
class Results:
    states: list = field(default_factory=list)
    software: str = "FEBio-Warp"

    @property
    def times(self):
        return np.array([s.time for s in self.states])
