"""Time-dependent material parameters.

In FEBio any float parameter can be tied to a load curve with the `lc`
attribute; the curve value multiplies the parameter at each time step. Because
parameters here live in an array, the same thing is one host-side update of
that array per step. No kernel changes.

This is what the published transcatheter pulmonary valve simulations use for
the nitinol frame: an elastic law whose Young's modulus follows a step load
curve between the compression and release phases (Zelonis et al.,
arXiv 2507.06337, Section "Material models"). See models/nitinol.py.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoadCurve:
    """Piecewise load curve through (time, value) points.

    interpolate: "linear" or "step". A step curve holds each value until the
    next point is reached. Outside the
    points the end values are held. Check both against the FEBio load controller
    before relying on them for a FEBio comparison.
    """

    times: tuple
    values: tuple
    interpolate: str = "linear"

    def __post_init__(self):
        if len(self.times) != len(self.values) or len(self.times) == 0:
            raise ValueError("load curve needs matching, non-empty times and values")
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("load curve times must increase")
        if self.interpolate not in ("linear", "step"):
            raise ValueError("interpolate must be 'linear' or 'step'")

    def __call__(self, time):
        if self.interpolate == "linear":
            return float(np.interp(time, self.times, self.values))
        index = int(np.searchsorted(self.times, time, side="right")) - 1
        return float(self.values[max(index, 0)])
