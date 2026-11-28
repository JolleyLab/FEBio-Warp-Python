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
    """Piecewise load curve through (time, value) points, as FEBio evaluates it.

    Follows PointCurve::value
    (https://github.com/febiosoftware/FEBio/blob/develop/FECore/PointCurve.cpp)
    for the two interpolation modes the applications use, with the CONSTANT
    extension: outside the points the end values are held.

    interpolate "linear": straight lines between points.
    interpolate "step":   the value of the first point whose time is greater
        than t. The first point's value holds up to and including its time;
        after that the curve already has the value of the next point, and at an
        interior point it has the value of the point after it. So a step curve
        through (0.998, 1.0) and (1.0, 0.75) is 1.0 up to and including
        t = 0.998 and 0.75 after it. This is the reverse of holding each value
        until the next point. Confirmed by FEBio 4.13 runs: a STEP curve through
        (0.25, 1), (0.5, 0.6), (0.75, 0.3) gave 1 at t = 0.25, 0.6 at t = 0.375
        and 0.3 at t = 0.5.
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
        if time <= self.times[0]:
            return float(self.values[0])
        if time >= self.times[-1]:
            return float(self.values[-1])
        # PointCurve's binarySearch: the first point whose time is greater
        # than `time`.
        index = int(np.searchsorted(self.times, time, side="right"))
        return float(self.values[index])
