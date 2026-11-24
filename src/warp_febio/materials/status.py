"""Failure reporting from material kernels.

FEBio reports a failed material point by throwing: an element with a
non-positive Jacobian raises NegativeJacobian, the domain collects these and
raises NegativeJacobianDetected, and the solver cuts the time step
(https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticSolidDomain.cpp).

A Warp kernel cannot throw, and one bad point must not stop or corrupt the
other points in the same launch. So every kernel that evaluates a material
writes one status code per point, and the host inspects the array after the
launch. The rule is:

1. The kernel checks J = det F before calling the material. If J <= 0 it writes
   NEGATIVE_JACOBIAN, writes zeros to its outputs, and does not call the
   material. Material functions may therefore assume J > 0.
2. After the material call, a result that is not finite (overflow of an
   exponential, for example) is reported as NON_FINITE and its outputs are
   zeroed, so NaN or infinity never reaches assembly.
3. The host calls `worst` (or `raise_on_failure`) on the status array. A solver
   treats any non-zero status the way FEBio treats its exception: reject the
   iterate and cut the step.

Codes are ordered by severity so the worst status of a launch is its maximum.
The check costs one determinant and one comparison per point, and it does not
enter differentiation: status is an integer output.
"""

import numpy as np

OK = 0
NON_FINITE = 1
NEGATIVE_JACOBIAN = 2

NAMES = {
    OK: "ok",
    NON_FINITE: "non-finite result",
    NEGATIVE_JACOBIAN: "negative Jacobian",
}


class MaterialFailure(RuntimeError):
    """Raised on the host when a launch reported failed material points."""


def worst(status):
    """Worst status code in an array (Warp or NumPy)."""
    values = status.numpy() if hasattr(status, "numpy") else np.asarray(status)
    return int(values.max()) if values.size else OK


def raise_on_failure(status, where="material evaluation"):
    """Host-side counterpart of FEBio's NegativeJacobianDetected."""
    values = status.numpy() if hasattr(status, "numpy") else np.asarray(status)
    if values.size == 0 or values.max() == OK:
        return
    counts = {NAMES[code]: int((values == code).sum()) for code in NAMES if code != OK}
    first = int(np.argmax(values != OK))
    detail = ", ".join(f"{n} at {c} point(s)" for n, c in counts.items() if c)
    raise MaterialFailure(f"{where}: {detail}; first at point {first}")
