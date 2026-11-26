"""Building the geometry vector from FEBio-style input.

Material axes and fiber directions are reference-configuration data. FEBio
evaluates them through "valuators" each time a material is called. Here they
are evaluated once, at setup, into per-point arrays that the kernels read.
"""

import numpy as np


def axes_from_vectors(a, d):
    """Material axes from FEBio's <mat_axis type="vector"> with vectors a and d.

    Follows FEMat3dVectorMap::Init
    (https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEMat3dValuator.cpp):
    the first axis is a; the third is a x d; the second is (a x d) x a. If a and
    d are nearly parallel, d is replaced by the x axis, or by the y axis if a is
    along x. Returns Q (3x3) whose columns are the three axes.
    """
    a = np.asarray(a, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    if not np.any(a) or not np.any(d):
        raise ValueError("mat_axis vectors must be non-zero")
    a = a / np.linalg.norm(a)
    d = d / np.linalg.norm(d)
    if abs(a @ d) > 0.999:
        d = np.array([1.0, 0.0, 0.0])
        if abs(a @ d) > 0.999:
            d = np.array([0.0, 1.0, 0.0])
    c = np.cross(a, d)
    b = np.cross(c, a)
    return np.column_stack(
        [a / np.linalg.norm(a), b / np.linalg.norm(b), c / np.linalg.norm(c)]
    )


def check_axes(Q, tolerance=1.0e-8):
    """Raise if Q (..., 3, 3) is not a right-handed orthonormal frame.

    Models assume it. Axes read from a mapped field or produced by an
    optimization are not guaranteed to be orthonormal, so this is the check to
    run before packing them into the geometry vector.
    """
    Q = np.asarray(Q, dtype=np.float64).reshape(-1, 3, 3)
    error = np.abs(Q.transpose(0, 2, 1) @ Q - np.eye(3)).max()
    if error > tolerance:
        raise ValueError(f"material axes are not orthonormal (error {error:.2e})")
    if np.any(np.linalg.det(Q) < 0.0):
        raise ValueError("material axes are left-handed")
