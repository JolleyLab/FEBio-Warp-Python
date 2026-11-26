"""Symmetric fourth-order tensors as 6x6 matrices, in FEBio's tens4ds order.

Index pairs: 0 -> 11, 1 -> 22, 2 -> 33, 3 -> 12, 4 -> 23, 5 -> 13
(https://github.com/febiosoftware/FEBio/blob/develop/FECore/tens4d.h).
Entry [a, b] is the tensor component c_ijkl itself for pairs a = (i,j),
b = (k,l). No factors of two are folded in for shear.

The helpers below are FEBio's fourth-order products written for this storage,
so a model's tangent can be transcribed term by term:

    add_dyad1s(c, a, w)      c += w * dyad1s(a)        (a (x) a)_ijkl = a_ij a_kl
    add_identity4s(c, w)     c += w * dyad4s(I)        the symmetric identity
    add_identity_dyad(c, w)  c += w * dyad1s(I)        I (x) I
    add_dyad1s_pair(c, a, b, w)  c += w * dyad1s(a, b)   a (x) b + b (x) a
"""

from typing import Any

import warp as wp

VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))

_vec6i = wp.types.vector(length=6, dtype=wp.int32)
VI = wp.constant(_vec6i(0, 1, 2, 0, 1, 0))
VJ = wp.constant(_vec6i(0, 1, 2, 1, 2, 2))


@wp.func
def zeros66(like: Any):
    """A 6x6 of zeros with the scalar type of `like` (a 3x3 matrix)."""
    return wp.matrix(shape=(6, 6), dtype=like.dtype)


@wp.func
def add_dyad1s(c: Any, a: Any, w: Any):
    for r in range(6):
        ar = a[VI[r], VJ[r]]
        for q in range(6):
            c[r, q] = c[r, q] + w * ar * a[VI[q], VJ[q]]
    return c


@wp.func
def add_identity4s(c: Any, w: Any):
    # dyad4s(I)_ijkl = (d_ik d_jl + d_il d_jk) / 2: 1 on the normal diagonal,
    # 1/2 on the shear diagonal.
    half = w * w.dtype(0.5)
    for r in range(3):
        c[r, r] = c[r, r] + w
        c[r + 3, r + 3] = c[r + 3, r + 3] + half
    return c


@wp.func
def add_identity_dyad(c: Any, w: Any):
    for r in range(3):
        for q in range(3):
            c[r, q] = c[r, q] + w
    return c


@wp.func
def add_dyad1s_pair(c: Any, a: Any, b: Any, w: Any):
    """c += w (a (x) b + b (x) a), FEBio's dyad1s(a, b) for symmetric a and b."""
    for r in range(6):
        ar = a[VI[r], VJ[r]]
        br = b[VI[r], VJ[r]]
        for q in range(6):
            c[r, q] = c[r, q] + w * (ar * b[VI[q], VJ[q]] + br * a[VI[q], VJ[q]])
    return c


@wp.func
def contract(c: Any, d: Any):
    """c : d for symmetric d, returned as a symmetric 3x3."""
    out = wp.matrix(shape=(3, 3), dtype=d.dtype)
    two = d.dtype(2.0)
    for r in range(6):
        value = d.dtype(0.0)
        for q in range(3):
            value = value + c[r, q] * d[q, q]
        for q in range(3, 6):
            value = value + two * c[r, q] * d[VI[q], VJ[q]]
        out[VI[r], VJ[r]] = value
        out[VJ[r], VI[r]] = value
    return out
