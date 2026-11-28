"""NumPy transcriptions of FEBio closed forms. Test-only code.

Each function returns (Cauchy stress 3x3, spatial tangent 6x6 in tens4ds order)
for one deformation gradient, written independently of the Warp models from the
FEBio source named in its docstring. FEBio's fourth-order products are

    dyad1s(a)_ijkl = a_ij a_kl
    dyad4s(a)_ijkl = (a_ik a_jl + a_il a_jk) / 2

(https://github.com/febiosoftware/FEBio/blob/develop/FECore/tens4d.h)
"""

import numpy as np

VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
_I = np.eye(3)


def _voigt(c4):
    return np.array([[c4[i, j, k, l] for (k, l) in VOIGT] for (i, j) in VOIGT])


def _dyad1s(a):
    return np.einsum("ij,kl->ijkl", a, a)


def _dyad4s(a):
    return 0.5 * (np.einsum("ik,jl->ijkl", a, a) + np.einsum("il,jk->ijkl", a, a))


def _lame(E, v):
    return v * E / ((1 + v) * (1 - 2 * v)), 0.5 * E / (1 + v)


def lee_sacks_isotropic(F, c0, c1, c2, k):
    """FEIsotropicLeeSacks::Stress and ::Tangent with tangent_scale = 1."""
    b = F @ F.T
    J = np.linalg.det(F)
    s = np.trace(b) - 3.0
    eQ = np.exp(c2 * s * s)
    lnJ = np.log(J)
    p = k * lnJ / J
    pJ = k * (1.0 - lnJ) / J**2
    sigma = p * _I + (c0 / J) * (b - _I) + (2.0 * c1 * c2 * s * eQ / J) * b
    cw = 4.0 * c1 * c2 * eQ * (1.0 + 2.0 * c2 * s * s) / J
    c4 = cw * _dyad1s(b) + (p + J * pJ) * _dyad1s(_I) + (2.0 * c0 / J - 2.0 * p) * _dyad4s(_I)
    return sigma, _voigt(c4)


def neo_hookean(F, E, v):
    """FENeoHookean::Stress and ::Tangent."""
    lam, mu = _lame(E, v)
    b = F @ F.T
    J = np.linalg.det(F)
    lnJ = np.log(J)
    sigma = (mu / J) * (b - _I) + (lam * lnJ / J) * _I
    c4 = (lam / J) * _dyad1s(_I) + 2.0 * ((mu - lam * lnJ) / J) * _dyad4s(_I)
    return sigma, _voigt(c4)


def isotropic_elastic(F, E, v):
    """FEIsotropicElastic::Stress and ::Tangent."""
    lam, mu = _lame(E, v)
    b = F @ F.T
    J = np.linalg.det(F)
    trE = 0.5 * (np.trace(b) - 3.0)
    sigma = (b * (lam * trE - mu) + mu * (b @ b)) / J
    c4 = (lam / J) * _dyad1s(b) + (2.0 * mu / J) * _dyad4s(b)
    return sigma, _voigt(c4)


def hgo_unconstrained(F, Q, c, k1, k2, kappa, gamma, k):
    """FEHolzapfelUnconstrained::Stress and ::Tangent. Q holds the material axes;
    its first two columns span the fiber plane."""
    b = F @ F.T
    C = F.T @ F
    J = np.linalg.det(F)
    I1 = np.trace(C)
    angle = gamma * np.pi / 180.0
    n1, n2 = Q[:, 0], Q[:, 1]
    s = c * (b - _I) + _I * ((J * J - 1.0) * (k * 0.5))
    c4 = _dyad4s(_I) * (2.0 * c - (J * J - 1.0) * k) + _dyad1s(_I) * (J * J * k)
    for sign in (1.0, -1.0):
        ar = n1 * np.cos(angle) + sign * n2 * np.sin(angle)
        a = F @ ar
        E = kappa * (I1 - 3.0) + (1.0 - 3.0 * kappa) * (ar @ C @ ar - 1.0)
        if E >= 0:
            h = kappa * b + (1.0 - 3.0 * kappa) * np.outer(a, a)
            s = s + h * (2.0 * k1 * E * np.exp(k2 * E * E))
            c4 = c4 + _dyad1s(h) * (4.0 * k1 * (1.0 + 2.0 * k2 * E * E) * np.exp(k2 * E * E))
    return s / J, _voigt(c4 / J)


CLOSED_FORMS = {
    "lee_sacks_isotropic": lee_sacks_isotropic,
    "neo_hookean": neo_hookean,
    "isotropic_elastic": isotropic_elastic,
}


# ---------------------------------------------------------------------------
# Discrete springs (FEBioMech/FENonlinearSpring.cpp, FECore/PointCurve.cpp)
# ---------------------------------------------------------------------------


def point_curve_linear(t, points, extend="extrapolate"):
    """PointCurve::value for LINEAR interpolation, CONSTANT or EXTRAPOLATE extension."""
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    last = len(x) - 1
    if t == x[0]:
        return y[0]
    if t == x[last]:
        return y[last]
    if t < x[0] or t > x[last]:  # PointCurve::ExtendValue
        if extend == "constant":
            return y[0] if t < x[0] else y[last]
        a, b = (0, 1) if t < x[0] else (last - 1, last)
        return y[a] + (y[b] - y[a]) * (t - x[a]) / (x[b] - x[a])
    n = int(np.searchsorted(x, t, side="right"))  # binarySearch in PointCurve.cpp
    return y[n - 1] + (y[n] - y[n - 1]) * (t - x[n - 1]) / (x[n] - x[n - 1])


def point_curve_linear_slope(t, points, extend="extrapolate"):
    """Exact slope of the active segment, with the segment to the right at a knot.

    FEBio's PointCurve::derive uses a finite difference of value() for linear
    curves; this is the slope that difference approximates.
    """
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    last = len(x) - 1
    if extend == "constant" and (t < x[0] or t > x[last]):
        return 0.0
    n = int(np.clip(np.searchsorted(x, t, side="right"), 1, last))
    return (y[n] - y[n - 1]) / (x[n] - x[n - 1])


def febio_point_curve_derive(t, points, extend="extrapolate"):
    """PointCurve::derive for a LINEAR curve: differences of value(), as FEBio does."""
    pts = np.asarray(points, dtype=float)
    span = pts[-1, 0] - pts[0, 0]
    dt = span * 1e-9
    value = lambda s: point_curve_linear(s, points, extend)
    if t >= pts[-1, 0]:
        return (value(t - 2 * dt) - 4 * value(t - dt) + 3 * value(t)) / (2 * dt)
    if t < pts[0, 0]:
        return (-value(t + 2 * dt) + 4 * value(t + dt) - 3 * value(t)) / (2 * dt)
    return (value(t + dt) - value(t - dt)) / (2 * dt)


def nonlinear_spring(r0, rt, scale, points, measure="elongation", extend="extrapolate"):
    """Force vector, 3x3 stiffness and deformation measure of FENonlinearSpringMaterial.

    The stiffness uses the exact slope of the point function (see
    point_curve_linear_slope).
    """
    r0 = np.asarray(r0, dtype=float)
    rt = np.asarray(rt, dtype=float)
    L0 = np.linalg.norm(r0)
    Lt = np.linalg.norm(rt)
    DL = Lt - L0
    s, ds = {
        "elongation": (DL, 1.0),
        "strain": (DL / L0, 1.0 / L0),
        "stretch": (Lt / L0, 1.0 / L0),
    }[measure]
    F = scale * point_curve_linear(s, points, extend)
    E = scale * point_curve_linear_slope(s, points, extend) * ds
    if Lt == 0.0:
        F, Lt, e = 0.0, 1.0, np.ones(3)
    else:
        e = rt / Lt
    K = (E - F / Lt) * np.outer(e, e) + (F / Lt) * np.eye(3)
    return F * e, K, s
