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
    c4 = (
        cw * _dyad1s(b)
        + (p + J * pJ) * _dyad1s(_I)
        + (2.0 * c0 / J - 2.0 * p) * _dyad4s(_I)
    )
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
            c4 = c4 + _dyad1s(h) * (
                4.0 * k1 * (1.0 + 2.0 * k2 * E * E) * np.exp(k2 * E * E)
            )
    return s / J, _voigt(c4 / J)


CLOSED_FORMS = {
    "lee_sacks_isotropic": lee_sacks_isotropic,
    "neo_hookean": neo_hookean,
    "isotropic_elastic": isotropic_elastic,
}
