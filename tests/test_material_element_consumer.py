"""The material layer as an element would use it.

A single eight-node hexahedron in an updated Lagrangian formulation is written
here in NumPy, test-only. It takes from the material layer exactly what a
FEBio-Warp solid element will take: Cauchy stress and the 6x6 spatial tangent
at each integration point. Its stiffness matrix, built from those, must equal
the derivative of its internal force vector, obtained by finite differences.

This catches errors that material-point checks cannot: a wrong index order or
shear factor in the 6x6, or a tangent that is not the one the updated
Lagrangian stiffness needs (FEBio: FEElasticSolidDomain::ElementMaterialStiffness
and ElementGeometricalStiffness,
https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticSolidDomain.cpp).
It is not a comparison with a FEBio run.
"""

import unittest

import numpy as np

from warp_febio.materials import model as model_api
from warp_febio.materials import models  # noqa: F401
from warp_febio.materials import status as material_status
from warp_febio.materials.compose import solid_mixture
from warp_febio.materials.exchange import evaluate_exchange
from warp_febio.materials.voigt import VOIGT

# Reference hexahedron: unit cube corners in FEBio hex8 node order.
CORNERS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)
GAUSS = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float) / np.sqrt(3.0)

# Fourth-order tensor from its 6x6 storage: index pair (i, j) -> row.
PAIR = np.zeros((3, 3), dtype=int)
for row, (i, j) in enumerate(VOIGT):
    PAIR[i, j] = PAIR[j, i] = row


def shape_gradients(xi):
    """d(N_a)/d(xi) for the trilinear hexahedron, shape (8, 3)."""
    out = np.zeros((8, 3))
    for a, corner in enumerate(CORNERS):
        factors = 1.0 + corner * xi
        for k in range(3):
            others = np.prod(np.delete(factors, k))
            out[a, k] = 0.125 * corner[k] * others
    return out


def element(model, params, geometry, X, u):
    """Internal force (24,) and stiffness (24, 24) of one hexahedron."""
    grads_ref, volumes, F = [], [], []
    for xi in GAUSS:
        dN = shape_gradients(xi)
        J0 = X.T @ dN  # d(X)/d(xi)
        dNdX = dN @ np.linalg.inv(J0)
        grads_ref.append(dNdX)
        volumes.append(np.linalg.det(J0))  # Gauss weight is 1
        F.append(np.eye(3) + u.T @ dNdX)
    F = np.array(F)
    out = evaluate_exchange(model, F, params, geometry=geometry)
    material_status.raise_on_failure(out["status"])
    force = np.zeros((8, 3))
    K = np.zeros((8, 3, 8, 3))
    for q in range(8):
        dNdx = grads_ref[q] @ np.linalg.inv(F[q])  # spatial gradients
        dv = np.linalg.det(F[q]) * volumes[q]  # current volume
        sigma = out["cauchy"][q]
        c6 = out["tangent"][q]
        c4 = c6[PAIR[:, :, None, None], PAIR[None, None, :, :]]  # c_ijkl
        force += dNdx @ sigma * dv
        # material stiffness: dN_a/dx_j c_ijkl dN_b/dx_l
        K += np.einsum("aj,ijkl,bl->aibk", dNdx, c4, dNdx) * dv
        # geometric (initial stress) stiffness: delta_ik dN_a/dx_j sigma_jl dN_b/dx_l
        K += np.einsum("aj,jl,bl,ik->aibk", dNdx, sigma, dNdx, np.eye(3)) * dv
    return force.reshape(24), K.reshape(24, 24)


class ElementConsumerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        # A mildly distorted element and a finite, non-uniform displacement.
        self.X = 0.5 * CORNERS + 0.05 * rng.normal(size=(8, 3))
        self.u = 0.08 * rng.normal(size=(8, 3))
        axes = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]])
        neo = model_api.get("neo_hookean")
        hgo = model_api.get("hgo_unconstrained")
        self.cases = {
            "neo-Hookean": (neo, dict(E=1000.0, v=0.3), None),
            "isotropic elastic": (
                model_api.get("isotropic_elastic"),
                dict(E=1000.0, v=0.3),
                None,
            ),
            "isotropic Lee-Sacks": (
                model_api.get("lee_sacks_isotropic"),
                dict(c0=200.0, c1=2968.4, c2=0.2661, k=5000.0),
                None,
            ),
            "HGO unconstrained": (
                hgo,
                dict(c=200.0, k1=13480.0, k2=1.06, kappa=0.1, gamma=35.0, k=1500.0),
                {"mat_axis": axes.reshape(9)},
            ),
            "mixture of neo-Hookean and HGO": (
                solid_mixture("test_element_mix", [("ground", neo), ("wall", hgo)]),
                {
                    "ground.E": 500.0,
                    "ground.v": 0.3,
                    "wall.c": 200.0,
                    "wall.k1": 13480.0,
                    "wall.k2": 1.06,
                    "wall.kappa": 0.1,
                    "wall.gamma": 35.0,
                    "wall.k": 1500.0,
                },
                {"wall.mat_axis": axes.reshape(9)},
            ),
        }

    def test_stiffness_is_the_derivative_of_internal_force(self):
        h = 1.0e-6
        for label, (model, params, geometry) in self.cases.items():
            with self.subTest(material=label):
                force, K = element(model, params, geometry, self.X, self.u)
                numeric = np.zeros((24, 24))
                for column in range(24):
                    step = np.zeros(24)
                    step[column] = h
                    up, _ = element(model, params, geometry, self.X, self.u + step.reshape(8, 3))
                    down, _ = element(model, params, geometry, self.X, self.u - step.reshape(8, 3))
                    numeric[:, column] = (up - down) / (2 * h)
                scale = np.abs(numeric).max()
                self.assertLess(np.abs(K - numeric).max() / scale, 1e-7)
                # A hyperelastic element stiffness is symmetric.
                self.assertLess(np.abs(K - K.T).max() / scale, 1e-12)
                # Internal forces are self-equilibrated: no net force.
                self.assertLess(
                    np.abs(force.reshape(8, 3).sum(axis=0)).max() / np.abs(force).max(),
                    1e-12,
                )

    def test_rigid_motion_produces_no_force(self):
        angle = 0.7
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        u = self.X @ R.T + np.array([0.3, -0.2, 0.1]) - self.X
        for label, (model, params, geometry) in self.cases.items():
            with self.subTest(material=label):
                force, K = element(model, params, geometry, self.X, u)
                self.assertLess(np.abs(force).max() / np.abs(K).max(), 1e-12)


if __name__ == "__main__":
    unittest.main()
