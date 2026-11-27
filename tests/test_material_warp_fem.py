"""The material functions used directly inside warp.fem integrands.

warp.fem (https://nvidia.github.io/warp/modules/fem.html) already provides
hexahedral and tetrahedral meshes, Lagrange spaces, quadrature, sparse assembly
and boundary-condition projection, in single or double precision. A FEBio-Warp
material is a set of plain Warp functions, so it can be called from a warp.fem
integrand without any adapter.

This test assembles the residual and the tangent of a small hexahedral grid in
double precision and checks the tangent against finite differences of the
residual. warp.fem works with gradients in the reference configuration, so the
forms here use the first Piola stress and its directional derivative, which
every model carries alongside the Cauchy form:

    residual:  integral of  P(F) : grad v
    tangent:   integral of  dP[F][grad du] : grad v
"""

import os
import unittest

import numpy as np
import warp as wp
import warp.fem as fem

from warp_febio.materials import model as model_api
from warp_febio.materials import models  # noqa: F401

# WARP_FEBIO_TEST_DEVICE selects the device ("cpu" by default, "cuda:0" for a
# GPU). It becomes Warp's current device while this module's tests run, so
# calls that do not name a device follow it.
DEVICE = os.environ.get("WARP_FEBIO_TEST_DEVICE", "cpu")
_device_scope = wp.ScopedDevice(DEVICE)


def setUpModule():
    _device_scope.__enter__()


def tearDownModule():
    _device_scope.__exit__(None, None, None)


AXES = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]])
CASES = {
    "neo_hookean": (dict(E=1000.0, v=0.3), {}),
    "hgo_unconstrained": (
        dict(c=200.0, k1=13480.0, k2=1.06, kappa=0.1, gamma=35.0, k=1500.0),
        {"mat_axis": AXES.reshape(9)},
    ),
}


def make_forms(model):
    """Residual and tangent integrands for one model."""
    pvec, gvec = model.param_vector(), model.geometry_vector()
    pk1, tangent_action = model.pk1, model.tangent_action

    @fem.integrand
    def residual_form(s: fem.Sample, u: fem.Field, v: fem.Field, g: gvec, p: pvec):
        F = wp.identity(n=3, dtype=wp.float64) + fem.grad(u, s)
        return wp.ddot(pk1(F, g, p), fem.grad(v, s))

    @fem.integrand
    def tangent_form(s: fem.Sample, u: fem.Field, du: fem.Field, v: fem.Field, g: gvec, p: pvec):
        F = wp.identity(n=3, dtype=wp.float64) + fem.grad(u, s)
        return wp.ddot(tangent_action(F, fem.grad(du, s), g, p), fem.grad(v, s))

    return residual_form, tangent_form


def dense(matrix):
    """Dense copy of a block sparse matrix, for a small test problem."""
    size = matrix.block_shape[0]
    offsets = matrix.offsets.numpy()
    columns = matrix.columns.numpy()
    values = matrix.values.numpy()
    out = np.zeros((matrix.nrow * size, matrix.ncol * size))
    for row in range(matrix.nrow):
        for k in range(offsets[row], offsets[row + 1]):
            c = columns[k]
            out[row * size : (row + 1) * size, c * size : (c + 1) * size] += values[k]
    return out


class WarpFemTests(unittest.TestCase):
    def test_materials_assemble_through_warp_fem(self):
        geometry = fem.Grid3D(res=wp.vec3i(2), scalar_type=wp.float64)
        space = fem.make_polynomial_space(geometry, degree=1, dtype=wp.vec3d)
        domain = fem.Cells(geometry)
        test = fem.make_test(space, domain=domain)
        trial = fem.make_trial(space, domain=domain)
        u = space.make_field()
        nodes = u.dof_values.shape[0]
        displacement = 0.03 * np.random.default_rng(0).normal(size=(nodes, 3))

        for name, (params, geometry_fields) in CASES.items():
            model = model_api.get(name)
            residual_form, tangent_form = make_forms(model)
            g = model.geometry_vector()(*model.pack_geometry(1, **geometry_fields)[0])
            p = model.pack(**params)

            def residual(values):
                u.dof_values = wp.array(values, dtype=wp.vec3d)
                out = fem.integrate(
                    residual_form,
                    fields={"u": u, "v": test},
                    values={"g": g, "p": p},
                    output_dtype=wp.vec3d,
                )
                return out.numpy().reshape(-1)

            with self.subTest(material=name):
                u.dof_values = wp.array(displacement, dtype=wp.vec3d)
                K = dense(
                    fem.integrate(
                        tangent_form,
                        fields={"u": u, "du": trial, "v": test},
                        values={"g": g, "p": p},
                        output_dtype=wp.float64,
                    )
                )
                self.assertEqual(K.dtype, np.float64)
                h = 1.0e-6
                numeric = np.zeros_like(K)
                for column in range(3 * nodes):
                    step = np.zeros(3 * nodes)
                    step[column] = h
                    step = step.reshape(nodes, 3)
                    numeric[:, column] = (residual(displacement + step) - residual(displacement - step)) / (2 * h)
                scale = np.abs(numeric).max()
                self.assertLess(np.abs(K - numeric).max() / scale, 1e-8)
                self.assertLess(np.abs(K - K.T).max() / scale, 1e-12)


if __name__ == "__main__":
    unittest.main()
