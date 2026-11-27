"""Eight-node trilinear brick: internal force, stiffness and stress.

One kernel serves two FEBio element types.

hex8 solid
    FEBio: FEElasticSolidDomain with the FEHex8G8 rule (2x2x2 Gauss).
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticSolidDomain.cpp

quad4 shell ("elastic-shell")
    FEBio: FEElasticShellDomain with the FEShellQuad4G8 rule.
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticShellDomain.cpp
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FESSIShellDomain.cpp
    A FEBio shell node carries a front-face displacement (x, y, z) and a
    back-face displacement (sx, sy, sz). The position inside the shell is
        x(r, s, t) = sum_i M_i(r, s) [ x_i - D_i (1 - t) / 2 ],
        D_i = d0_i + u_i - su_i,
    (FESSIShellDomain::CoBaseVectors), which is linear through the thickness
    between the back face (t = -1, at x_i - D_i) and the front face (t = +1).
    That is the trilinear brick whose nodes 0..3 are the back-face points and
    nodes 4..7 the front-face points. FEBio's shell gradients
        gradMu = (gradM (1 + t) + g3 M) / 2,  gradMd = (gradM (1 - t) - g3 M) / 2
    are the gradients of that brick's shape functions, and FEShellQuad4G8 uses
    the same eight points, in the same order, as FEHex8G8. So the shell is
    evaluated here as a brick; only the node and degree-of-freedom maps differ
    (compiled.py builds them).

Formulation (updated Lagrangian, as in FEBio):
    internal force   f_i   = int  sigma grad N_i  dv
    material part    K_ij  = int  grad N_i . c . grad N_j  dv      (vdotTdotv)
    geometric part   K_ij += int (grad N_i . sigma grad N_j) I dv

Degrees of freedom are ordered node by node: (node 0: x, y, z), (node 1: ...).
"""

import numpy as np
import warp as wp

from ..materials.status import NEGATIVE_JACOBIAN, NON_FINITE, OK

# Status codes written per element, shared with the material layer
# (materials/status.py). A kernel cannot throw; the host reads these after the
# launch and treats a failure as FEBio treats its NegativeJacobian exception.
# FEHex8G8 / FEShellQuad4G8 integration points, in FEBio's order.
# https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEElementTraits.cpp
_A = 1.0 / np.sqrt(3.0)
GAUSS_POINTS = _A * np.array(
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
    dtype=np.float64,
)
GAUSS_WEIGHTS = np.ones(8, dtype=np.float64)

# Natural coordinates of the eight nodes, FEBio hex8 order.
NODE_RST = np.array(
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
    dtype=np.float64,
)


def shape_derivatives(points=GAUSS_POINTS):
    """dN_i/d(r,s,t) at each point: array of shape (points, 8, 3)."""
    out = np.zeros((len(points), 8, 3))
    for q, (r, s, t) in enumerate(points):
        for i, (ri, si, ti) in enumerate(NODE_RST):
            out[q, i] = 0.125 * np.array(
                [
                    ri * (1 + si * s) * (1 + ti * t),
                    si * (1 + ri * r) * (1 + ti * t),
                    ti * (1 + ri * r) * (1 + si * s),
                ]
            )
    return out


@wp.func
def voigt(i: int, j: int):
    """Row of FEBio's tens4ds storage for the index pair (i, j):
    11 -> 0, 22 -> 1, 33 -> 2, 12 -> 3, 23 -> 4, 13 -> 5."""
    if i == j:
        return i
    s = i + j
    if s == 1:
        return 3
    if s == 3:
        return 4
    return 5


_KERNELS = {}


def make_brick_kernel(material):
    """Build the element kernel for one material.

    `material` is a BoundMaterial (materials/febio.py). Warp kernels cannot
    dispatch on a material type at run time, so the model's function is
    captured here and one kernel is generated per material model. A solid
    mixture is one model, so it is one kernel with no branching.
    """
    key = material.model.name
    if key in _KERNELS:
        return _KERNELS[key]
    stress_and_tangent = material.stress_and_tangent()
    pvec = material.model.param_vector(wp.float64)
    gvec = material.model.geometry_vector(wp.float64)

    def brick(
        X: wp.array2d(dtype=wp.vec3d),  # (E, 8) reference positions
        dofmap: wp.array2d(dtype=wp.int32),  # (E, 24) global degrees of freedom
        u: wp.array(dtype=wp.float64),  # global displacement vector
        dN: wp.array3d(dtype=wp.float64),  # (Q, 8, 3) shape derivatives
        gw: wp.array(dtype=wp.float64),  # (Q,) weights
        params: wp.array(dtype=pvec),  # one entry per domain for now
        geom: wp.array(dtype=gvec),  # (E,) material axes / fibers
        fe: wp.array2d(dtype=wp.float64),  # (E, 24) internal force
        ke: wp.array3d(dtype=wp.float64),  # (E, 24, 24) stiffness
        sigma: wp.array2d(dtype=wp.mat33d),  # (E, Q) Cauchy stress
        status: wp.array(dtype=wp.int32),  # (E,)
    ):
        e = wp.tid()
        nn = X.shape[1]
        nq = gw.shape[0]
        ndof = dofmap.shape[1]
        p = params[0]
        g = geom[e]
        for a in range(ndof):
            fe[e, a] = wp.float64(0.0)
            for b in range(ndof):
                ke[e, a, b] = wp.float64(0.0)
        status[e] = wp.static(OK)

        for q in range(nq):
            # Jacobians of the reference and current maps, J = sum x_i (x) dN_i.
            J0 = wp.mat33d()
            Jt = wp.mat33d()
            for i in range(nn):
                Xi = X[e, i]
                ui = wp.vec3d(
                    u[dofmap[e, 3 * i]],
                    u[dofmap[e, 3 * i + 1]],
                    u[dofmap[e, 3 * i + 2]],
                )
                dn = wp.vec3d(dN[q, i, 0], dN[q, i, 1], dN[q, i, 2])
                J0 = J0 + wp.outer(Xi, dn)
                Jt = Jt + wp.outer(Xi + ui, dn)
            detJ0 = wp.determinant(J0)
            detJt = wp.determinant(Jt)
            if detJ0 <= wp.float64(0.0) or detJt <= wp.float64(0.0):
                status[e] = wp.static(NEGATIVE_JACOBIAN)
                return
            F = Jt * wp.inverse(J0)
            JtiT = wp.transpose(wp.inverse(Jt))
            s, c = stress_and_tangent(F, g, p)
            if not (wp.isfinite(s) and wp.isfinite(c)):
                status[e] = wp.static(NON_FINITE)
                return
            sigma[e, q] = s
            w = detJt * gw[q]

            for i in range(nn):
                # Spatial gradient of N_i.
                gi = JtiT * wp.vec3d(dN[q, i, 0], dN[q, i, 1], dN[q, i, 2])
                fi = s * gi
                for a in range(3):
                    fe[e, 3 * i + a] = fe[e, 3 * i + a] + fi[a] * w
                for j in range(nn):
                    gj = JtiT * wp.vec3d(dN[q, j, 0], dN[q, j, 1], dN[q, j, 2])
                    # Geometric (initial stress) part.
                    kg = wp.dot(gi, s * gj) * w
                    for a in range(3):
                        for b in range(3):
                            # Material part: (gi . c . gj)_ab = gi_m c_ambn gj_n,
                            # FEBio's vdotTdotv.
                            val = wp.float64(0.0)
                            for m in range(3):
                                for n in range(3):
                                    val = val + gi[m] * c[voigt(a, m), voigt(b, n)] * gj[n]
                            ke[e, 3 * i + a, 3 * j + b] = ke[e, 3 * i + a, 3 * j + b] + val * w
                        ke[e, 3 * i + a, 3 * j + a] = ke[e, 3 * i + a, 3 * j + a] + kg

    kernel = wp.kernel(brick, module="unique")
    _KERNELS[key] = kernel
    return kernel


@wp.kernel
def scatter_internal_force(
    fe: wp.array2d(dtype=wp.float64),
    dofmap: wp.array2d(dtype=wp.int32),
    residual: wp.array(dtype=wp.float64),
):
    """residual -= internal force (FEBio's sign: R = F_external - F_internal)."""
    e, a = wp.tid()
    wp.atomic_sub(residual, dofmap[e, a], fe[e, a])


@wp.kernel
def stiffness_triplets(
    ke: wp.array3d(dtype=wp.float64),
    dofmap: wp.array2d(dtype=wp.int32),
    offset: int,
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    vals: wp.array(dtype=wp.float64),
):
    e, a, b = wp.tid()
    n = dofmap.shape[1]
    k = offset + (e * n + a) * n + b
    rows[k] = dofmap[e, a]
    cols[k] = dofmap[e, b]
    vals[k] = ke[e, a, b]
