"""Exchange with FEBio-convention solvers (FEBio domains, the FEBioMFEMiFSI plugin).

Any registered model, whichever way it was written, is evaluated on a batch of
quadrature points and returns what FEBio solid domains consume: Cauchy stress
and the spatial elasticity tensor as a symmetric 6x6 in FEBio tens4ds order
(11, 22, 33, 12, 23, 13). See
https://github.com/febiosoftware/FEBio/blob/develop/FECore/tens4d.h
and, for how the MFEM plugin calls FEBio materials,
https://github.com/JolleyLab/FEBioMFEMiFSI/blob/main/MFEMImSolid/MFEMImmersedElasticDomain.cpp

The tangent is recovered from the reference-frame functions:

    J c:d = dP[d F] F^T + P (d F)^T - d tau - tau d,    tau = P F^T

which is the inverse of the relation used in model.from_spatial.

The caller owns the memory. `wrap` builds Warp arrays on caller pointers so
nothing is copied. On GPU, pass the device pointer that MFEM returns from
Vector::Read() / Vector::Write(). The GPU pointer path has not been run yet.
"""

import numpy as np
import warp as wp

# Voigt index pairs in FEBio tens4ds order.
VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))

_EXCHANGE_KERNELS = {}


def make_exchange_kernel(model, dtype=wp.float64):
    key = (model.name, dtype)
    if key in _EXCHANGE_KERNELS:
        return _EXCHANGE_KERNELS[key]
    mat = wp.types.matrix(shape=(3, 3), dtype=dtype)
    mat66 = wp.types.matrix(shape=(6, 6), dtype=dtype)
    vec = model.param_vector(dtype)
    pk1, tangent_action = model.pk1, model.tangent_action
    vec6i = wp.types.vector(length=6, dtype=wp.int32)
    vi = wp.constant(vec6i(0, 1, 2, 0, 1, 0))
    vj = wp.constant(vec6i(0, 1, 2, 1, 2, 2))

    def exchange(
        F: wp.array(dtype=mat),
        Q: wp.array(dtype=mat),
        region: wp.array(dtype=int),
        params: wp.array(dtype=vec),
        sigma: wp.array(dtype=mat),
        c: wp.array(dtype=mat66),
    ):
        n = wp.tid()
        # The caller chooses the time level of F. FEBioMFEMiFSI, for example,
        # passes the generalized-alpha intermediate value.
        Fn = F[n]
        p = params[region[n]]
        J = wp.determinant(Fn)
        Ft = wp.transpose(Fn)
        P = pk1(Fn, Q[n], p)
        tau = P * Ft  # Kirchhoff stress, tau = P F^T = J sigma
        sigma[n] = tau / J
        out = mat66()
        half = dtype(0.5)
        # Column b of the 6x6 is c contracted with the b-th symmetric unit
        # tensor. Shear columns use (e_k e_l + e_l e_k)/2, so the entry is c_ijkl
        # itself (the tens4ds convention), not engineering shear.
        for b in range(6):
            d = mat()
            d[vi[b], vj[b]] = d[vi[b], vj[b]] + half
            d[vj[b], vi[b]] = d[vj[b], vi[b]] + half
            H = d * Fn  # the perturbation of F whose velocity gradient is d
            cd = (
                tangent_action(Fn, H, Q[n], p) * Ft
                + P * wp.transpose(H)
                - d * tau
                - tau * d
            ) / J
            for a in range(6):
                out[a, b] = cd[vi[a], vj[a]]
        c[n] = out

    kernel = wp.kernel(exchange, module="unique")
    _EXCHANGE_KERNELS[key] = kernel
    return kernel


def wrap(ptr, count, dtype, device="cpu"):
    """Warp array view on caller-owned memory (no copy)."""
    return wp.array(ptr=ptr, shape=(count,), dtype=dtype, device=device)


def evaluate_exchange(model, F, params, Q=None, device="cpu"):
    """Host convenience (FP64): Cauchy stress (n,3,3) and tangent (n,6,6).

    Outputs are written by the kernel directly into NumPy buffers allocated
    here, which exercises the same caller-owned-memory path a solver would use.
    """
    F = np.ascontiguousarray(F, dtype=np.float64)
    n = len(F)
    Q = np.tile(np.eye(3), (n, 1, 1)) if Q is None else np.ascontiguousarray(Q)
    mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
    if device == "cpu":
        sigma = np.zeros((n, 3, 3))
        c = np.zeros((n, 6, 6))
        sigma_wp = wrap(sigma.ctypes.data, n, wp.mat33d)
        c_wp = wrap(c.ctypes.data, n, mat66)
    else:
        sigma_wp = wp.zeros(n, dtype=wp.mat33d, device=device)
        c_wp = wp.zeros(n, dtype=mat66, device=device)
    wp.launch(
        make_exchange_kernel(model),
        dim=n,
        inputs=[
            wp.array(F, dtype=wp.mat33d, device=device),
            wp.array(Q, dtype=wp.mat33d, device=device),
            wp.zeros(n, dtype=int, device=device),
            wp.array([model.pack(**params)], dtype=model.param_vector(), device=device),
        ],
        outputs=[sigma_wp, c_wp],
        device=device,
    )
    if device != "cpu":
        sigma, c = sigma_wp.numpy(), c_wp.numpy()
    return {"cauchy": sigma, "tangent": c}
