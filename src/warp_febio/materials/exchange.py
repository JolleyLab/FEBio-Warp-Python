"""Exchange with FEBio-convention solvers (FEBio domains, the FEBioMFEMiFSI plugin).

Any registered model, whichever way it was written, is evaluated on a batch of
quadrature points and returns what FEBio solid domains consume: Cauchy stress
and the spatial elasticity tensor as a symmetric 6x6 in FEBio tens4ds order
(11, 22, 33, 12, 23, 13). See
https://github.com/febiosoftware/FEBio/blob/develop/FECore/tens4d.h
and, for how the MFEM plugin calls FEBio materials,
https://github.com/JolleyLab/FEBioMFEMiFSI/blob/main/MFEMImSolid/MFEMImmersedElasticDomain.cpp

Column b of the 6x6 is the model's spatial_tangent_action applied to the b-th
symmetric unit tensor, so this kernel adds no mechanics of its own. A model may
instead supply `spatial_tangent`, a direct 6x6; whether that is faster is to be
measured per model on GPU (on CPU the two were within about 10 percent for
neo-Hookean, in a memory-bound run).

The caller owns the memory. `wrap` builds Warp arrays on caller pointers so
nothing is copied. On GPU, pass the device pointer that MFEM returns from
Vector::Read() / Vector::Write(). The GPU pointer path has not been run yet.
"""

import numpy as np
import warp as wp

from .status import NEGATIVE_JACOBIAN, NON_FINITE, OK
from .voigt import VI, VJ, VOIGT  # noqa: F401  (VOIGT re-exported for callers)

_EXCHANGE_KERNELS = {}


def make_exchange_kernel(model, dtype=wp.float64, combined=None):
    """combined: None uses the model's stress_and_tangent when it has one.
    False forces the six-action path, which the tests use to compare the two."""
    use_combined = model.stress_and_tangent is not None and combined is not False
    key = (model, dtype, use_combined)
    if key in _EXCHANGE_KERNELS:
        return _EXCHANGE_KERNELS[key]
    mat = wp.types.matrix(shape=(3, 3), dtype=dtype)
    mat66 = wp.types.matrix(shape=(6, 6), dtype=dtype)
    pvec = model.param_vector(dtype)
    gvec = model.geometry_vector(dtype)
    cauchy, spatial_tangent_action = model.cauchy, model.spatial_tangent_action
    vi, vj = VI, VJ
    stress_and_tangent = model.stress_and_tangent

    def exchange(
        F: wp.array(dtype=mat),
        g: wp.array(dtype=gvec),
        region: wp.array(dtype=int),
        params: wp.array(dtype=pvec),
        sigma: wp.array(dtype=mat),
        c: wp.array(dtype=mat66),
        status: wp.array(dtype=int),
    ):
        n = wp.tid()
        # The caller chooses the time level of F. FEBioMFEMiFSI, for example,
        # passes the generalized-alpha intermediate value.
        Fn = F[n]
        # Failure rule of status.py: no material call for J <= 0, and a
        # non-finite result is reported instead of written. The caller's
        # buffers are zeroed for failed points.
        zero33 = mat()
        zero66 = mat66()
        if wp.determinant(Fn) <= dtype(0.0):
            sigma[n] = zero33
            c[n] = zero66
            status[n] = wp.static(NEGATIVE_JACOBIAN)
            return
        p = params[region[n]]
        if wp.static(use_combined):
            sigma_n, out = stress_and_tangent(Fn, g[n], p)
        else:
            sigma_n = cauchy(Fn, g[n], p)
            out = mat66()
            half = dtype(0.5)
            # Shear columns use (e_k e_l + e_l e_k)/2, so the entry is c_ijkl
            # itself (the tens4ds convention), not engineering shear.
            for b in range(6):
                d = mat()
                d[vi[b], vj[b]] = d[vi[b], vj[b]] + half
                d[vj[b], vi[b]] = d[vj[b], vi[b]] + half
                cd = spatial_tangent_action(Fn, d, g[n], p)
                for a in range(6):
                    out[a, b] = cd[vi[a], vj[a]]
        if not (wp.isfinite(sigma_n) and wp.isfinite(out)):
            sigma[n] = zero33
            c[n] = zero66
            status[n] = wp.static(NON_FINITE)
            return
        sigma[n] = sigma_n
        c[n] = out
        status[n] = wp.static(OK)

    kernel = wp.kernel(exchange, module="unique")
    _EXCHANGE_KERNELS[key] = kernel
    return kernel


def wrap(ptr, count, dtype, device="cpu"):
    """Warp array view on caller-owned memory (no copy)."""
    return wp.array(ptr=ptr, shape=(count,), dtype=dtype, device=device)


def evaluate_exchange(model, F, params, geometry=None, device="cpu", combined=None):
    """Host convenience (FP64): Cauchy stress (n,3,3) and tangent (n,6,6).

    Outputs are written by the kernel directly into NumPy buffers allocated
    here, which exercises the same caller-owned-memory path a solver would use.
    """
    F = np.ascontiguousarray(F, dtype=np.float64)
    n = len(F)
    g = model.pack_geometry(n, **(geometry or {}))
    mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
    if device == "cpu":
        sigma = np.zeros((n, 3, 3))
        c = np.zeros((n, 6, 6))
        sigma_wp = wrap(sigma.ctypes.data, n, wp.mat33d)
        c_wp = wrap(c.ctypes.data, n, mat66)
    else:
        sigma_wp = wp.zeros(n, dtype=wp.mat33d, device=device)
        c_wp = wp.zeros(n, dtype=mat66, device=device)
    status = wp.zeros(n, dtype=int, device=device)
    wp.launch(
        make_exchange_kernel(model, combined=combined),
        dim=n,
        inputs=[
            wp.array(F, dtype=wp.mat33d, device=device),
            wp.array(g, dtype=model.geometry_vector(), device=device),
            wp.zeros(n, dtype=int, device=device),
            wp.array([model.pack(**params)], dtype=model.param_vector(), device=device),
        ],
        outputs=[sigma_wp, c_wp, status],
        device=device,
    )
    if device != "cpu":
        sigma, c = sigma_wp.numpy(), c_wp.numpy()
    return {"cauchy": sigma, "tangent": c, "status": status.numpy()}
