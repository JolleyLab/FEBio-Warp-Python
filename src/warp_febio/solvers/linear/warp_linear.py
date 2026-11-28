"""Linear solve of the constrained block system.

TEMPLATE (run path draft, November 2026). Structure for discussion, not final.

FEBio counterpart: LinearSolver (Pardiso, skyline, FGMRES, ...) in NumCore.

One call, so the choice of solver stays out of the Newton loop. What is here
is what Warp 1.19 provides (warp.optim.linear: CG, BiCGSTAB, GMRES with a
block-diagonal preconditioner, all on the device), a dense host solve for
checks on small models ("direct"), a sparse LU on the host ("sparse", SciPy)
and, where nvmath-python is installed, a sparse direct solve on the device
with NVIDIA cuDSS ("cudss", cudss.py). The interface does not assume
symmetry: the verification note records that the open device and dilation
models give a non-symmetric stiffness.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import warp as wp
import warp.optim.linear

from ...assembly.system import SolveError

METHODS = ("cg", "bicgstab", "gmres", "direct", "sparse", "cudss")
# Unknowns above which "direct" uses the sparse LU instead of a dense matrix.
DENSE_LIMIT = 3000


def dense(A) -> np.ndarray:
    """Host dense copy of a BSR matrix with 3x3 blocks. For checks only."""
    n = A.nrow
    nnz = A.nnz_sync()
    offsets = A.offsets.numpy()[: n + 1]
    cols = A.columns.numpy()[:nnz]
    vals = A.values.numpy()[:nnz]  # (nnz, 3, 3)
    rows = np.repeat(np.arange(n), np.diff(offsets))
    K = np.zeros((n, 3, A.ncol, 3))
    np.add.at(K, (rows, slice(None), cols, slice(None)), vals)
    return K.reshape(3 * n, 3 * A.ncol)


def sparse_host_solve(A, b: wp.array) -> np.ndarray:
    """A sparse LU on the host (SciPy SuperLU), for models too large for the
    dense check solve on machines without cuDSS. Returns a host array."""
    try:
        import scipy.sparse
        import scipy.sparse.linalg
    except ImportError as err:
        raise ImportError('the "sparse" linear solver needs SciPy, which is not installed') from err

    n = A.nrow
    nnz = A.nnz_sync()
    matrix = scipy.sparse.bsr_matrix(
        (A.values.numpy()[:nnz], A.columns.numpy()[:nnz], A.offsets.numpy()[: n + 1]), shape=(3 * n, 3 * A.ncol)
    ).tocsc()
    return scipy.sparse.linalg.splu(matrix).solve(b.numpy().reshape(-1))


def solve(A, b: wp.array, method: str = "direct", tol: float = 1e-13):
    """Solve A x = b. Returns (x, iterations); x and b are vec3d arrays."""
    if method not in METHODS:
        raise ValueError(f"linear solver must be one of {METHODS}")
    n = b.shape[0]
    if method == "direct" and 3 * n > DENSE_LIMIT and importlib.util.find_spec("scipy") is not None:
        # The dense solve is for checks on small models. Beyond that the
        # same direct answer comes from the sparse LU on the host, where
        # SciPy is installed (it is not a dependency of the package).
        method = "sparse"
    # A solve that fails numerically (a singular matrix, a non-finite result)
    # is a failed step: SolveError, so the time stepper can cut the step. A
    # method that is not installed is not: that error is passed on as it is.
    if method == "direct":
        try:
            x = np.linalg.solve(dense(A), b.numpy().reshape(-1))
        except np.linalg.LinAlgError as err:
            raise SolveError(f"the dense direct solve failed: {err}") from err
        return wp.array(x.reshape(n, 3), dtype=wp.vec3d, device=b.device), 1
    if method == "sparse":
        # Sparse LU on the host (SciPy). A direct solve for models beyond
        # the dense check solve; the matrix makes one round trip.
        try:
            x = sparse_host_solve(A, b)
        except RuntimeError as err:  # SuperLU: exactly singular
            raise SolveError(f"the sparse direct solve failed: {err}") from err
        if not np.isfinite(x).all():
            raise SolveError("the sparse direct solve returned a non-finite solution")
        return wp.array(x.reshape(n, 3), dtype=wp.vec3d, device=b.device), 1
    if method == "cudss":
        # Sparse direct solve on the device (NVIDIA cuDSS, optional).
        from . import cudss

        try:
            return cudss.solve(A, b), 1
        except cudss.CudssNumericalError as err:
            raise SolveError(str(err)) from err
    x = wp.zeros(n, dtype=wp.vec3d, device=b.device)
    bnorm = float(np.sqrt(wp.utils.array_inner(b, b)))
    if bnorm == 0.0:
        return x, 0
    solver = getattr(warp.optim.linear, method)
    M = warp.optim.linear.preconditioner(A, "diag")
    iterations, error, atol = solver(
        A,
        b,
        x,
        tol=tol,
        maxiter=max(200, 60 * n),
        M=M,
        check_every=1,
        use_cuda_graph=False,
    )
    if not np.isfinite(error) or error > max(atol, 10 * tol * bnorm):
        raise SolveError(f"linear solver {method} stopped at residual {error:.3e} after {iterations} iterations")
    return x, int(iterations)
