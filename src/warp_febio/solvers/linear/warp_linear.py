"""Linear solve of the constrained block system.

TEMPLATE (run path draft, November 2026). Structure for discussion, not final.

FEBio counterpart: LinearSolver (Pardiso, skyline, FGMRES, ...) in NumCore.

One call, so the choice of solver stays out of the Newton loop. What is here
is what Warp 1.19 provides (warp.optim.linear: CG, BiCGSTAB, GMRES with a
block-diagonal preconditioner, all on the device) and a dense host solve for
checks on small models. The interface does not assume symmetry: the
verification note records that the open device and dilation models give a
non-symmetric stiffness. A sparse direct solver on the device would be
added behind the same call.
"""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.optim.linear

from ...assembly.system import SolveError

METHODS = ("cg", "bicgstab", "gmres", "direct")


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


def solve(A, b: wp.array, method: str = "direct", tol: float = 1e-13):
    """Solve A x = b. Returns (x, iterations); x and b are vec3d arrays."""
    if method not in METHODS:
        raise ValueError(f"linear solver must be one of {METHODS}")
    n = b.shape[0]
    if method == "direct":
        x = np.linalg.solve(dense(A), b.numpy().reshape(-1))
        return wp.array(x.reshape(n, 3), dtype=wp.vec3d, device=b.device), 1
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
        raise SolveError(
            f"linear solver {method} stopped at residual {error:.3e} after {iterations} iterations"
        )
    return x, int(iterations)
