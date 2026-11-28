"""A sparse direct solve on the device with NVIDIA cuDSS. Optional.

TEMPLATE (linear solve draft, November 2026). Structure for discussion, not final.
Design note: docs/design/linear_solve.md.

FEBio counterpart: a direct LinearSolver (Pardiso, skyline), with its
three calls PreProcess (analysis of the pattern), Factor and BackSolve.

Why it is here. Warp 1.19 has no sparse direct solver, and its Krylov solvers
with the diagonal preconditioner need hundreds of iterations per solve on
shells (about 565 on an 1863-shell patch). cuDSS factors such a system in a
fraction of a second. It is reached through nvmath-python, which is not a
dependency of this package: install it to use this method, for example

    uv pip install "nvmath-python[cu12]" cupy-cuda12x

How it fits the run path. The tangent is a warp.sparse BSR matrix with 3x3
blocks and a fixed pattern. cuDSS takes scalar CSR. The pattern is converted
once per matrix, on the device, into CSR index arrays and a gather map; each
later solve gathers the new values through the map, factors and solves. The
analysis (reordering and symbolic factorization) is done once and reused for
as long as the pattern stays, which is the life of the System. Nothing
leaves the device: the Warp arrays are viewed by CuPy without a copy.

The matrix type is GENERAL: the tangent can be non-symmetric (follower
pressure) and indefinite (slack or compressed chords), and with the SPD type
cuDSS returns NaN on an indefinite matrix without raising. A symmetric type
would halve the work where it applies; that choice is left to decision L4.

Not here yet: the transpose solve for the adjoint from the same factors
(cuDSS 1.0 has it; nvmath-python 1.0.1 does not expose it), and any use of
the hybrid-memory or multi-GPU modes.
"""

from __future__ import annotations

import warp as wp


class CudssUnavailable(RuntimeError):
    """cuDSS cannot be used here: not installed, or not a CUDA device."""


class CudssNumericalError(RuntimeError):
    """cuDSS ran and did not produce a usable solution."""


def _modules():
    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import nvmath.sparse.advanced as nsa
    except ImportError as err:
        raise CudssUnavailable(
            'the "cudss" linear solver needs nvmath-python and CuPy, which are not installed: '
            'uv pip install "nvmath-python[cu12]" cupy-cuda12x'
        ) from err
    return cp, csp, nsa


def _threading_layer():
    """Path of cuDSS's OpenMP threading layer, which its wheel ships, or None.
    cuDSS uses it for the host part of the analysis (reordering); without it
    that part runs on one thread and nvmath warns at every analysis."""
    import glob
    import os
    import sys

    found = glob.glob(
        os.path.join(
            sys.prefix, "lib", "python3*", "site-packages", "nvidia", "cu1*", "lib", "libcudss_mtlayer_gomp.so*"
        )
    )
    return found[0] if found else None


def _csr_pattern(cp, A):
    """Scalar CSR index arrays of a BSR matrix with 3x3 blocks, and the map
    that gathers its value buffer into CSR order. All on the device."""
    n = A.nrow
    nnz = A.nnz_sync()
    off = cp.asarray(A.offsets)[: n + 1].astype(cp.int64)
    col = cp.asarray(A.columns)[:nnz]
    per_row = off[1:] - off[:-1]
    block_row = cp.repeat(cp.arange(n, dtype=cp.int64), per_row.get().tolist())
    local = cp.arange(nnz, dtype=cp.int64) - off[block_row]
    i = cp.arange(3, dtype=cp.int64)
    # Entry (block k, row i, column j) goes to
    # 9 off[r] + i * 3 * blocks_in_row[r] + 3 * local + j.
    dest = (
        (9 * off[block_row])[:, None, None]
        + i[None, :, None] * 3 * per_row[block_row][:, None, None]
        + 3 * local[:, None, None]
        + i[None, None, :]
    ).reshape(-1)
    gather = cp.empty(9 * nnz, dtype=cp.int64)
    gather[dest] = cp.arange(9 * nnz, dtype=cp.int64)
    indices = cp.empty(9 * nnz, dtype=cp.int32)
    columns = 3 * col[:, None, None] + cp.arange(3, dtype=cp.int32)[None, None, :] + cp.zeros((1, 3, 1), cp.int32)
    indices[dest] = columns.reshape(-1).astype(cp.int32)
    indptr = cp.zeros(3 * n + 1, dtype=cp.int32)
    indptr[1:] = cp.cumsum(cp.repeat(3 * per_row, 3)).astype(cp.int32)
    return indptr, indices, gather, nnz


class CudssSolver:
    """Analysis once, then factor and solve per call, for one BSR matrix
    whose pattern does not change."""

    def __init__(self, A):
        cp, csp, nsa = _modules()
        if not A.values.device.is_cuda:
            raise CudssUnavailable('the "cudss" linear solver runs on a CUDA device')
        self.cp = cp
        self.n = A.nrow
        self.device = A.values.device
        with cp.cuda.Device(self.device.ordinal):
            indptr, indices, self.gather, self.nnz = _csr_pattern(cp, A)
            # The block pattern, kept to recognise the same pattern in a
            # matrix that lives in other buffers.
            self.offsets = cp.asarray(A.offsets)[: self.n + 1].copy()
            self.columns = cp.asarray(A.columns)[: self.nnz].copy()
            data = cp.asarray(A.values).reshape(-1)[self.gather]
            self.matrix = csp.csr_matrix((data, indices, indptr), shape=(3 * self.n, 3 * self.n))
            self.rhs = cp.zeros(3 * self.n, dtype=cp.float64)
            options = nsa.DirectSolverOptions(
                sparse_system_type=nsa.DirectSolverMatrixType.GENERAL,
                blocking=True,
                multithreading_lib=_threading_layer(),
            )
            self.solver = nsa.DirectSolver(self.matrix, self.rhs, options=options, execution=nsa.ExecutionCUDA())
            self.solver.plan()

    def same_pattern(self, A) -> bool:
        cp = self.cp
        if A.nrow != self.n or A.nnz_sync() != self.nnz:
            return False
        with cp.cuda.Device(self.device.ordinal):
            return bool(
                cp.array_equal(cp.asarray(A.offsets)[: self.n + 1], self.offsets)
                and cp.array_equal(cp.asarray(A.columns)[: self.nnz], self.columns)
            )

    def solve(self, A, b: wp.array) -> wp.array:
        cp = self.cp
        wp.synchronize_device(self.device)
        with cp.cuda.Device(self.device.ordinal):
            # New values into the buffers the solver was planned with.
            self.matrix.data[...] = cp.asarray(A.values).reshape(-1)[self.gather]
            self.rhs[...] = cp.asarray(b).reshape(-1)
            self.solver.factorize()
            x = self.solver.solve()
            if not bool(cp.isfinite(x).all()):
                raise CudssNumericalError("cuDSS returned a non-finite solution")
            out = wp.empty(self.n, dtype=wp.vec3d, device=self.device)
            cp.asarray(out).reshape(-1)[...] = x
            cp.cuda.Stream.null.synchronize()
        return out

    def close(self) -> None:
        self.solver.free()


# One solver, with its analysis, per size of system. A structural matrix
# keeps its pattern for the life of a System, so the analysis is done once.
# With contact the pattern follows the pairs: the analysis is then repeated
# whenever the pattern differs from the last one, and only then.
_SOLVERS: dict = {}


def solve(A, b: wp.array) -> wp.array:
    """Solve A x = b with cuDSS, reusing the analysis while the pattern stays."""
    key = (A.values.device.alias, A.nrow)
    solver = _SOLVERS.get(key)
    if solver is None or not solver.same_pattern(A):
        if solver is not None:
            del _SOLVERS[key]  # first, so a failed analysis leaves no closed solver behind
            solver.close()
        solver = _SOLVERS[key] = CudssSolver(A)
    return solver.solve(A, b)
