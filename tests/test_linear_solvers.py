# SPDX-FileCopyrightText: Copyright (c) 2026 The FEBio-Warp-Python Authors
# SPDX-License-Identifier: Apache-2.0
"""The linear solve call on small matrices: the direct methods agree, a
singular matrix is a failed step, and "direct" changes method with size.

DRAFT (linear solve template, November 2026). Runs against FEBio references
are in test_run_path.py.
"""

import unittest

import numpy as np
import warp as wp
import warp.sparse
from support import DEVICE, setUpModule, tearDownModule  # noqa: F401  (device for the module)

from warp_febio.assembly.system import SolveError
from warp_febio.solvers.linear import warp_linear
from warp_febio.solvers.linear.warp_linear import dense, solve


def block_matrix(n=12, seed=0, symmetric=True, singular=False):
    """A sparse matrix of 3x3 blocks on the test device, and it dense on the host."""
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in {i, (i + 1) % n, (i + 5) % n}:
            rows.append(i)
            cols.append(j)
            vals.append(rng.normal(size=(3, 3)) + (8.0 * np.eye(3) if i == j else 0.0))
    if symmetric:
        rows, cols, vals = rows + cols, cols + rows, vals + [v.T for v in vals]
    if singular:
        vals = [0.0 * v if r == 0 or c == 0 else v for r, c, v in zip(rows, cols, vals)]
    A = warp.sparse.bsr_from_triplets(
        n,
        n,
        wp.array(rows, dtype=wp.int32, device=DEVICE),
        wp.array(cols, dtype=wp.int32, device=DEVICE),
        wp.array(np.array(vals), dtype=wp.mat33d, device=DEVICE),
        prune_numerical_zeros=False,
    )
    return A, dense(A)


class Methods(unittest.TestCase):
    def test_methods_agree_with_a_host_solve(self):
        for symmetric in (True, False):
            A, K = block_matrix(symmetric=symmetric)
            b = np.random.default_rng(1).normal(size=(A.nrow, 3))
            expected = np.linalg.solve(K, b.reshape(-1)).reshape(-1, 3)
            for method in ("direct", "sparse", "bicgstab", "gmres") + (("cg",) if symmetric else ()):
                with self.subTest(method=method, symmetric=symmetric):
                    x, _ = solve(A, wp.array(b, dtype=wp.vec3d, device=DEVICE), method)
                    np.testing.assert_allclose(x.numpy(), expected, rtol=0, atol=1e-9 * abs(expected).max())

    def test_unknown_method_is_refused(self):
        A, _ = block_matrix()
        with self.assertRaises(ValueError):
            solve(A, wp.zeros(A.nrow, dtype=wp.vec3d, device=DEVICE), "pardiso")

    def test_a_singular_matrix_is_a_failed_step(self):
        """SolveError, which the time stepper answers by cutting the step,
        and not an error that ends the run."""
        A, _ = block_matrix(singular=True)
        b = wp.array(np.ones((A.nrow, 3)), dtype=wp.vec3d, device=DEVICE)
        for method in ("direct", "sparse"):
            with self.subTest(method=method), self.assertRaises(SolveError):
                solve(A, b, method)

    def test_direct_uses_the_sparse_solve_above_the_limit(self):
        A, K = block_matrix()
        b = np.random.default_rng(2).normal(size=(A.nrow, 3))
        expected = np.linalg.solve(K, b.reshape(-1)).reshape(-1, 3)
        calls = []
        original, limit = warp_linear.sparse_host_solve, warp_linear.DENSE_LIMIT
        warp_linear.sparse_host_solve = lambda *args: calls.append(1) or original(*args)
        try:
            solve(A, wp.array(b, dtype=wp.vec3d, device=DEVICE), "direct")
            self.assertEqual(calls, [])  # 36 unknowns: the dense check solve
            warp_linear.DENSE_LIMIT = 10
            x, _ = solve(A, wp.array(b, dtype=wp.vec3d, device=DEVICE), "direct")
            self.assertEqual(calls, [1])
        finally:
            warp_linear.sparse_host_solve, warp_linear.DENSE_LIMIT = original, limit
        np.testing.assert_allclose(x.numpy(), expected, rtol=0, atol=1e-10 * abs(expected).max())


if __name__ == "__main__":
    wp.config.quiet = True
    unittest.main()
