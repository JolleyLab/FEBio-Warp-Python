"""Small-strain solid patch test through PUBLIC Warp FEM APIs.

This adapter demonstrates the full geometry -> fields -> forms -> BSR -> BC ->
CG workflow. It is not the nonlinear CardioFEM solid or director-shell solver.
The prescribed affine field is an exact solution of homogeneous linear elasticity.
"""

import time
import numpy as np
import warp as wp
import warp.fem as fem
from warp.optim.linear import cg
from warp.sparse import bsr_mv

from warp_febio_pilot.constraints.dirichlet import project_nodal_boundary
from warp_febio_pilot.elements.linear_elastic_solid import linear_elastic_stiffness


@fem.integrand
def prescribed_displacement(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    position = domain(s)
    # A deliberately small affine strain; coordinates and displacements are meters.
    exact = wp.vec3d(
        wp.float64(0.001) * position[0],
        wp.float64(-0.0002) * position[1],
        wp.float64(0.0003) * position[2],
    )
    return wp.dot(exact, v(s))


def run(device="cpu"):
    """Assemble and solve one synthetic 2x2x2-cell linear elastic cube."""
    wp.init()
    with wp.ScopedDevice(device):
        start = time.perf_counter()
        geometry = fem.Grid3D(
            res=wp.vec3i(2, 2, 2),
            bounds_lo=wp.vec3d(0.0),
            bounds_hi=wp.vec3d(0.01),
            scalar_type=wp.float64,
        )
        space = fem.make_polynomial_space(geometry, degree=1, dtype=wp.vec3d)
        cells = fem.Cells(geometry)
        test = fem.make_test(space, domain=cells)
        trial = fem.make_trial(space, domain=cells)
        young, poisson = 1.0e6, 0.3
        lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        shear = young / (2 * (1 + poisson))
        matrix = fem.integrate(
            linear_elastic_stiffness,
            fields={"u": trial, "v": test},
            values={"lame": wp.float64(lame), "shear": wp.float64(shear)},
        )
        rhs = wp.zeros(space.node_count(), dtype=wp.vec3d)
        boundary = fem.BoundarySides(geometry)
        project_nodal_boundary(matrix, rhs, space, boundary, prescribed_displacement)
        solution = wp.zeros_like(rhs)
        # This bounded linear problem is SPD: positive Lamé moduli, positive
        # quadrature, and full displacement constraints eliminating rigid modes.
        iterations, residual, tolerance = cg(
            matrix, rhs, solution, tol=1e-10, atol=1e-13, maxiter=200, check_every=1
        )
        wp.synchronize()
        elapsed = time.perf_counter() - start
        positions = space.node_positions().numpy()
        expected = positions * np.array([0.001, -0.0002, 0.0003])
        error = float(np.max(np.abs(solution.numpy() - expected)))
        # Check the actual projected residual independently of CG's recurrence.
        actual = wp.clone(rhs)
        bsr_mv(matrix, solution, actual, alpha=-1.0, beta=1.0)
        true_residual = float(np.linalg.norm(actual.numpy()))
        if error > 1e-10 or true_residual > max(1e-11, 10 * float(tolerance)):
            raise RuntimeError(
                f"Solid patch test failed: error={error}, residual={true_residual}"
            )
        return dict(
            case="affine linear-elastic solid verification",
            warp_version=wp.__version__,
            device=str(wp.get_device()),
            nodes=space.node_count(),
            elements=8,
            young_Pa=young,
            poisson=poisson,
            maximum_error_m=error,
            cg_iterations=int(iterations),
            cg_reported_residual=float(residual),
            true_projected_residual=true_residual,
            cg_tolerance=float(tolerance),
            setup_and_solve_wall_s=elapsed,
            timing_includes_first_compile=True,
            status="manufactured-solution verification only; not FEBio comparison",
        )
