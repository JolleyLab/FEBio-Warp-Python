"""Nodal displacement constraints for assembled Warp FEM systems."""

import warp as wp
import warp.fem as fem


@fem.integrand
def boundary_projector(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


def project_nodal_boundary(matrix, rhs, space, boundary, prescribed_form):
    """Apply a full vector constraint to the nodes on the selected boundary.

    prescribed_form is a Warp FEM linear form returning prescribed displacement
    dotted with the test field. The matrix and RHS are updated in place.
    """
    test = fem.make_test(space, domain=boundary)
    trial = fem.make_trial(space, domain=boundary)
    # Nodal integration gives one projector block per boundary node.
    projector = fem.integrate(
        boundary_projector, fields={"u": trial, "v": test}, assembly="nodal"
    )
    prescribed = fem.integrate(prescribed_form, fields={"v": test}, assembly="nodal")
    fem.project_linear_system(matrix, rhs, projector, prescribed)
