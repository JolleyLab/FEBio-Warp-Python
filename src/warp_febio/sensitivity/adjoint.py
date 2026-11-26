"""Gradients of an output with respect to parameters, through equilibrium.

TEMPLATE (run path draft, November 2026). Structure for discussion, not final.

FEBio counterpart: none. This is the addition.

The idea
    A converged state satisfies R(u, p) = 0, with R = F_external - F_internal
    and K = -dR/du the tangent the Newton loop already forms. For an output
    J(u), implicit differentiation gives

        dJ/dp = lambda . dR/dp,        K^T lambda = dJ/du,

    so a gradient with respect to any number of parameters costs one extra
    linear solve with the transpose of the same matrix, plus one reverse pass
    of the residual. Nothing about the Newton iterations is recorded.

How Warp is used
    The reverse pass is a `wp.Tape` over the same `update` and `residual`
    kernels the forward solve launches. Each term's parameters live in device
    arrays (`Term.parameters()`), so the tape returns d(residual)/d(parameter)
    contracted with lambda for all of them at once. A term that has its
    derivative in closed form can supply it directly instead (contact has
    `residual_parameter_vjp`).

Limits of this template
    Static equilibrium at one state. Outputs that depend on parameters
    directly (a stress, say) need the explicit term dJ/dp added. Gradients
    through a dynamic trajectory, through history, and through a change of
    contact pairs are not designed here.
"""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.sparse

from ..assembly.system import State, System
from ..solvers.linear.warp_linear import solve


def parameter_gradient(system: System, state: State, dJ_du: np.ndarray,
                       linear_solver: str = "direct"):
    """dJ/dp for every parameter array of the system, at a converged state.

    dJ_du: (n_entries, 3) derivative of the output with respect to the
    solution. Returns (gradients, lambda): a dict keyed like
    `system.parameters()`, and the adjoint solution on the host.
    """
    device = system.device
    # Adjoint solve with the transpose of the constrained tangent. Constrained
    # components carry no sensitivity through equilibrium: their rows are
    # identity and the right-hand side is masked.
    system.update(state)
    A = system.constrained_matrix(system.tangent(state))
    rhs = wp.array(dJ_du * system.fields.free, dtype=wp.vec3d, device=device)
    lam, _ = solve(warp.sparse.bsr_transposed(A), rhs, linear_solver)
    system.mask(lam)

    # Reverse pass of the residual: lambda . dR/dp for all parameters at once.
    params = system.parameters()
    tape = wp.Tape()
    with tape:
        system.update(state)
        R = system.residual(state, requires_grad=True)
    tape.backward(grads={R: lam})
    gradients = {key: array.grad.numpy().copy() for key, array in params.items()}
    tape.zero()
    return gradients, lam.numpy()
