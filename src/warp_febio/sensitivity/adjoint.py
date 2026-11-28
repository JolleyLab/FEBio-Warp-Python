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

Two ways to call it
    parameter_gradient   given dJ/du as an array, returns dJ/dp. For a check
                         or a hand-written output.
    solve_on_tape        registers the solve on a `wp.Tape` as one operation
                         whose reverse step is the adjoint. Kernels recorded
                         before it (parameters computed from design
                         variables) and after it (an output computed from the
                         solution) are differentiated by Warp as usual, so
                         `tape.backward(loss)` gives the whole chain.

Limits of this template
    Static equilibrium at the final state of a path-independent model.
    Outputs that read parameters or point data directly (a stress, say) need
    the explicit term dJ/dp, which is not wired here. The design for dynamics,
    history and contact is in docs/design/run_path.md; none is implemented.
"""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.sparse

from ..assembly.system import State, System
from ..solvers.linear.warp_linear import solve


def adjoint_solve(
    system: System,
    state: State,
    dJ_du,
    linear_solver: str = "direct",
    exact: bool = True,
) -> wp.array:
    """lambda from K^T lambda = dJ/du at a converged state.

    The tangent is taken in its exact form: a tangent that Newton may
    approximate (FEBio's symmetrized pressure stiffness) would give a wrong
    gradient. Constrained components carry no sensitivity through
    equilibrium: their rows are identity and the right-hand side is masked.
    """
    system.update(state)
    system.tangent(state, exact=exact)
    A = system.constrained_matrix()
    rhs = wp.array(np.asarray(dJ_du) * system.fields.free, dtype=wp.vec3d, device=system.device)
    lam, _ = solve(warp.sparse.bsr_transposed(A), rhs, linear_solver)
    system.mask(lam)
    return lam


def accumulate_parameter_gradients(system: System, state: State, lam: wp.array) -> None:
    """Add lambda . dR/dp to the `.grad` of every parameter array.

    One reverse pass of the forward `update` and `residual` kernels. Adds to
    what is already in `.grad`, as a Warp reverse pass does, and leaves the
    gradients of the terms' work arrays at zero.
    """
    params = system.parameters()
    inner = wp.Tape()
    with inner:
        system.update(state)
        R = system.residual(state, requires_grad=True)
    inner.backward(grads={R: lam})
    totals = {key: wp.clone(array.grad) for key, array in params.items()}
    inner.zero()  # clears the work arrays, and the parameters with them
    for key, array in params.items():
        wp.copy(array.grad, totals[key])


def parameter_gradient(
    system: System,
    state: State,
    dJ_du: np.ndarray,
    linear_solver: str = "direct",
    exact: bool = True,
):
    """dJ/dp for every parameter array of the system, at a converged state.

    dJ_du: (n_entries, 3) derivative of the output with respect to the
    solution. Returns (gradients, lambda): a dict keyed like
    `system.parameters()`, and the adjoint solution on the host.
    exact=False uses the tangent Newton uses; it exists only so a test can
    show that the gradient is then wrong.
    """
    params = system.parameters()
    for array in params.values():
        array.grad.zero_()
    lam = adjoint_solve(system, state, dJ_du, linear_solver, exact)
    accumulate_parameter_gradients(system, state, lam)
    gradients = {key: array.grad.numpy().copy() for key, array in params.items()}
    for array in params.values():
        array.grad.zero_()
    return gradients, lam.numpy()


def _active_tape():
    # Warp 1.19 has no public way to ask whether a tape is recording.
    try:
        return wp._src.context.runtime.tape
    except AttributeError:
        return None


def solve_on_tape(system: System, tape: wp.Tape, linear_solver: str = "direct") -> State:
    """Run the model and put the solve on `tape` as one differentiable step.

    Returns the final State; its `u` has requires_grad. In the reverse pass
    of `tape`, whatever gradient has reached `u` is turned into gradients of
    the system's parameter arrays by one adjoint solve and one reverse pass
    of the residual. The Newton iterations are not recorded.

        tape = wp.Tape()
        with tape:
            wp.launch(design_to_pressure, ...)      # optional, upstream
        state = solve_on_tape(system, tape)
        with tape:
            wp.launch(output_kernel, inputs=[state.u], outputs=[loss])
        tape.backward(loss)                          # gradients everywhere

    Call it outside a `with tape:` block, so the forward iterations are not
    recorded.
    """
    from ..analysis.steps import run

    if _active_tape() is not None:
        raise RuntimeError("solve_on_tape must be called outside a `with tape:` block")
    _, states = run(system, linear_solver=linear_solver, keep_states=True)
    final = states[-1]
    out = State(final.time, wp.clone(final.u, requires_grad=True))

    def backward():
        lam = adjoint_solve(system, out, out.u.grad.numpy(), linear_solver)
        accumulate_parameter_gradients(system, out, lam)

    tape.record_func(backward, arrays=[out.u, *system.parameters().values()])
    return out
