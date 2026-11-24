"""Reverse-mode gradient after overwriting a component of a local vector.

    python examples/material_interface/warp_component_write_gradient.py

Warp's differentiability guide says a vector component may be assigned only
once (https://nvidia.github.io/warp/user_guide/differentiability.html). This
script shows what that means for a history update written the obvious way:
copy the committed state, then overwrite the component that changed. The
forward value is right. The gradient with respect to the overwritten component
of the input is not, because its adjoint is not cleared. No warning is raised.

Expected output with Warp 1.19.0 (exact d/ds is [5, 3]):

    copy, then write     value 72.0  d/dx 4.0  d/ds [6. 3.]   <- wrong
    constructor          value 72.0  d/dx 4.0  d/ds [5. 3.]
    zero, then write     value 72.0  d/dx 4.0  d/ds [5. 3.]

The material layer therefore builds a new state with a constructor
(see materials/models/nitinol.py), and the conformance tests check gradients
through a loading history.
"""

import warp as wp

wp.config.quiet = True


@wp.func
def copy_then_write(s: wp.vec2d, x: wp.float64):
    t = s
    t[0] = x * x
    return t


@wp.func
def constructor(s: wp.vec2d, x: wp.float64):
    return wp.vec2d(x * x, s[1])


@wp.func
def zero_then_write(s: wp.vec2d, x: wp.float64):
    t = wp.vec2d()
    t[0] = x * x
    t[1] = s[1]
    return t


def make_kernel(update):
    @wp.kernel(module="unique")
    def kernel(
        s: wp.array(dtype=wp.vec2d),
        x: wp.array(dtype=wp.float64),
        out: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        t = update(s[i], x[i])
        # out = x^2 + 3 s1 + 5 s0, so d/dx = 2x and d/ds = (5, 3)
        out[i] = t[0] + wp.float64(3.0) * t[1] + wp.float64(5.0) * s[i][0]

    return kernel


def gradients(update):
    s = wp.array([[7.0, 11.0]], dtype=wp.vec2d, requires_grad=True)
    x = wp.array([2.0], dtype=wp.float64, requires_grad=True)
    out = wp.zeros(1, dtype=wp.float64, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(make_kernel(update), dim=1, inputs=[s, x], outputs=[out])
    tape.backward(grads={out: wp.ones(1, dtype=wp.float64)})
    return out.numpy()[0], x.grad.numpy()[0], s.grad.numpy()[0]


if __name__ == "__main__":
    cases = (
        ("copy, then write", copy_then_write),
        ("constructor", constructor),
        ("zero, then write", zero_then_write),
    )
    for label, update in cases:
        value, dx, ds = gradients(update)
        print(f"{label:20s} value {value:.1f}  d/dx {dx:.1f}  d/ds {ds}")
