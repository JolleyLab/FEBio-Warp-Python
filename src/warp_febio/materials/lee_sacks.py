"""fp64 coupled isotropic Lee–Sacks energy, PK1 stress and exact derivative.

Source equations and deliberate numerical differences: docs/PHYSICS.md.
Input stresses use the model's units; deformation gradients are dimensionless.
"""

import numpy as np
import warp as wp


@wp.func
def lee_sacks_energy(
    deformation: wp.mat33d,
    c0: wp.float64,
    c1: wp.float64,
    c2: wp.float64,
    bulk: wp.float64,
):
    jacobian = wp.determinant(deformation)
    invariant = wp.ddot(deformation, deformation) - wp.float64(3.0)
    log_j = wp.log(jacobian)
    return (
        c0 * (wp.float64(0.5) * invariant - log_j)
        + wp.float64(0.5) * c1 * (wp.exp(c2 * invariant * invariant) - wp.float64(1.0))
        + wp.float64(0.5) * bulk * log_j * log_j
    )


@wp.func
def lee_sacks_pk1(
    deformation: wp.mat33d,
    c0: wp.float64,
    c1: wp.float64,
    c2: wp.float64,
    bulk: wp.float64,
):
    invariant = wp.ddot(deformation, deformation) - wp.float64(3.0)
    coefficient = c0 + wp.float64(2.0) * c1 * c2 * invariant * wp.exp(
        c2 * invariant * invariant
    )
    pressure = bulk * wp.log(wp.determinant(deformation)) - c0
    return coefficient * deformation + pressure * wp.transpose(wp.inverse(deformation))


@wp.func
def lee_sacks_tangent_action(
    deformation: wp.mat33d,
    direction: wp.mat33d,
    c0: wp.float64,
    c1: wp.float64,
    c2: wp.float64,
    bulk: wp.float64,
):
    inverse_t = wp.transpose(wp.inverse(deformation))
    invariant = wp.ddot(deformation, deformation) - wp.float64(3.0)
    exponential = wp.exp(c2 * invariant * invariant)
    coefficient = c0 + wp.float64(2.0) * c1 * c2 * invariant * exponential
    coefficient_change = (
        wp.float64(4.0)
        * c1
        * c2
        * exponential
        * (wp.float64(1.0) + wp.float64(2.0) * c2 * invariant * invariant)
        * wp.ddot(deformation, direction)
    )
    pressure = bulk * wp.log(wp.determinant(deformation)) - c0
    pressure_change = bulk * wp.ddot(inverse_t, direction)
    return (
        coefficient_change * deformation
        + coefficient * direction
        + pressure_change * inverse_t
        - pressure * inverse_t * wp.transpose(direction) * inverse_t
    )


@wp.kernel
def evaluate_material(
    deformations: wp.array(dtype=wp.mat33d),
    directions: wp.array(dtype=wp.mat33d),
    c0: wp.float64,
    c1: wp.float64,
    c2: wp.float64,
    bulk: wp.float64,
    energy: wp.array(dtype=wp.float64),
    pk1: wp.array(dtype=wp.mat33d),
    cauchy: wp.array(dtype=wp.mat33d),
    tangent: wp.array(dtype=wp.mat33d),
    jacobian: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    deformation = deformations[index]
    first_piola = lee_sacks_pk1(deformation, c0, c1, c2, bulk)
    determinant = wp.determinant(deformation)
    energy[index] = lee_sacks_energy(deformation, c0, c1, c2, bulk)
    pk1[index] = first_piola
    cauchy[index] = first_piola * wp.transpose(deformation) / determinant
    tangent[index] = lee_sacks_tangent_action(
        deformation, directions[index], c0, c1, c2, bulk
    )
    jacobian[index] = determinant


def evaluate(deformations, parameters, directions=None, device="cpu"):
    """Evaluate prescribed material states, rejecting inverted/nonfinite inputs.

    The tangent is dP/dF, not FEBio's c0-scaled spatial solver tangent.
    No force parameters or stresses are rescaled by ``tangent_scale``.
    """
    deformations = np.asarray(deformations, dtype=np.float64)
    if deformations.ndim != 3 or deformations.shape[1:] != (3, 3):
        raise ValueError("deformations must have shape (count, 3, 3)")
    if not len(deformations) or not np.isfinite(deformations).all():
        raise ValueError("finite, nonempty deformation gradients required")
    if np.any(np.linalg.det(deformations) <= 0):
        raise ValueError("Lee–Sacks requires J > 0")
    directions = (
        np.zeros_like(deformations)
        if directions is None
        else np.asarray(directions, dtype=np.float64)
    )
    if directions.shape != deformations.shape or not np.isfinite(directions).all():
        raise ValueError("finite directions must match deformation shape")
    coefficients = [float(parameters[key]) for key in ("c0", "c1", "c2", "k")]
    if not np.isfinite(coefficients).all():
        raise ValueError("material coefficients must be finite")
    count = len(deformations)
    results = {
        name: wp.empty(count, dtype=dtype, device=device)
        for name, dtype in (
            ("energy", wp.float64),
            ("pk1", wp.mat33d),
            ("cauchy", wp.mat33d),
            ("tangent_action", wp.mat33d),
            ("J", wp.float64),
        )
    }
    wp.launch(
        evaluate_material,
        dim=count,
        inputs=[
            wp.array(deformations, dtype=wp.mat33d, device=device),
            wp.array(directions, dtype=wp.mat33d, device=device),
            *coefficients,
        ],
        outputs=list(results.values()),
        device=device,
    )
    host = {name: value.numpy() for name, value in results.items()}
    if any(not np.isfinite(value).all() for value in host.values()):
        raise FloatingPointError(
            "Nonfinite constitutive result; do not clamp the material law"
        )
    return host
