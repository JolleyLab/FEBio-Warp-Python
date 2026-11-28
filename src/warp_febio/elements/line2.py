"""Two-node line elements: what trusses and springs have in common.

DRAFT (chordae elements, November 2026). Structure for discussion, not final.

FEBio keeps two domains for two-node elements that carry an axial force:

    "linear-truss"   FELinearTrussDomain, elements of a part, a truss material
    "discrete"       FEDiscreteElasticDomain, element sets of the Discrete
                     section, a discrete (spring) material

    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FELinearTrussDomain.cpp
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEDiscreteElasticDomain.cpp

Both stay separate domains here (elements/truss_domain.py and
elements/discrete_domain.py), because their inputs, parameters, mass and
outputs differ. Their mechanics is one case, though. With a and b the two
nodes, e the unit vector from a to b, l the current length, and N(l) the axial
force (tension positive):

    internal force on b    N e                 (and -N e on a)
    stiffness block        K = (dN/dl - N/l) e (x) e + (N/l) I
    element matrix         [[K, -K], [-K, K]]

FEBio's two stiffness expressions are this K:

    truss    N = tau V / l,  dN/dl = V / l^2 (E_t - tau)
             K = V / l^2 (E_t - 2 tau) e (x) e + (tau V / l^2) I
             (FELinearTrussDomain::ElementStiffness)
    spring   K = (E - F / l) e (x) e + (F / l) I,  E = dF/dl
             (FENonlinearSpringMaterial::Stiffness, FESpringMaterial::Stiffness)

So K is the exact derivative of the internal force for both, and one function
serves both domains. It is also what makes the axial force N the natural
per-element output: the chord tension is N, whichever way the chord is modeled.
"""

from typing import Any

import numpy as np
import warp as wp


def require_no_history(model) -> None:
    """Both domains call laws without history so far (linear truss, springs).

    A law with history (a superelastic chord or strut, say) needs committed
    and trial state arrays on the domain, swapped on convergence as
    materials/uniaxial.py describes. That is reserved, so such a law is
    refused here and not run with its history ignored.
    """
    if tuple(model.initial_state) != (0.0,):
        raise NotImplementedError(f"{model.name}: uniaxial laws with history are not yet supported on line elements")


@wp.kernel
def _reference_length(
    X: wp.array2d(dtype=wp.vec3d),  # (E, 2) reference positions
    length: wp.array(dtype=wp.float64),  # (E,)
):
    i = wp.tid()
    length[i] = wp.length(X[i, 1] - X[i, 0])


def geometric_length(X: wp.array) -> np.ndarray:
    """Distance between the two reference nodes of each element, computed on
    the device of `X` with the arithmetic of the update kernels.

    FEBio computes the initial and the current length with the same code, so
    an undeformed element has a stretch of exactly 1, and a tension-only law
    is active there (extension >= 0). The same has to hold here. A rest length
    from NumPy on the host does not give it: on CUDA the kernel's length
    differs from NumPy's in the last bit for about one element in nine, and
    half of those would start slack, with no stiffness in the first iteration.
    """
    out = wp.zeros(X.shape[0], dtype=wp.float64, device=X.device)
    wp.launch(_reference_length, dim=X.shape[0], inputs=[X], outputs=[out], device=X.device)
    return out.numpy()


@wp.func
def axial_stiffness_block(N: Any, dN_dl: Any, e: Any, l: Any):
    """K = (dN/dl - N/l) e (x) e + (N/l) I for one two-node axial element."""
    ratio = N / l
    return (dN_dl - ratio) * wp.outer(e, e) + ratio * wp.identity(n=3, dtype=N.dtype)


@wp.kernel
def axial_internal_force(
    emap: wp.array2d(dtype=wp.int32),  # (E, 2) entry of each node
    axial_force: wp.array(dtype=wp.float64),  # (E,) N, tension positive
    axis: wp.array(dtype=wp.vec3d),  # (E,) unit vector from node a to node b
    residual: wp.array(dtype=wp.vec3d),
):
    """residual -= internal force (FEBio's sign, R = external - internal).

    A chord in tension pulls its two nodes toward each other: node a receives
    +N e and node b receives -N e in the residual, as
    FELinearTrussDomain::ElementInternalForces and
    FEDiscreteElasticDomain::InternalForces assemble them. Written for reverse
    mode: it only reads N and e and adds to the residual.
    """
    i = wp.tid()
    f = axial_force[i] * axis[i]
    wp.atomic_add(residual, emap[i, 0], f)
    wp.atomic_sub(residual, emap[i, 1], f)


@wp.kernel
def axial_stiffness(
    emap: wp.array2d(dtype=wp.int32),
    axial_force: wp.array(dtype=wp.float64),  # (E,) N
    axial_tangent: wp.array(dtype=wp.float64),  # (E,) dN/dl
    axis: wp.array(dtype=wp.vec3d),
    length: wp.array(dtype=wp.float64),  # (E,) current length l
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    vals: wp.array(dtype=wp.mat33d),
):
    """Four 3x3 blocks per element, at fixed positions. Not taped."""
    i = wp.tid()
    K = axial_stiffness_block(axial_force[i], axial_tangent[i], axis[i], length[i])
    a = emap[i, 0]
    b = emap[i, 1]
    t = 4 * i
    rows[t] = a
    cols[t] = a
    vals[t] = K
    rows[t + 1] = a
    cols[t + 1] = b
    vals[t + 1] = -K
    rows[t + 2] = b
    cols[t + 2] = a
    vals[t + 2] = -K
    rows[t + 3] = b
    cols[t + 3] = b
    vals[t + 3] = K


@wp.kernel
def sum_axial_force(
    axial_force: wp.array(dtype=wp.float64),
    weight: wp.array(dtype=wp.float64),
    total: wp.array(dtype=wp.float64),
):
    """total[0] += sum_i weight_i N_i: a chord tension output as a kernel.

    An output used as an objective has to be a kernel, so a tape can
    differentiate it (decision O4 of docs/design/output.md). The weights pick
    the chords: one chord, a chord group, or all chords at a papillary tip.
    """
    i = wp.tid()
    wp.atomic_add(total, 0, weight[i] * axial_force[i])
