"""Common material interface (proposal for discussion; see docs/MATERIALS.md).

One form consumed by elements, two ways to write a model.

Consumed by elements (current configuration, updated Lagrangian, as in FEBio):

    energy(F, Q, p)                     -> scalar  strain energy per reference volume
    cauchy(F, Q, p)                     -> mat33   Cauchy stress (symmetric)
    spatial_tangent_action(F, d, Q, p)  -> mat33   c : d for symmetric d, where c is
                                                   FEBio's spatial elasticity tensor

    F  deformation gradient, F_iJ = dx_i / dX_J (dimensionless)
    Q  local material frame; its columns are the FEBio mat_axis vectors.
       Isotropic models accept it and ignore it so every model has one signature.
    p  fixed-length parameter vector, in the order given by `param_names`,
       using the FEBio XML tag names where a FEBio counterpart exists.

These are the quantities FEBio materials return (FEElasticMaterial::Stress and
::Tangent, https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticMaterial.h),
so element residuals and stiffness can be compared with FEBio term by term, and
contact and boundary conditions that live in the current configuration see the
same stress measure. This follows Jeff Weiss's guidance on the first draft,
which consumed first Piola stress instead.

Writing a model, form A (spatial, the FEBio form): supply the three functions
above and call `from_spatial`.
Writing a model, form B (reference): supply energy(F, Q, p), pk1(F, Q, p) and
tangent_action(F, H, Q, p) = dP[F][H] and call `from_reference`.

Whichever way a model is written, both sets of functions are available on the
descriptor. The reference-frame pair (pk1, tangent_action) is kept for gradient
checks and for MFEM's HyperelasticModel convention (EvalW, EvalP):
https://github.com/mfem/mfem/blob/master/fem/nonlininteg.hpp

All functions are generic in the float type (they use F.dtype for constants),
so one source compiles for FP32 and FP64.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import warp as wp


@dataclass(frozen=True)
class MaterialModel:
    """Descriptor for one constitutive model. Holds functions, not state."""

    name: str  # FEBio-Warp name, used as the registry key
    febio_type: str  # FEBio <material type="..."> it reproduces, or "" if none
    param_names: tuple  # order of the entries of the parameter vector
    energy: Callable
    cauchy: Callable  # consumed by elements
    spatial_tangent_action: Callable  # consumed by elements
    pk1: Callable  # reference-frame equivalent, derived or authored
    tangent_action: Callable  # dP[H], derived or authored
    # "coupled" (unconstrained; use with displacement elements) or "uncoupled"
    # (deviatoric + U(J); only for near-incompressibility and only together with
    # a multi-field element, per FEBio practice).
    formulation: str = "coupled"
    uses_frame: bool = False  # True when Q changes the response (fiber models)
    state_size: int = 0  # history variables per point; 0 means elastic
    update: Optional[Callable] = None  # reserved for path-dependent models
    authored: str = "spatial"  # "spatial" or "reference": how the model was written
    source: str = ""  # link to the FEBio source this model follows

    def param_vector(self, dtype=wp.float64):
        """Warp vector type that holds this model's parameters."""
        return wp.types.vector(length=len(self.param_names), dtype=dtype)

    def pack(self, dtype=wp.float64, **values):
        """Build a parameter vector from named values; rejects typos and omissions."""
        missing = set(self.param_names) - set(values)
        extra = set(values) - set(self.param_names)
        if missing or extra:
            raise ValueError(
                f"{self.name}: missing {sorted(missing)}, unknown {sorted(extra)}"
            )
        return self.param_vector(dtype)(*[values[k] for k in self.param_names])


def from_spatial(
    name, febio_type, param_names, energy, cauchy, spatial_tangent_action, **kw
):
    """Build the consumed form out of FEBio-style spatial functions.

    FEBio materials return the Cauchy stress (FEElasticMaterial::Stress) and the
    spatial elasticity tensor c (FEElasticMaterial::Tangent). The author supplies
    those two, the tangent already contracted with a symmetric d, and this
    function derives first Piola stress and its directional derivative. No
    per-material algebra is needed.
    """

    @wp.func
    def pk1(F: Any, Q: Any, p: Any):
        # P = J sigma F^-T (Piola transformation of the Cauchy stress)
        return wp.determinant(F) * cauchy(F, Q, p) * wp.transpose(wp.inverse(F))

    @wp.func
    def tangent_action(F: Any, H: Any, Q: Any, p: Any):
        Finv = wp.inverse(F)
        FinvT = wp.transpose(Finv)
        J = wp.determinant(F)
        # A perturbation H of F is a spatial velocity gradient l = H F^-1,
        # with symmetric part d.
        l = H * Finv
        d = (l + wp.transpose(l)) * F.dtype(0.5)
        tau = J * cauchy(F, Q, p)  # Kirchhoff stress
        # c is defined through the Lie (Truesdell) rate of tau:
        #     d(tau) = J c:d + l tau + tau l^T
        # and since P = tau F^-T,
        #     dP = d(tau) F^-T - tau F^-T H^T F^-T
        dtau = (
            J * spatial_tangent_action(F, d, Q, p) + l * tau + tau * wp.transpose(l)
        )
        return dtau * FinvT - tau * FinvT * wp.transpose(H) * FinvT

    return MaterialModel(
        name,
        febio_type,
        tuple(param_names),
        energy=energy,
        cauchy=cauchy,
        spatial_tangent_action=spatial_tangent_action,
        pk1=pk1,
        tangent_action=tangent_action,
        authored="spatial",
        **kw,
    )


def from_reference(name, febio_type, param_names, energy, pk1, tangent_action, **kw):
    """Build the consumed (spatial) form out of reference-frame functions.

    The author supplies first Piola stress and its directional derivative,
    typically derived from the energy. This function produces the Cauchy stress
    and c : d that elements consume. It is the inverse of `from_spatial`.
    """

    @wp.func
    def cauchy(F: Any, Q: Any, p: Any):
        # sigma = P F^T / J
        return pk1(F, Q, p) * wp.transpose(F) / wp.determinant(F)

    @wp.func
    def spatial_tangent_action(F: Any, d: Any, Q: Any, p: Any):
        # Choose the perturbation H = d F, whose velocity gradient is l = d.
        # With tau = P F^T:   d(tau) = dP[H] F^T + P H^T
        # and the definition  d(tau) = J c:d + d tau + tau d   gives c : d.
        Ft = wp.transpose(F)
        P = pk1(F, Q, p)
        tau = P * Ft
        H = d * F
        dtau = tangent_action(F, H, Q, p) * Ft + P * wp.transpose(H)
        return (dtau - d * tau - tau * d) / wp.determinant(F)

    return MaterialModel(
        name,
        febio_type,
        tuple(param_names),
        energy=energy,
        cauchy=cauchy,
        spatial_tangent_action=spatial_tangent_action,
        pk1=pk1,
        tangent_action=tangent_action,
        authored="reference",
        **kw,
    )


# Registry. Lets an input reader go from the FEBio <material type="..."> string
# to a model, and lets the conformance test loop over every model.
_REGISTRY = {}


def register(model: MaterialModel):
    if model.name in _REGISTRY:
        raise ValueError(f"material model {model.name!r} is already registered")
    _REGISTRY[model.name] = model
    return model


def get(name):
    return _REGISTRY[name]


def all_models():
    return list(_REGISTRY.values())


def by_febio_type(febio_type):
    for model in _REGISTRY.values():
        if model.febio_type == febio_type:
            return model
    raise KeyError(f"no FEBio-Warp model registered for FEBio type {febio_type!r}")


_POINT_KERNELS = {}


def make_point_kernel(model: MaterialModel, dtype=wp.float64):
    """Kernel that evaluates one model on a batch of material points.

    Warp has no virtual dispatch inside kernels. The model's functions are
    captured here and a kernel is generated per (model, precision), the same way
    warp.fem specializes integrands. Elements will do the same with their own
    quadrature loop and will call only energy, cauchy and spatial_tangent_action;
    this material-point kernel also returns the reference-frame pair so the two
    can be checked against each other.
    """
    key = (model.name, dtype)
    if key in _POINT_KERNELS:
        return _POINT_KERNELS[key]
    mat = wp.types.matrix(shape=(3, 3), dtype=dtype)
    vec = model.param_vector(dtype)
    energy, cauchy, spatial_tangent_action = (
        model.energy,
        model.cauchy,
        model.spatial_tangent_action,
    )
    pk1, tangent_action = model.pk1, model.tangent_action

    def point(
        F: wp.array(dtype=mat),
        H: wp.array(dtype=mat),
        Q: wp.array(dtype=mat),
        region: wp.array(dtype=int),
        params: wp.array(dtype=vec),
        W: wp.array(dtype=dtype),
        sigma: wp.array(dtype=mat),
        cd: wp.array(dtype=mat),
        P: wp.array(dtype=mat),
        dP: wp.array(dtype=mat),
    ):
        i = wp.tid()
        # Parameters are an array indexed by region, so they can differ per
        # element and can carry gradients on a wp.Tape.
        p = params[region[i]]
        Fi = F[i]
        # H is a perturbation of F. Its rate of deformation is d = sym(H F^-1).
        l = H[i] * wp.inverse(Fi)
        d = (l + wp.transpose(l)) * dtype(0.5)
        W[i] = energy(Fi, Q[i], p)
        sigma[i] = cauchy(Fi, Q[i], p)  # consumed form
        cd[i] = spatial_tangent_action(Fi, d, Q[i], p)  # consumed form
        P[i] = pk1(Fi, Q[i], p)  # reference-frame equivalent
        dP[i] = tangent_action(Fi, H[i], Q[i], p)

    kernel = wp.kernel(point, module="unique")
    _POINT_KERNELS[key] = kernel
    return kernel


def evaluate(model, F, params, H=None, Q=None, dtype=wp.float64, device="cpu"):
    """Host convenience for tests and small studies. Not a production path.

    F, H, Q: arrays of shape (n, 3, 3). params: dict of named values.
    Returns NumPy arrays: energy (n,), cauchy (n,3,3), spatial_tangent_action
    (n,3,3; c : d with d = sym(H F^-1)), pk1 (n,3,3), tangent_action (n,3,3).
    """
    F = np.asarray(F, dtype=np.float64)
    n = len(F)
    H = np.zeros_like(F) if H is None else np.asarray(H, dtype=np.float64)
    Q = np.tile(np.eye(3), (n, 1, 1)) if Q is None else np.asarray(Q, dtype=np.float64)
    mat = wp.types.matrix(shape=(3, 3), dtype=dtype)
    W = wp.zeros(n, dtype=dtype, device=device)
    sigma = wp.zeros(n, dtype=mat, device=device)
    cd = wp.zeros(n, dtype=mat, device=device)
    P = wp.zeros(n, dtype=mat, device=device)
    dP = wp.zeros(n, dtype=mat, device=device)
    wp.launch(
        make_point_kernel(model, dtype),
        dim=n,
        inputs=[
            wp.array(F, dtype=mat, device=device),
            wp.array(H, dtype=mat, device=device),
            wp.array(Q, dtype=mat, device=device),
            wp.zeros(n, dtype=int, device=device),
            wp.array(
                [model.pack(dtype, **params)],
                dtype=model.param_vector(dtype),
                device=device,
            ),
        ],
        outputs=[W, sigma, cd, P, dP],
        device=device,
    )
    return {
        "energy": W.numpy(),
        "cauchy": sigma.numpy(),
        "spatial_tangent_action": cd.numpy(),
        "pk1": P.numpy(),
        "tangent_action": dP.numpy(),
    }
