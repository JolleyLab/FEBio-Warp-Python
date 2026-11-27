"""Material model architecture (draft v0.3; see docs/design/materials.md).

What a constitutive model is here
---------------------------------
A set of pure functions of four inputs, evaluated at one material point:

    F      deformation gradient, F_iJ = dx_i / dX_J (dimensionless)
    g      geometry vector: reference-configuration data the model needs at this
           point, such as material axes and fiber directions. Its layout is
           declared by the model (`geometry`). Models that need none ignore it.
    p      parameter vector: the floating-point constitutive parameters, in the
           order given by `param_names`, with FEBio XML tag names where a FEBio
           counterpart exists.
    state  history variables, for path-dependent models only (see uniaxial.py
           for the first one).

Solid models return, in the current configuration (updated Lagrangian), the same
quantities a FEBio material returns (FEElasticMaterial::Stress and ::Tangent,
https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticMaterial.h):

    energy(F, g, p)                     -> scalar  strain energy per reference volume
    cauchy(F, g, p)                     -> mat33   Cauchy stress (symmetric)
    spatial_tangent_action(F, d, g, p)  -> mat33   c : d for symmetric d

How this maps to FEBio
----------------------
FEBio composes materials at run time from objects: a material owns parameters
of several types, a material axis, optional fiber properties, and possibly
child materials (solid mixture). Warp kernels cannot dispatch at run time, so
the same structure is expressed when the kernel is built:

    FEBio                                   here
    --------------------------------------  -----------------------------------
    FEParamDouble (constant, mapped, or     entries of p; p is an array, so a
      driven by a load curve)                 value can differ by region or
                                              element and can change in time
                                              (schedule.py)
    int / bool / enum parameters            `options` of the model, fixed when
      (for example pressure_model)            the kernel is built
    mat_axis, fiber, other valuators        named fields of g, evaluated once at
                                              setup into per-point arrays
    density                                 `Material.density`, a domain property
                                              that never enters the kernels here
    solid mixture                           compose.solid_mixture(...)
    uncoupled material + U(J)               formulation="uncoupled" plus
                                              volumetric.py, combined by a
                                              multi-field element
    material point history                  committed / trial state arrays

Everything is generic in the float type (functions use F.dtype for constants),
so one source compiles for FP32 and FP64.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import warp as wp

from .status import NEGATIVE_JACOBIAN, NON_FINITE, OK

# Identity material axes, flattened row by row. Used when a model declares a
# "mat_axis" field and the caller supplies none.
IDENTITY_AXES = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class GeometryField:
    """One named block of the geometry vector g.

    FEBio keeps material axes (mat_axis) and fiber directions (fiber) as
    separate properties, and a material may have several fibers. Each becomes
    its own named field here, so a fiber is never assumed to be a column of the
    material axes.
    """

    name: str  # for example "mat_axis" or "fiber"
    size: int  # number of floats: 9 for a frame, 3 for a direction


@dataclass(frozen=True)
class ParamRange:
    """Admissible range of one float parameter.

    Mirrors FEBio's FE_RANGE validators on ADD_PARAMETER
    (https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEParamValidator.h),
    so input that FEBio would reject at initialization is rejected here when the
    parameter vector is packed, not later inside a kernel. A bound of None
    means unbounded on that side.
    """

    name: str
    low: Optional[float] = None
    high: Optional[float] = None
    low_closed: bool = True  # True: value >= low is allowed; False: value > low
    high_closed: bool = True

    def check(self, value):
        if self.low is not None and (value < self.low or (value == self.low and not self.low_closed)):
            return False
        if self.high is not None and (value > self.high or (value == self.high and not self.high_closed)):
            return False
        return bool(np.isfinite(value))

    def describe(self):
        left = "" if self.low is None else f"{self.low} {'<=' if self.low_closed else '<'} "
        right = "" if self.high is None else f" {'<=' if self.high_closed else '<'} {self.high}"
        return f"{left}{self.name}{right}"


@dataclass(frozen=True)
class MaterialModel:
    """Descriptor for one constitutive model. Holds functions, not data."""

    name: str  # registry key
    febio_type: str  # FEBio <material type="..."> it reproduces, or "" if none
    param_names: tuple  # order of the entries of p
    energy: Callable
    cauchy: Callable  # consumed by elements
    spatial_tangent_action: Callable  # consumed by elements
    pk1: Callable  # reference-frame equivalent, derived or authored
    tangent_action: Callable  # dP[H], derived or authored
    geometry: tuple = ()  # GeometryField entries, in the order they sit in g
    param_ranges: tuple = ()  # ParamRange entries; parameters not listed are unchecked
    # Build-time choices (FEBio int / bool / enum parameters). They select code,
    # so they are fixed when the kernel is built and are not differentiable.
    options: tuple = ()  # ((name, value), ...)
    # "coupled" (unconstrained; use with displacement elements) or "uncoupled"
    # (deviatoric part only; U(J) comes from volumetric.py and the two are
    # combined by a multi-field element). FEBio practice, per Jeff Weiss: use
    # uncoupled only for near-incompressibility.
    formulation: str = "coupled"
    authored: str = "spatial"  # "spatial", "reference" or "composed"
    source: str = ""  # link to the FEBio source this model follows
    # Optional: Cauchy stress and the full spatial tangent (6x6, FEBio tens4ds
    # order) in one pass, sharing intermediate quantities. Assembled-stiffness
    # elements and the exchange kernel use it when present; otherwise the 6x6
    # is built from six tangent actions. The two must agree, and the
    # conformance checks compare them.
    stress_and_tangent: Optional[Callable] = None

    @property
    def geometry_size(self):
        """Length of g. Never zero: geometry-free models get one unused entry."""
        return max(1, sum(f.size for f in self.geometry))

    def param_vector(self, dtype=wp.float64):
        return wp.types.vector(length=len(self.param_names), dtype=dtype)

    def geometry_vector(self, dtype=wp.float64):
        return wp.types.vector(length=self.geometry_size, dtype=dtype)

    def pack(self, dtype=wp.float64, **values):
        """Parameter vector from named values; rejects typos and omissions."""
        missing = set(self.param_names) - set(values)
        extra = set(values) - set(self.param_names)
        if missing or extra:
            raise ValueError(f"{self.name}: missing {sorted(missing)}, unknown {sorted(extra)}")
        self.validate(values)
        return self.param_vector(dtype)(*[values[k] for k in self.param_names])

    def validate(self, values):
        """Reject parameter values outside the ranges FEBio enforces."""
        bad = [
            f"{r.describe()} (got {values[r.name]})"
            for r in self.param_ranges
            if r.name in values and not r.check(values[r.name])
        ]
        if bad:
            raise ValueError(f"{self.name}: parameter out of range: " + "; ".join(bad))

    def pack_geometry(self, count, **fields):
        """Geometry array of shape (count, geometry_size) from named fields.

        Each field is an array of shape (count, size) or (size,) to repeat at
        every point. A "mat_axis" field that is not supplied defaults to the
        identity. Any other missing field is an error.
        """
        out = np.zeros((count, self.geometry_size))
        unknown = set(fields) - {f.name for f in self.geometry}
        if unknown:
            raise ValueError(f"{self.name}: unknown geometry fields {sorted(unknown)}")
        offset = 0
        for f in self.geometry:
            if f.name in fields:
                value = np.asarray(fields[f.name], dtype=np.float64).reshape(-1, f.size)
            elif f.name.endswith("mat_axis"):
                value = np.asarray(IDENTITY_AXES).reshape(1, 9)
            else:
                raise ValueError(f"{self.name}: geometry field {f.name!r} is required")
            out[:, offset : offset + f.size] = value
            offset += f.size
        return out


@dataclass
class Material:
    """A model with its data, as assigned to a domain.

    This is the object an input reader produces from one FEBio <material>
    element. `density` is a FEBio material parameter, but no constitutive
    routine uses it; domains read it from here for dynamic analyses.
    """

    model: MaterialModel
    params: dict
    density: float = 0.0
    schedules: dict = field(default_factory=dict)  # parameter name -> LoadCurve

    def params_at(self, time):
        """Parameter values at a given time, with any load curves applied."""
        values = dict(self.params)
        for name, curve in self.schedules.items():
            # FEBio semantics: the load curve value multiplies the parameter.
            values[name] = self.params[name] * curve(time)
        return values


def from_spatial(name, febio_type, param_names, energy, cauchy, spatial_tangent_action, **kw):
    """Build a model from FEBio-style spatial functions.

    The author supplies the Cauchy stress and the spatial tangent contracted
    with a symmetric d. First Piola stress and its directional derivative are
    derived here for gradient checks and for MFEM's HyperelasticModel convention
    (https://github.com/mfem/mfem/blob/master/fem/nonlininteg.hpp).
    """

    @wp.func
    def pk1(F: Any, g: Any, p: Any):
        # P = J sigma F^-T (Piola transformation of the Cauchy stress)
        return wp.determinant(F) * cauchy(F, g, p) * wp.transpose(wp.inverse(F))

    @wp.func
    def tangent_action(F: Any, H: Any, g: Any, p: Any):
        Finv = wp.inverse(F)
        FinvT = wp.transpose(Finv)
        J = wp.determinant(F)
        # A perturbation H of F is a spatial velocity gradient l = H F^-1,
        # with symmetric part d.
        l = H * Finv
        d = (l + wp.transpose(l)) * F.dtype(0.5)
        tau = J * cauchy(F, g, p)  # Kirchhoff stress
        # c is defined through the Lie (Truesdell) rate of tau:
        #     d(tau) = J c:d + l tau + tau l^T
        # and since P = tau F^-T,
        #     dP = d(tau) F^-T - tau F^-T H^T F^-T
        dtau = J * spatial_tangent_action(F, d, g, p) + l * tau + tau * wp.transpose(l)
        return dtau * FinvT - tau * FinvT * wp.transpose(H) * FinvT

    kw.setdefault("authored", "spatial")
    return MaterialModel(
        name,
        febio_type,
        tuple(param_names),
        energy=energy,
        cauchy=cauchy,
        spatial_tangent_action=spatial_tangent_action,
        pk1=pk1,
        tangent_action=tangent_action,
        **kw,
    )


def from_reference(name, febio_type, param_names, energy, pk1, tangent_action, **kw):
    """Build a model from reference-frame functions (energy, P, dP[H]).

    Produces the Cauchy stress and c : d that elements consume. Inverse of
    `from_spatial`.
    """

    @wp.func
    def cauchy(F: Any, g: Any, p: Any):
        # sigma = P F^T / J
        return pk1(F, g, p) * wp.transpose(F) / wp.determinant(F)

    @wp.func
    def spatial_tangent_action(F: Any, d: Any, g: Any, p: Any):
        # Choose the perturbation H = d F, whose velocity gradient is l = d.
        # With tau = P F^T:   d(tau) = dP[H] F^T + P H^T
        # and the definition  d(tau) = J c:d + d tau + tau d   gives c : d.
        Ft = wp.transpose(F)
        P = pk1(F, g, p)
        tau = P * Ft
        H = d * F
        dtau = tangent_action(F, H, g, p) * Ft + P * wp.transpose(H)
        return (dtau - d * tau - tau * d) / wp.determinant(F)

    kw.setdefault("authored", "reference")
    return MaterialModel(
        name,
        febio_type,
        tuple(param_names),
        energy=energy,
        cauchy=cauchy,
        spatial_tangent_action=spatial_tangent_action,
        pk1=pk1,
        tangent_action=tangent_action,
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
    # Keyed by the model object, not its name: two models that happen to share
    # a name (two mixtures built with the same label, say) must not share a kernel.
    key = (model, dtype)
    if key in _POINT_KERNELS:
        return _POINT_KERNELS[key]
    mat = wp.types.matrix(shape=(3, 3), dtype=dtype)
    pvec = model.param_vector(dtype)
    gvec = model.geometry_vector(dtype)
    energy, cauchy, spatial_tangent_action = (
        model.energy,
        model.cauchy,
        model.spatial_tangent_action,
    )
    pk1, tangent_action = model.pk1, model.tangent_action

    def point(
        F: wp.array(dtype=mat),
        H: wp.array(dtype=mat),
        g: wp.array(dtype=gvec),
        region: wp.array(dtype=int),
        params: wp.array(dtype=pvec),
        W: wp.array(dtype=dtype),
        sigma: wp.array(dtype=mat),
        cd: wp.array(dtype=mat),
        P: wp.array(dtype=mat),
        dP: wp.array(dtype=mat),
        status: wp.array(dtype=int),
    ):
        i = wp.tid()
        Fi = F[i]
        # Failure rule, step 1 (see status.py): no material call for J <= 0.
        # Outputs keep the zeros they were allocated with.
        if wp.determinant(Fi) <= dtype(0.0):
            status[i] = wp.static(NEGATIVE_JACOBIAN)
            return
        # Parameters are an array indexed by region, so they can differ per
        # element and can carry gradients on a wp.Tape. Geometry is per point.
        p = params[region[i]]
        gi = g[i]
        # H is a perturbation of F. Its rate of deformation is d = sym(H F^-1).
        l = H[i] * wp.inverse(Fi)
        d = (l + wp.transpose(l)) * dtype(0.5)
        energy_i = energy(Fi, gi, p)
        sigma_i = cauchy(Fi, gi, p)  # consumed form
        cd_i = spatial_tangent_action(Fi, d, gi, p)  # consumed form
        # Failure rule, step 2: a non-finite result is reported, not written.
        if not (wp.isfinite(energy_i) and wp.isfinite(sigma_i) and wp.isfinite(cd_i)):
            status[i] = wp.static(NON_FINITE)
            return
        W[i] = energy_i
        sigma[i] = sigma_i
        cd[i] = cd_i
        P[i] = pk1(Fi, gi, p)  # reference-frame equivalent
        dP[i] = tangent_action(Fi, H[i], gi, p)
        status[i] = wp.static(OK)

    kernel = wp.kernel(point, module="unique")
    _POINT_KERNELS[key] = kernel
    return kernel


def evaluate(model, F, params, H=None, geometry=None, dtype=wp.float64, device="cpu"):
    """Host convenience for tests and small studies. Not a production path.

    F, H: arrays of shape (n, 3, 3). params: dict of named values. geometry:
    dict of named fields for model.pack_geometry, or None.
    Returns NumPy arrays: energy (n,), cauchy (n,3,3), spatial_tangent_action
    (n,3,3; c : d with d = sym(H F^-1)), pk1 (n,3,3), tangent_action (n,3,3),
    and status (n,), the per-point codes of status.py. Failed points hold zeros.
    This function does not raise on failure; call status.raise_on_failure.
    """
    F = np.asarray(F, dtype=np.float64)
    n = len(F)
    H = np.zeros_like(F) if H is None else np.asarray(H, dtype=np.float64)
    g = model.pack_geometry(n, **(geometry or {}))
    mat = wp.types.matrix(shape=(3, 3), dtype=dtype)
    W = wp.zeros(n, dtype=dtype, device=device)
    sigma = wp.zeros(n, dtype=mat, device=device)
    cd = wp.zeros(n, dtype=mat, device=device)
    P = wp.zeros(n, dtype=mat, device=device)
    dP = wp.zeros(n, dtype=mat, device=device)
    status = wp.zeros(n, dtype=int, device=device)
    wp.launch(
        make_point_kernel(model, dtype),
        dim=n,
        inputs=[
            wp.array(F, dtype=mat, device=device),
            wp.array(H, dtype=mat, device=device),
            wp.array(g, dtype=model.geometry_vector(dtype), device=device),
            wp.zeros(n, dtype=int, device=device),
            wp.array(
                [model.pack(dtype, **params)],
                dtype=model.param_vector(dtype),
                device=device,
            ),
        ],
        outputs=[W, sigma, cd, P, dP, status],
        device=device,
    )
    return {
        "status": status.numpy(),
        "energy": W.numpy(),
        "cauchy": sigma.numpy(),
        "spatial_tangent_action": cd.numpy(),
        "pk1": P.numpy(),
        "tangent_action": dP.numpy(),
    }
