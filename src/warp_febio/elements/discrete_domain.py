"""Discrete domain: two-node springs as a term.

DRAFT (chordae elements, November 2026). Structure for discussion, not final.

FEBio counterpart
    FEDiscreteElasticDomain ("discrete") with a discrete elastic material
    ("linear spring", "tension-only linear spring", "nonlinear spring"):
    Update, InternalForces, StiffnessMatrix.
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEDiscreteElasticDomain.cpp
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FESpringMaterial.cpp
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FENonlinearSpring.cpp

Same as FEBio
    The force is along the current axis of the spring, F e, with F the
    material's function of the elongation dl = L - L0 (or of the strain or
    the stretch, for a nonlinear spring with that `measure`). The stiffness
    block is (E - F / L) e (x) e + (F / L) I with E = dF/dL, and a spring of
    zero current length gets FEBio's guard (F = 0, L = 1, e = (1, 1, 1)).
    Springs add no mass. Outputs: "discrete element force" (the vector F e),
    "discrete element signed force" (F), "discrete element stretch",
    "discrete element elongation".

Different, and why
    FEBio makes one domain per <discrete> entry. Here one domain is a batch
    of all springs that share a kernel, that is, a law with the same code:
    the eight chord groups of the open mitral input are one launch. Each
    spring has the index of its discrete set, and the material parameters are
    one row per set. A set is still addressed as FEBio addresses it, by its
    name: `parameters()` lists the parameter row and the rest lengths of each
    set under "<set name>.material" and "<set name>.rest_length", as views of
    the domain's arrays, and `plot` can return the springs of one set. A
    gradient with respect to a set's row is the gradient for that chord
    group.

    FEBio takes the rest length L0 from the reference node positions. Here it
    is an array per spring that defaults to that length, so a chord or
    neochord length can be a design variable without moving nodes. It carries
    gradients, as the parameter rows do.

Not implemented
    Springs attached to rigid bodies (FEBio's domain carries the rigid
    rotation degrees of freedom), "discrete composite", "Hill", "torsion
    spring", "experimental spring", deformable springs.
"""

# No `from __future__ import annotations` here: the kernels below are built in a
# closure and Warp evaluates their annotations (pvec) when they are made.
from typing import Any

import numpy as np
import warp as wp

from ..assembly.system import BlockTangent, State, Term
from ..materials.status import NON_FINITE, OK
from ..materials.uniaxial import UniaxialModel
from .line2 import axial_internal_force, axial_stiffness, require_no_history

vec3d = wp.vec3d
mat33d = wp.mat33d

_KERNELS = {}


def _make_update(model: UniaxialModel):
    """The update kernel of a discrete domain for one spring law."""
    if model in _KERNELS:
        return _KERNELS[model]
    require_no_history(model)
    law = model.update
    pvec = model.param_vector(wp.float64)
    measure = model.option("measure", "elongation")
    is_strain = measure == "strain"
    is_stretch = measure == "stretch"

    def update(
        X: wp.array2d(dtype=vec3d),  # (E, 2) reference positions
        emap: wp.array2d(dtype=wp.int32),  # (E, 2) entry of each node
        u: wp.array(dtype=vec3d),  # solution, one 3-vector per entry
        rest_length: wp.array(dtype=wp.float64),  # (E,) L0
        region: wp.array(dtype=wp.int32),  # (E,) discrete set of each spring
        params: wp.array(dtype=pvec),  # one parameter row per discrete set
        length: wp.array(dtype=wp.float64),  # (E,) L, or 1 for a collapsed spring
        axis: wp.array(dtype=vec3d),  # (E,) e
        measure: wp.array(dtype=wp.float64),  # (E,) FEBio's deformation measure s
        axial_force: wp.array(dtype=wp.float64),  # (E,) F, tension positive
        axial_tangent: wp.array(dtype=wp.float64),  # (E,) dF/dL
        status: wp.array(dtype=wp.int32),
    ):
        """Kinematics, law and force of one spring: FEDiscreteElasticDomain::Update
        with FENonlinearSpringMaterial::Force and ::Stiffness."""
        i = wp.tid()
        zero = wp.float64(0.0)
        one = wp.float64(1.0)
        r = (X[i, 1] + u[emap[i, 1]]) - (X[i, 0] + u[emap[i, 0]])
        L = wp.length(r)
        L0 = rest_length[i]
        s = L - L0
        ds = one
        if wp.static(is_strain):
            s = (L - L0) / L0
            ds = one / L0
        if wp.static(is_stretch):
            s = L / L0
            ds = one / L0
        F, dF, unused = law(s, params[region[i]], wp.vector(zero, length=1))
        # FEBio: if (Lt == 0) { F = 0; Lt = 1; e = vec3d(1, 1, 1); }
        collapsed = L == zero
        Lg = wp.where(collapsed, one, L)
        e = wp.where(collapsed, vec3d(one, one, one), r / Lg)
        F = wp.where(collapsed, zero, F)
        if wp.isfinite(F) and wp.isfinite(dF):
            length[i] = Lg
            axis[i] = e
            measure[i] = s
            axial_force[i] = F
            axial_tangent[i] = dF * ds
            status[i] = wp.static(OK)
        else:
            status[i] = wp.static(NON_FINITE)

    kernel = wp.kernel(update, module="unique")
    _KERNELS[model] = kernel
    return kernel


class DiscreteDomain(Term):
    """All springs of a model that share one law: FEBio "discrete" domains, batched.

    Args:
        name: Name of the term.
        model: The spring law (a uniaxial model whose strain is FEBio's
            deformation measure and whose stress is the spring force).
        X: Reference positions of the two nodes of each spring, shape (E, 2, 3).
        emap: Entry of the `displacement` variable at each node, shape (E, 2).
        params: One parameter row per discrete set, in the order of
            `model.param_names`, shape (S, P).
        region: Discrete set of each spring, shape (E,). All zero unless given.
        set_names: FEBio names of the discrete sets, in row order.
        n_entries: Number of entries of the field table.
        device: Warp device.
        rest_length: Rest length per spring. The distance between the
            reference node positions unless given, as in FEBio.
    """

    internal = True
    febio_type = "discrete"
    element_type = "discrete"

    def __init__(self, name, model, X, emap, params, n_entries, device, region=None, set_names=(), rest_length=None):
        self.name = name
        self.law = model
        self.device = device
        self.n = n_entries
        self.count = E = len(X)
        self.set_names = tuple(set_names)
        self._update = _make_update(model)
        self.pvec = model.param_vector(wp.float64)

        X = np.asarray(X, dtype=np.float64)
        geometric = np.linalg.norm(X[:, 1] - X[:, 0], axis=1)
        dev = dict(device=device)
        grad_on = dict(requires_grad=True, **dev)
        self.X = wp.array(X, dtype=vec3d, **dev)
        self.emap = wp.array(np.asarray(emap, dtype=np.int32), dtype=wp.int32, **dev)
        self.region = wp.array(
            np.zeros(E, np.int32) if region is None else np.asarray(region, np.int32), dtype=wp.int32, **dev
        )
        # Differentiable parameters.
        self.rest_length = wp.array(geometric if rest_length is None else rest_length, dtype=wp.float64, **grad_on)
        self.params = wp.array(np.atleast_2d(np.asarray(params, dtype=np.float64)), dtype=self.pvec, **grad_on)
        # Element data of the last update.
        self.data = self._new_data(requires_grad=True)
        self.status = wp.zeros(E, dtype=wp.int32, **dev)
        self.block_count = 4 * E
        # Springs of a set are consecutive when the domain is built from a
        # FEBio input. Then each set has a slice of the arrays, and its
        # parameters can be named by the set.
        self.set_slices = self._slices_by_set(self.region.numpy(), self.set_names)

    #: `parameters()` returns full names ("<set name>.material"); the system
    #: does not put the term's name in front of them.
    qualified_parameters = True

    @staticmethod
    def _slices_by_set(region, set_names) -> dict:
        slices, start = {}, 0
        for k, name in enumerate(set_names):
            count = int((region == k).sum())
            if count == 0 or not (region[start : start + count] == k).all():
                return {}
            slices[name] = (k, start, start + count)
            start += count
        return slices if start == len(region) else {}

    def _new_data(self, requires_grad: bool) -> dict:
        opts = dict(device=self.device, requires_grad=requires_grad)
        E = self.count
        return dict(
            length=wp.zeros(E, dtype=wp.float64, **opts),
            axis=wp.zeros(E, dtype=vec3d, **opts),
            measure=wp.zeros(E, dtype=wp.float64, **opts),
            axial_force=wp.zeros(E, dtype=wp.float64, **opts),
            axial_tangent=wp.zeros(E, dtype=wp.float64, **opts),
        )

    def _launch_update(self, state: State, data: dict, status: wp.array) -> None:
        wp.launch(
            self._update,
            dim=self.count,
            inputs=[self.X, self.emap, state.u, self.rest_length, self.region, self.params],
            outputs=[
                data["length"],
                data["axis"],
                data["measure"],
                data["axial_force"],
                data["axial_tangent"],
                status,
            ],
            device=self.device,
        )

    # -- Term ----------------------------------------------------------------

    def update(self, state: State) -> int:
        self._launch_update(state, self.data, self.status)
        return int(self.status.numpy().max()) if self.count else OK

    def residual(self, state: State, history: Any, out: wp.array) -> None:
        wp.launch(
            axial_internal_force,
            dim=self.count,
            inputs=[self.emap, self.data["axial_force"], self.data["axis"]],
            outputs=[out],
            device=self.device,
        )

    def linearize(self, state: State, history: Any, exact: bool = False) -> BlockTangent:
        # The block is the derivative of the force for a law that returns its
        # exact slope, which the laws here do (see nonlinear_spring.py on the
        # one difference from FEBio's finite-difference slope).
        d = self.data
        wp.launch(
            axial_stiffness,
            dim=self.count,
            inputs=[self.emap, d["axial_force"], d["axial_tangent"], d["axis"], d["length"]],
            outputs=[self.rows, self.cols, self.vals],
            device=self.device,
        )
        return BlockTangent(self.n, self.rows, self.cols, self.vals)

    def parameters(self) -> dict:
        """Differentiable arrays by FEBio's discrete set names. Each entry is
        a view of the domain's own array, so values and gradients are shared
        with it. Without set names (a domain built in code), the whole
        arrays under the term's name."""
        if not self.set_slices:
            return {f"{self.name}.material": self.params, f"{self.name}.rest_length": self.rest_length}
        out = {}
        for name, (row, start, end) in self.set_slices.items():
            out[f"{name}.material"] = self.params[row : row + 1]
            out[f"{name}.rest_length"] = self.rest_length[start:end]
        return out

    # -- beyond Term ---------------------------------------------------------

    @property
    def parameter_names(self) -> tuple:
        return tuple(self.law.param_names)

    def axial_force(self, state: State) -> wp.array:
        """Signed spring force per spring at `state` (tension positive), in a
        new array. For outputs and objectives; see TrussDomain.axial_force."""
        data = self._new_data(requires_grad=True)
        self._launch_update(state, data, wp.zeros(self.count, dtype=wp.int32, device=self.device))
        return data["axial_force"]

    def plot_by_set(self, variable: str) -> dict:
        """`plot(variable)` split by discrete set: {set name: array}."""
        values = self.plot(variable)
        return {name: values[start:end] for name, (_, start, end) in self.set_slices.items()}

    def plot(self, variable: str) -> np.ndarray:
        """Host copy of a plot variable at the last update, by FEBio's name
        (FEPlotDiscreteElement* in FEBioMechPlot.cpp)."""
        d = self.data
        if variable == "discrete element signed force":
            return d["axial_force"].numpy().copy()
        if variable == "discrete element force":
            return d["axial_force"].numpy()[:, None] * d["axis"].numpy()
        if variable == "discrete element length":
            return d["length"].numpy().copy()
        if variable == "discrete element elongation":
            return d["length"].numpy() - self.rest_length.numpy()
        if variable == "discrete element stretch":
            return d["length"].numpy() / self.rest_length.numpy()
        raise KeyError(variable)
