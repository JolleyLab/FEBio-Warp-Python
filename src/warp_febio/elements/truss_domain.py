"""Truss domain: two-node trusses of one part as a term.

DRAFT (chordae elements, November 2026). Structure for discussion, not final.
Static only: mass, inertia and body forces are described in
docs/design/chordae.md and wait for the dynamics work.

FEBio counterpart
    FELinearTrussDomain ("linear-truss") with a truss material ("linear
    truss"): Update, InternalForces, StiffnessMatrix.
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FELinearTrussDomain.cpp
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FETrussMaterial.cpp

Same as FEBio
    With L the reference length, l the current length, lambda = l / L,
    A the reference cross-sectional area and V = L A the reference volume:

        Kirchhoff stress   tau = material(ln lambda)      (E ln lambda)
        axial force        N = tau V / l
        stiffness          V / l^2 (E_t - 2 tau) e (x) e + (tau V / l^2) I
        reported stress    Cauchy, (tau / J) e (x) e, J = lambda^(1 - 2 v)
        output             "stress", "truss stretch"

    The truss carries compression: FEBio's law has no cut-off.

Different, and why
    A domain is a batch on the device with one kernel per call. What FEBio
    holds per element object is held in flat arrays over elements: rest
    length, area, and a row of the material parameters. FEBio takes the rest
    length from the reference node positions; here it is an array that
    defaults to that length, so a chord length can be a design variable
    without moving nodes (decision C3 of the note). All three arrays carry
    gradients.

    `update` also stores the axial force N per element. N is the chord
    tension, which FEBio does not output for trusses (it writes stress and
    stretch). `axial_force` evaluates it into arrays of its own so it can be
    used as an objective on a tape.

    The material is called through the uniaxial interface
    (materials/uniaxial.py): stress and tangent from the logarithmic stretch.
    A nonlinear chord law can replace the linear one without touching this
    file (decision C2).
"""

# No `from __future__ import annotations` here: the kernels below are built in a
# closure and Warp evaluates their annotations (pvec) when they are made.
from typing import Any

import numpy as np
import warp as wp

from ..assembly.system import BlockTangent, State, Term
from ..materials.models.linear_truss import LINEAR_TRUSS
from ..materials.status import NEGATIVE_JACOBIAN, NON_FINITE, OK
from ..materials.uniaxial import UniaxialModel
from .line2 import axial_internal_force, axial_stiffness, geometric_length, require_no_history

vec3d = wp.vec3d
mat33d = wp.mat33d

_KERNELS = {}


def _make_update(model: UniaxialModel):
    """The update kernel of a truss domain for one uniaxial law."""
    if model in _KERNELS:
        return _KERNELS[model]
    require_no_history(model)
    law = model.update
    pvec = model.param_vector(wp.float64)

    def update(
        X: wp.array2d(dtype=vec3d),  # (E, 2) reference positions
        emap: wp.array2d(dtype=wp.int32),  # (E, 2) entry of each node
        u: wp.array(dtype=vec3d),  # solution, one 3-vector per entry
        rest_length: wp.array(dtype=wp.float64),  # (E,) L
        area: wp.array(dtype=wp.float64),  # (E,) A, reference
        region: wp.array(dtype=wp.int32),  # (E,) row of `params`
        params: wp.array(dtype=pvec),  # material parameter rows
        length: wp.array(dtype=wp.float64),  # (E,) l
        axis: wp.array(dtype=vec3d),  # (E,) unit vector, node a to node b
        stretch: wp.array(dtype=wp.float64),  # (E,) lambda
        tau: wp.array(dtype=wp.float64),  # (E,) Kirchhoff stress
        axial_force: wp.array(dtype=wp.float64),  # (E,) N = tau V / l
        axial_tangent: wp.array(dtype=wp.float64),  # (E,) dN/dl
        status: wp.array(dtype=wp.int32),
    ):
        """Kinematics, material and axial force of one truss. The only call
        to the material in an iteration."""
        i = wp.tid()
        r = (X[i, 1] + u[emap[i, 1]]) - (X[i, 0] + u[emap[i, 0]])
        l = wp.length(r)
        L = rest_length[i]
        # A truss of zero length has no axis. FEBio divides by the length
        # here; a kernel cannot throw, so the element reports a status.
        if l > wp.float64(0.0) and L > wp.float64(0.0):
            lam = l / L
            t, Et, unused = law(wp.log(lam), params[region[i]], wp.vector(wp.float64(0.0), length=1))
            V = L * area[i]
            N = t * V / l
            if wp.isfinite(N):
                length[i] = l
                axis[i] = r / l
                stretch[i] = lam
                tau[i] = t
                axial_force[i] = N
                # d/dl of tau V / l, with d(tau)/dl = E_t / l.
                axial_tangent[i] = V / (l * l) * (Et - t)
                status[i] = wp.static(OK)
            else:
                status[i] = wp.static(NON_FINITE)
        else:
            status[i] = wp.static(NEGATIVE_JACOBIAN)

    kernel = wp.kernel(update, module="unique")
    _KERNELS[model] = kernel
    return kernel


class TrussDomain(Term):
    """One part of two-node trusses with one truss material: FEBio "linear-truss".

    Args:
        name: Name of the part.
        X: Reference positions of the two nodes of each element, shape (E, 2, 3).
        emap: Entry of the `displacement` variable at each node, shape (E, 2).
        area: Reference cross-sectional area, a scalar or one value per element.
        params: Material parameter rows in the order of `model.param_names`,
            shape (R, P). One row for the whole part, as a FEBio input gives
            it, or one row per element.
        n_entries: Number of entries of the field table.
        device: Warp device.
        model: The uniaxial law, FEBio's "linear truss" unless given.
        rest_length: Rest length per element. The distance between the
            reference node positions unless given, as in FEBio.
        region: Row of `params` used by each element. All zero unless given.
        poisson_ratio: FEBio's `v`, used only for the reported Cauchy stress.
        density: Reference density, kept for the mass matrix.
    """

    internal = True
    febio_type = "linear-truss"
    element_type = "line2"

    def __init__(
        self,
        name,
        X,
        emap,
        area,
        params,
        n_entries,
        device,
        model: UniaxialModel = LINEAR_TRUSS,
        rest_length=None,
        region=None,
        poisson_ratio: float = 0.5,
        density: float = 1.0,
    ):
        self.name = name
        self.law = model
        self.device = device
        self.n = n_entries
        self.count = E = len(X)
        self.poisson_ratio = float(poisson_ratio)
        self.density = float(density)
        self._update = _make_update(model)
        self.pvec = model.param_vector(wp.float64)

        X = np.asarray(X, dtype=np.float64)
        params = np.atleast_2d(np.asarray(params, dtype=np.float64))
        dev = dict(device=device)
        grad_on = dict(requires_grad=True, **dev)
        self.X = wp.array(X, dtype=vec3d, **dev)
        geometric = geometric_length(self.X)
        # FEBio's L0 (el.m_L0), which the mass uses whatever the rest length.
        self.geometric_length = np.array(geometric)
        self.emap = wp.array(np.asarray(emap, dtype=np.int32), dtype=wp.int32, **dev)
        self.region = wp.array(
            np.zeros(E, np.int32) if region is None else np.asarray(region, np.int32), dtype=wp.int32, **dev
        )
        # Differentiable parameters, one device array each.
        self.rest_length = wp.array(geometric if rest_length is None else rest_length, dtype=wp.float64, **grad_on)
        self.area = wp.array(np.broadcast_to(np.asarray(area, np.float64), (E,)).copy(), dtype=wp.float64, **grad_on)
        self.params = wp.array(params, dtype=self.pvec, **grad_on)
        # Element data of the last update.
        self.data = self._new_data(requires_grad=True)
        self.status = wp.zeros(E, dtype=wp.int32, **dev)
        # Four 3x3 blocks per element, at fixed (row, column) positions.
        self.block_count = 4 * E

    def _new_data(self, requires_grad: bool) -> dict:
        opts = dict(device=self.device, requires_grad=requires_grad)
        E = self.count
        return dict(
            length=wp.zeros(E, dtype=wp.float64, **opts),
            axis=wp.zeros(E, dtype=vec3d, **opts),
            stretch=wp.zeros(E, dtype=wp.float64, **opts),
            tau=wp.zeros(E, dtype=wp.float64, **opts),
            axial_force=wp.zeros(E, dtype=wp.float64, **opts),
            axial_tangent=wp.zeros(E, dtype=wp.float64, **opts),
        )

    def _launch_update(self, state: State, data: dict, status: wp.array) -> None:
        wp.launch(
            self._update,
            dim=self.count,
            inputs=[self.X, self.emap, state.u, self.rest_length, self.area, self.region, self.params],
            outputs=[
                data["length"],
                data["axis"],
                data["stretch"],
                data["tau"],
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
        # FEBio's truss stiffness is the derivative of its internal force, so
        # Newton and the adjoint use the same blocks.
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
        return {"material": self.params, "area": self.area, "rest_length": self.rest_length}

    # -- beyond Term ---------------------------------------------------------

    @property
    def parameter_names(self) -> tuple:
        return tuple(self.law.param_names)

    def axial_force(self, state: State) -> wp.array:
        """Axial force N per element at `state` (tension positive), in a new array.

        For outputs and objectives. It launches the same kernel as `update`
        into arrays of its own, so on a `wp.Tape` the result depends on
        `state.u` and on the parameter arrays and on nothing the solver or
        the adjoint writes to.
        """
        data = self._new_data(requires_grad=True)
        self._launch_update(state, data, wp.zeros(self.count, dtype=wp.int32, device=self.device))
        return data["axial_force"]

    def mass_blocks(self) -> np.ndarray:
        """Consistent mass per element, (E, 2, 2): rho A L / 3 on the diagonal
        and rho A L / 6 off it (FELinearTrussDomain::ElementMassMatrix). Each
        entry multiplies a 3x3 identity, at the positions of the stiffness
        blocks. Not used by the static run path."""
        # With the geometric length, as FEBio (m_a0 * m_L0), not the rest
        # length parameter: changing a chord's rest length does not change its
        # mass. Computed on the host when the system is built, so the mass does
        # not follow later changes of the area and carries no gradient yet.
        m = self.density * self.area.numpy() * self.geometric_length
        return m[:, None, None] * np.array([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]])

    def plot(self, variable: str) -> np.ndarray:
        """Host copy of a plot variable at the last update, by FEBio's name.

        "stress"         (E, 1, 3, 3) Cauchy stress (tau / J) e (x) e with
                         J = lambda^(1 - 2 v), as FELinearTrussDomain::Update
        "truss stretch"  (E,) lambda
        "axial force"    (E,) N. Not a FEBio plot variable.
        """
        d = self.data
        if variable == "truss stretch":
            return d["stretch"].numpy().copy()
        if variable == "axial force":
            return d["axial_force"].numpy().copy()
        if variable == "stress":
            lam = d["stretch"].numpy()
            e = d["axis"].numpy()
            J = np.power(np.where(lam > 0, lam, 1.0), 1.0 - 2.0 * self.poisson_ratio)
            s = d["tau"].numpy() / J
            return (s[:, None, None] * e[:, :, None] * e[:, None, :])[:, None]
        raise KeyError(variable)
