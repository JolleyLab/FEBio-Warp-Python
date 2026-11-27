"""Contact as a PhysicsTerm: residual, tangent action, committed state.

``ContactPotentialTerm`` adapts ``potential.ContactPotential`` to the
``contracts.PhysicsTerm`` protocol with the repository's residual sign
convention, internal plus contact minus external: the contact residual is
+dE_c/dx, the negative of the repulsive nodal force.

State is a ``wp.array(dtype=wp.vec3d)`` of nodal positions. The potential has no
history, so ``commit`` and ``rollback`` do nothing; they exist so that a
frictional law (``sliding.py``, planned) can slot in behind the same interface
with a committed footprint and slip state.

Pair selection is explicit: call ``update_pairs(state)`` once per Newton
iteration (or per step, for a frozen-pairing adjoint), then ``residual`` and
``linearize`` run over that list without host synchronization.

Coupling to PyTorch and JAX. ``energy_and_gradient`` is the building block
for a custom op: it records the energy on a ``wp.Tape`` and returns dE/dx on
the device, so a ``torch.autograd.Function`` or a ``jax.custom_vjp`` can wrap
it with zero-copy views (``wp.to_torch``, ``wp.to_jax`` / ``wp.from_jax``).
Second derivatives come from the analytic pair Hessian (``linearize`` gives a
matrix-free Hessian-vector product, ``assemble`` a BSR matrix), not from a
tape of a tape, which Warp does not provide; the frameworks consume the
Hessian action as a linear operator for Gauss-Newton, Newton or adjoint
solves. The pair list is selected separately and never differentiated; the
energy is C2 in the positions because phi and its first two derivatives vanish
at R_out (with ``w_tol > 0`` the normal test makes it discontinuous).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .interface import ContactInterfaceBase
from .points import ContactPoints
from .potential import ContactPotential, PotentialParams

vec3d = wp.vec3d


@wp.kernel
def _axpy(alpha: wp.float64, x: wp.array(dtype=vec3d), y: wp.array(dtype=vec3d)):
    i = wp.tid()
    y[i] = y[i] + alpha * x[i]


class ContactTangent:
    """TangentOperator of the contact potential at the state it was linearized at.

    Matrix-free, in the shape of an MFEM Operator (Mult and MultTranspose, no
    assembled matrix): the action interpolates nodes to contact points, applies
    the stored pair blocks, and scatters back. ``assemble`` gives the BSR
    matrix when a preconditioner or direct solver needs one.
    """

    def __init__(self, term: "ContactPotentialTerm", project: bool):
        self.term = term
        self.project = project
        self.height = self.width = 3 * term.potential.n_nodes
        self.term.potential.hessian_blocks(project)

    def apply(self, vector: wp.array, out: wp.array) -> None:
        self.term.potential.hessian_matvec(vector, out, project=self.project, rebuild_blocks=False)

    def apply_transpose(self, vector: wp.array, out: wp.array) -> None:
        # the contact Hessian is symmetric (projected or not)
        self.apply(vector, out)

    def assemble(self, dest=None):
        """BSR matrix of 3x3 blocks on the device (warp.sparse), for preconditioners."""
        return self.term.potential.hessian_bsr(project=self.project, dest=dest)


class ContactPotentialTerm(ContactInterfaceBase):
    """Contact potential behind the PhysicsTerm and ContactInterface protocols."""

    name = "contact_potential"

    def __init__(
        self,
        points: ContactPoints,
        params: PotentialParams,
        pair_class: np.ndarray,
        ref_node_x: np.ndarray,
        n_nodes: int,
        exclusion: str = "neighbors",
        w_tol: float = 0.0,
        weight_mode: str = "reference",
        project_tangent: bool = False,
        device: str = "cpu",
        **search_kwargs,
    ):
        self.potential = ContactPotential(
            points,
            params,
            pair_class,
            ref_node_x,
            n_nodes,
            exclusion=exclusion,
            w_tol=w_tol,
            weight_mode=weight_mode,
            device=device,
            **search_kwargs,
        )
        self.project_tangent = project_tangent
        self.device = device
        self._f = wp.zeros(int(n_nodes), dtype=vec3d, device=device, requires_grad=True)

    def update_pairs(self, state: wp.array, sync: bool = True) -> int:
        return self.potential.build_pairs(state, sync=sync)

    def residual(self, state: wp.array, history, out: wp.array) -> None:
        """out += dE_c/dx (contact enters the residual with the internal-force sign)."""
        self._f.zero_()
        self.potential.energy_force(state, self._f)
        wp.launch(
            _axpy,
            dim=self._f.shape[0],
            inputs=[wp.float64(-1.0), self._f, out],
            device=self.device,
        )

    def energy(self) -> float:
        return self.potential.energy_value()

    def energy_and_gradient(self, state: wp.array):
        """(E_c as a device scalar array, dE_c/dx as a device array) through wp.Tape.

        ``state`` must have ``requires_grad=True``. The gradient equals the
        residual contribution and is returned as a device array that a PyTorch
        or JAX custom op can view through Warp interop. In ``weight_mode="current"``
        the weights are held fixed in the backward pass, as in the residual, so
        the two agree in both modes.
        """
        if not state.requires_grad:
            raise ValueError("state must be allocated with requires_grad=True")
        pot = self.potential
        self._f.zero_()
        tape = wp.Tape()
        with tape:
            E = pot.energy_force(state, self._f)
        state.grad.zero_()
        tape.backward(E)
        grad = wp.clone(state.grad)
        tape.zero()
        return E, grad

    def residual_parameter_vjp(self, state: wp.array, lam: wp.array) -> dict:
        """lam^T dR/dtheta for the contact parameters, through wp.Tape.

        This is the term the equilibrium adjoint needs after solving
        K^T lam = dJ/dx. Returns ``{"kc": array over pair classes}`` on the
        host; ``R = -f`` so the sign of the force adjoint is flipped. The pair
        list and the integration weights are held fixed, as in ``residual``.
        """
        pot = self.potential
        self._f.zero_()
        tape = wp.Tape()
        with tape:
            pot.energy_force(state, self._f)
        self._f.grad.zero_()
        wp.launch(
            _axpy,
            dim=self._f.shape[0],
            inputs=[wp.float64(-1.0), lam, self._f.grad],
            device=self.device,
        )
        tape.backward(grads={self._f: self._f.grad})
        out = {"kc": pot.kc_gradient_from_tables()}
        tape.zero()
        return out

    def linearize(self, state: wp.array, history) -> ContactTangent:
        """Tangent action at ``state`` over the current pair list.

        The points are refreshed from ``state``; the pair list is whatever
        ``update_pairs`` last built, which is the frozen-pairing linearization
        the equilibrium adjoint wants.
        """
        self.potential.search.update_points(state)
        return ContactTangent(self, self.project_tangent)

    def results(self) -> dict:
        """Point forces and the Wu 2022 contact area per body, after a residual call."""
        pot = self.potential
        bodies = np.unique(pot.points.body)
        return {
            "point_force": pot.point_forces(),
            "contact_area": {int(b): pot.contact_area(body=int(b)) for b in bodies},
            "energy": pot.energy_value(),
            "n_pairs": pot.search.n_pairs,
        }
