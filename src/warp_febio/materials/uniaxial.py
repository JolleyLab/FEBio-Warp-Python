"""Uniaxial material models with history (draft v0.3).

A uniaxial model relates one axial strain to one axial stress. It is what a
beam element integrates over the fibers of its cross-section, and what truss
and chord elements use directly. The element owns the strain measure and
supplies it; the model only maps strain to stress.

This is also the first model kind with history, so it fixes the history
contract for the solid models that will follow:

    update(strain, p, committed) -> (stress, tangent, trial)

    strain     axial strain supplied by the element (dimensionless)
    p          parameter vector, order given by `param_names`
    committed  state vector at the last converged step (read only)
    stress     axial stress
    tangent    d(stress)/d(strain), consistent with the update
    trial      state vector for this iterate

The solver calls `update` as often as it likes within a step, always from the
same committed state. On convergence it swaps the trial array into the
committed one; on a rejected step it discards the trial. Nothing is updated in
place. This mirrors FEBio's material point Init/Update and the commit/rollback
in warp_febio/contracts.py, and it keeps the update a pure function, which is
what makes it differentiable and safe to run in parallel.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import warp as wp


@dataclass(frozen=True)
class UniaxialModel:
    name: str
    param_names: tuple
    state_names: tuple  # order of the entries of the state vector
    update: Callable
    initial_state: tuple  # state of virgin material
    source: str = ""
    febio_type: str = ""  # FEBio type string, when FEBio has this model
    options: tuple = ()  # build-time choices as (name, value) pairs

    def option(self, name, default=None):
        return dict(self.options).get(name, default)

    def param_vector(self, dtype=wp.float64):
        return wp.types.vector(length=len(self.param_names), dtype=dtype)

    def state_vector(self, dtype=wp.float64):
        return wp.types.vector(length=len(self.state_names), dtype=dtype)

    def pack(self, dtype=wp.float64, **values):
        missing = set(self.param_names) - set(values)
        extra = set(values) - set(self.param_names)
        if missing or extra:
            raise ValueError(
                f"{self.name}: missing {sorted(missing)}, unknown {sorted(extra)}"
            )
        return self.param_vector(dtype)(*[values[k] for k in self.param_names])


_KERNELS = {}


def make_update_kernel(model: UniaxialModel, dtype=wp.float64):
    """Kernel that updates a batch of fibers from their committed state."""
    key = (model, dtype)
    if key in _KERNELS:
        return _KERNELS[key]
    pvec = model.param_vector(dtype)
    svec = model.state_vector(dtype)
    update = model.update

    def step(
        strain: wp.array(dtype=dtype),
        region: wp.array(dtype=int),
        params: wp.array(dtype=pvec),
        committed: wp.array(dtype=svec),
        stress: wp.array(dtype=dtype),
        tangent: wp.array(dtype=dtype),
        trial: wp.array(dtype=svec),
    ):
        i = wp.tid()
        s, t, new_state = update(strain[i], params[region[i]], committed[i])
        stress[i] = s
        tangent[i] = t
        trial[i] = new_state

    kernel = wp.kernel(step, module="unique")
    _KERNELS[key] = kernel
    return kernel


def integrate(model, strain_path, params, dtype=wp.float64, device="cpu"):
    """Drive fibers through a strain history, committing after every step.

    strain_path: array (steps, n). Each column is one fiber's strain history.
    Returns stress (steps, n), tangent (steps, n), state (steps, n, state size).
    Host convenience for tests and for fitting uniaxial data.
    """
    strain_path = np.atleast_2d(np.asarray(strain_path, dtype=np.float64))
    if strain_path.shape[0] == 1:
        strain_path = strain_path.T
    steps, n = strain_path.shape
    svec = model.state_vector(dtype)
    committed = wp.array(
        np.tile(np.asarray(model.initial_state, dtype=np.float64), (n, 1)),
        dtype=svec,
        device=device,
    )
    trial = wp.zeros(n, dtype=svec, device=device)
    stress = wp.zeros(n, dtype=dtype, device=device)
    tangent = wp.zeros(n, dtype=dtype, device=device)
    region = wp.zeros(n, dtype=int, device=device)
    p = wp.array(
        [model.pack(dtype, **params)], dtype=model.param_vector(dtype), device=device
    )
    kernel = make_update_kernel(model, dtype)
    out_s, out_t, out_q = [], [], []
    for k in range(steps):
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                wp.array(strain_path[k], dtype=dtype, device=device),
                region,
                p,
                committed,
            ],
            outputs=[stress, tangent, trial],
            device=device,
        )
        committed, trial = trial, committed  # commit: swap, never copy in place
        out_s.append(stress.numpy().copy())
        out_t.append(tangent.numpy().copy())
        out_q.append(committed.numpy().copy())
    return np.array(out_s), np.array(out_t), np.array(out_q)
