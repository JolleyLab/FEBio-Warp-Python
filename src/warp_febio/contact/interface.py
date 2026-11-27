"""Base class for contact interfaces, in the spirit of FEBio's FEContactInterface.

FEBio organizes contact as interfaces (sliding-elastic, sticky, tied, contact
potential, ...) that each own their surfaces, their search, their enforcement
law and their outputs, behind one set of calls the solver makes: update the
pairing, add to the residual, add to the stiffness, augment, serialize state.
This module gives FEBio-Warp the same seam so that new laws can be added
without touching the solver.

Every interface implements ``contracts.PhysicsTerm`` plus a pairing update and
a result dictionary. The first implementation is the volume potential
(``ContactPotentialTerm``). Planned next, in the order the targets need them:

* ``SlidingContact``: gap and normal from a ``wp.Mesh`` closest-point query,
  penalty with augmented Lagrangian, Coulomb friction with committed footprint
  and slip (FEBio sliding-elastic and edge-to-surface sliding).
* ``StickyContact`` / ``TiedContact``: pair springs created at engagement with
  a maximum traction (FEBio sticky, tied-elastic); clip grasp and tines.

An interface never owns the solver loop. The analysis procedure decides when
``update_pairs`` runs (each Newton iteration for the forward solve, once for a
frozen-pairing adjoint), when ``augment`` runs, and when ``commit`` is called.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import warp as wp


@runtime_checkable
class ContactInterface(Protocol):
    """What the solver may call on any contact law."""

    name: str

    def update_pairs(self, state: Any, sync: bool = True) -> int:
        """Refresh the candidate pairing for ``state``; return the pair count (or 0 if unknown)."""
        ...

    def residual(self, state: Any, history: Any, out: Any) -> None:
        """Add the contact contribution with the internal-force sign to ``out``."""
        ...

    def linearize(self, state: Any, history: Any) -> Any:
        """Return a TangentOperator (apply, apply_transpose) at ``state``."""
        ...

    def augment(self, state: Any) -> bool:
        """Augmented-Lagrangian multiplier update; True when converged or not applicable."""
        ...

    def commit(self, trial_history: Any) -> None: ...

    def rollback(self) -> None: ...

    def results(self) -> dict[str, Any]:
        """Named outputs for io/results (gap, traction, contact area, ...)."""
        ...


class ContactInterfaceBase:
    """Defaults shared by the concrete interfaces: no augmentation, no history."""

    name = "contact"

    def augment(self, state: Any) -> bool:
        return True

    def commit(self, trial_history: Any) -> None:
        return None

    def rollback(self) -> None:
        return None

    def results(self) -> dict[str, Any]:
        return {}


class ContactSet:
    """Several interfaces acting on one state, driven as a unit by the procedure."""

    def __init__(self, interfaces: list[ContactInterface]):
        self.interfaces = list(interfaces)

    def update_pairs(self, state: wp.array, sync: bool = True) -> int:
        return sum(ci.update_pairs(state, sync=sync) for ci in self.interfaces)

    def residual(self, state: wp.array, history: Any, out: wp.array) -> None:
        for ci in self.interfaces:
            ci.residual(state, history, out)

    def linearize(self, state: wp.array, history: Any) -> _SumTangent:
        return _SumTangent([ci.linearize(state, history) for ci in self.interfaces])

    def augment(self, state: wp.array) -> bool:
        return all([ci.augment(state) for ci in self.interfaces])

    def commit(self, trial_history: Any) -> None:
        for ci in self.interfaces:
            ci.commit(trial_history)

    def rollback(self) -> None:
        for ci in self.interfaces:
            ci.rollback()

    def results(self) -> dict[str, Any]:
        out = {}
        for ci in self.interfaces:
            for k, v in ci.results().items():
                out[f"{ci.name}/{k}"] = v
        return out


class _SumTangent:
    def __init__(self, parts):
        self.parts = parts

    def apply(self, vector, out) -> None:
        for t in self.parts:
            t.apply(vector, out)

    def apply_transpose(self, vector, out) -> None:
        for t in self.parts:
            t.apply_transpose(vector, out)
