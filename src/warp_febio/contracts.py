"""Extension contracts; these Protocols describe interfaces, not hidden solvers."""

from typing import Any, Protocol


class TangentOperator(Protocol):
    """Jacobian at a declared state, with boundary treatment already specified."""

    def apply(self, vector: Any, out: Any) -> None: ...
    def apply_transpose(self, vector: Any, out: Any) -> None: ...


class PhysicsTerm(Protocol):
    """Planned static residual sign: internal + contact - external, in newtons.

    Path-dependent materials must not commit history during rejected Newton trials.
    The pilot examples under pilots/ are not adapters to this protocol.
    """

    def residual(self, state: Any, history: Any, out: Any) -> None: ...
    def linearize(self, state: Any, history: Any) -> TangentOperator: ...
    def commit(self, trial_history: Any) -> None: ...
    def rollback(self) -> None: ...
