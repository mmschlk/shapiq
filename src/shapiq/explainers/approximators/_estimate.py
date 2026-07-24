"""The estimate: an inert game-with-provenance carried between policy verbs.

An estimate IS a value function (the intensional surrogate the family's
coefficients define) plus a provenance record: the evidence it was
derived from and the banked budget remainder. It holds no policy and no
process verbs — estimation lives on the approximator
(``policy.refine(estimate, budget)``), math lives here.

Everything else is derived from the evidence, per the provenance rule
(evidence plus checkpointed non-derivables; the bank covers the one
pre-evidence banking edge the cuts cannot): ``spent``, ``units_done``,
and the stall counter are pure functions of the stored stream, which is
what makes rollback-and-resample replay exact even across a stall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from shapiq.errors import InsufficientSamplesError
from shapiq.games import ParametricGame
from shapiq.sampling import ApproximationState, SamplingState

if TYPE_CHECKING:
    from collections.abc import Collection

    from jax import Array

    from shapiq.coalitions import CoalitionArray
    from shapiq.explanations import ExplanationArray


@dataclass(frozen=True)
class Estimate:
    """An estimated game: coefficient plane, game plane, provenance."""

    evidence: ApproximationState
    bank: int
    n_players: int
    view: ExplanationArray[Array] | None
    """The family's coefficient view, or ``None`` before enough evidence."""

    deduplicated: bool = False
    """Whether the evidence was gathered deduplicating (a provenance fact)."""

    shortfall: InsufficientSamplesError | None = None
    """The family's coverage error when evidence is not enough for a view."""

    target_shape: tuple[int, ...] = ()
    value_shape: tuple[int, ...] = ()

    def __getitem__(self, interaction: Collection[int]) -> Array:
        """Read one coefficient (the coefficient plane)."""
        if isinstance(interaction, int):
            msg = (
                f"interactions are player collections: read player {interaction} "
                f"with estimate[({interaction},)]"
            )
            raise TypeError(msg)
        # the empty interaction keeps each family's own teaching error for
        # now; the surrogate's empty coefficient becomes a real slot when
        # explanations dissolve into parametric games (arc 3)
        return self._require_view()(tuple(interaction))

    def __call__(self, coalitions: CoalitionArray) -> Array:
        """Evaluate the estimated surrogate game (the game plane)."""
        return self.as_game()(coalitions)

    def as_game(self) -> ParametricGame:
        """Return the surrogate as a moebius-basis parametric game.

        The coefficient view is read as a k-additive game: exact surrogate
        semantics for the projection family (SV, FSII, FBII, kADD), the
        declared efficient reading for the derivative family.

        Raises:
            InsufficientSamplesError: If there is no evidence yet.
            ValueError: If the estimate carries target or value axes;
                slice a scalar target first (full dissolution is arc 3).
        """
        view = self._require_view()
        if self.target_shape != () or self.value_shape != ():
            msg = (
                "the game plane is scalar for now: this estimate carries "
                f"target_shape={self.target_shape} and value_shape="
                f"{self.value_shape}; read coefficients with [] instead"
            )
            raise ValueError(msg)
        baseline = 0.0 if view.baseline is None else float(jnp.asarray(view.baseline))
        coefficients: dict[Collection[int], float] = {(): baseline}
        for interaction in view.iter_interactions():
            coefficients[interaction] = float(jnp.asarray(view(interaction)))
        return ParametricGame("moebius", coefficients, self.n_players)

    @property
    def index(self) -> object:
        """Return the interaction index this estimate was made under."""
        return self._require_view().index

    @property
    def spent(self) -> int:
        """Return evaluations spent, derived from the evidence."""
        if not isinstance(self.evidence, SamplingState):
            return 0
        if self.deduplicated:
            return len(self.evidence.key_index())
        return self.evidence.n_samples

    def _require_view(self) -> ExplanationArray[Array]:
        if self.view is None:
            if self.shortfall is not None:  # the family said exactly what is missing
                raise self.shortfall
            msg = (
                "no estimate yet: this carry holds banked budget but not "
                "enough evidence; refine it with a larger budget first"
            )
            raise InsufficientSamplesError(msg)
        return self.view

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"{type(self).__name__}(n_samples={self.evidence.n_samples!r}, "
            f"bank={self.bank!r}, has_view={self.view is not None!r})"
        )


def trailing_quiet_units(evidence: ApproximationState, unit_rows: int, n_seed_samples: int) -> int:
    """Derive the stall counter: trailing whole units with no novel coalition.

    A row is novel exactly when it is its coalition's first occurrence in
    the stream (``row == key_index[key]``). Deriving the counter from the
    evidence — instead of carrying it — is what makes rollback and replay
    exact across a stall: the counter can never disagree with the stream.
    """
    if not isinstance(evidence, SamplingState):
        return 0
    index = evidence.key_index()
    keys = _row_keys(evidence)
    quiet = 0
    unit_end = evidence.n_samples
    while unit_end - unit_rows >= n_seed_samples:
        start = unit_end - unit_rows
        novel = any(index[keys[row]] == row for row in range(start, unit_end))
        if novel:
            break
        quiet += 1
        unit_end = start
    return quiet


def _row_keys(evidence: SamplingState) -> list[bytes]:
    packed = evidence.packed_keys()
    width = packed.shape[-1]
    blob = packed.tobytes()
    return [blob[start : start + width] for start in range(0, len(blob), width)]
