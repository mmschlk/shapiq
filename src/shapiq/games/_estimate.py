"""The estimate: a basis game with its provenance tagging along.

An estimate IS a game — the readable surrogate an explainer produced —
and everything about *how* it was produced rides in one record: the
:class:`Provenance`. The carry is (data, record), nothing more; process
verbs live on the frozen policies, math lives on the game, and the
record answers for history. Flat reads (``estimate.bank``,
``estimate.spent``) delegate into the record so ergonomics stay flat
while construction stays two-argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shapiq.errors import InsufficientSamplesError
from shapiq.games._basis import BasisGame
from shapiq.sampling import Evidence, SampledEvidence

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    import numpy as np
    from jax import Array

    from shapiq.coalitions import CoalitionArray


@dataclass(frozen=True)
class Provenance:
    """How an estimate came to be: evidence, budget bookkeeping, identity."""

    evidence: Evidence
    """The record the coefficients were derived from."""

    bank: int = 0
    """The banked budget remainder."""

    index: object | None = None
    """The interaction index the estimate was made under."""

    fingerprint: tuple[object, ...] | None = None
    """The producing policy's structural identity; policies refuse to
    refine a carry whose fingerprint is not theirs."""

    deduplicated: bool = False
    """Whether the evidence was gathered deduplicating."""

    variance: Mapping[frozenset[int], float] | None = None
    """Per-interaction uncertainty — a capability, present when the
    estimator provides it."""

    shortfall: str | None = None
    """Why the evidence cannot support coefficients yet, or ``None``
    when the estimate is ready. A carry with a shortfall stays legal
    (banked budgets ride it) but its planes refuse to read."""


class Estimate(BasisGame):
    """An estimated game: a readable surrogate plus its provenance."""

    def __init__(self, surrogate: BasisGame, provenance: Provenance) -> None:
        """Initialize the carry from a surrogate game and its record.

        Args:
            surrogate: The readable game the producer built (possibly
                empty while evidence is insufficient).
            provenance: The record of how it came to be.
        """
        super().__init__(
            surrogate.basis,
            None,
            surrogate.n_players,
            terms=surrogate.terms,
            values=surrogate.coefficients,
            value_shape=surrogate.value_shape,
            target_shape=surrogate.target_shape,
        )
        self.provenance = provenance

    @property
    def evidence(self) -> Evidence:
        """Return the evidence the coefficients were derived from."""
        return self.provenance.evidence

    @property
    def bank(self) -> int:
        """Return the banked budget remainder."""
        return self.provenance.bank

    @property
    def index(self) -> object | None:
        """Return the interaction index the estimate was made under."""
        return self.provenance.index

    @property
    def fingerprint(self) -> tuple[object, ...] | None:
        """Return the producing policy's structural identity."""
        return self.provenance.fingerprint

    @property
    def deduplicated(self) -> bool:
        """Return whether the evidence was gathered deduplicating."""
        return self.provenance.deduplicated

    @property
    def variance(self) -> Mapping[frozenset[int], float] | None:
        """Return per-interaction uncertainty when the estimator provides it."""
        return self.provenance.variance

    @property
    def ready(self) -> bool:
        """Return whether the evidence supported coefficients."""
        return self.provenance.shortfall is None

    @property
    def spent(self) -> int:
        """Return evaluations spent, derived from the evidence."""
        if not isinstance(self.evidence, SampledEvidence):
            return 0
        if self.deduplicated:
            return len(self.evidence.key_index())
        return self.evidence.n_samples

    def __getitem__(self, interaction: Collection[int]) -> np.ndarray | float:
        """Read one coefficient; raises while the estimate is not ready."""
        self._require_ready()
        return super().__getitem__(interaction)

    def _call(self, coalitions: CoalitionArray) -> Array:
        self._require_ready()
        return super()._call(coalitions)

    def _host_values(self, masks: np.ndarray) -> np.ndarray:
        self._require_ready()
        return super()._host_values(masks)

    def _require_ready(self) -> None:
        if self.provenance.shortfall is not None:
            raise InsufficientSamplesError(self.provenance.shortfall)

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"{type(self).__name__}(index={self.index!r}, "
            f"n_samples={self.evidence.n_samples!r}, spent={self.spent!r}, "
            f"bank={self.bank!r}, ready={self.ready!r})"
        )
