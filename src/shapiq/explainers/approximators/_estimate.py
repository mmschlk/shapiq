"""The estimate: a basis game carrying its estimation provenance.

An estimate IS a game — the intensional surrogate the family's
coefficients define — with the record of how it was estimated riding
along: the evidence, the banked budget remainder, the index it was made
under. It holds no policy and no process verbs; estimation lives on the
approximator (``policy.refine(estimate, budget)``), math lives here, and
everything a game can do (evaluate, compose, project, measure fidelity)
an estimate can do.

The empty interaction is an ordinary coefficient slot; what it holds is
each family's declared logic — the empty-coalition value for the
efficiency family, the fitted intercept where the family fits one,
nothing (zero) where the index defines no order-0 attribution.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from shapiq.errors import InsufficientSamplesError
from shapiq.games import BasisGame, MoebiusBasis
from shapiq.sampling import ApproximationState, SamplingState

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from jax import Array

    from shapiq.coalitions import CoalitionArray
    from shapiq.games._basis import Basis


class Estimate(BasisGame):
    """An estimated game: coefficient plane, game plane, provenance."""

    def __init__(
        self,
        *,
        terms: tuple[frozenset[int], ...],
        values: np.ndarray,
        n_players: int,
        evidence: ApproximationState,
        bank: int,
        index: object = None,
        basis: Basis | None = None,
        value_shape: tuple[int, ...] = (),
        target_shape: tuple[int, ...] = (),
        deduplicated: bool = False,
        variance: Mapping[frozenset[int], float] | None = None,
        unready_reason: str | None = None,
    ) -> None:
        """Initialize an estimate from coefficients and provenance.

        Args:
            terms: Interactions aligned with ``values``.
            values: Host float64 coefficients, canonical leading layout
                ``(*value_shape, *target_shape, n_terms)``.
            n_players: Number of players of the estimated game.
            evidence: The evidence the coefficients were derived from.
            bank: The banked budget remainder.
            index: The interaction index the estimate was made under.
            basis: The basis the coefficients live in (moebius default).
            value_shape: The game's value shape.
            target_shape: The game's explanation-target shape.
            deduplicated: Whether the evidence was gathered deduplicating.
            variance: Per-interaction uncertainty — a capability, present
                when the estimator provides it.
            unready_reason: When the evidence cannot support coefficients
                yet, the reason reads raise with; the carry stays legal
                (banked budgets ride it) but its planes refuse.
        """
        super().__init__(
            MoebiusBasis() if basis is None else basis,
            None,
            n_players,
            terms=terms,
            values=values,
            value_shape=value_shape,
            target_shape=target_shape,
        )
        self.evidence = evidence
        self.bank = bank
        self.index = index
        self.deduplicated = deduplicated
        self.variance = variance
        self._unready_reason = unready_reason

    @property
    def ready(self) -> bool:
        """Return whether the evidence supported coefficients."""
        return self._unready_reason is None

    @property
    def spent(self) -> int:
        """Return evaluations spent, derived from the evidence."""
        if not isinstance(self.evidence, SamplingState):
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
        if self._unready_reason is not None:
            raise InsufficientSamplesError(self._unready_reason)

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"{type(self).__name__}(index={self.index!r}, "
            f"n_samples={self.evidence.n_samples!r}, spent={self.spent!r}, "
            f"bank={self.bank!r}, ready={self.ready!r})"
        )


def leading_blocks_to_terms(
    attributions: Mapping[int, Array],
    n_players: int,
    empty: Array | None,
) -> tuple[tuple[frozenset[int], ...], np.ndarray]:
    """Assemble per-order coefficient blocks into aligned terms and values.

    Args:
        attributions: Per-interaction-size blocks in the canonical leading
            layout ``(*value_shape, *target_shape, n_interactions_of_size)``,
            interactions in ``combinations`` order.
        n_players: Number of players the interactions range over.
        empty: The empty interaction's coefficient block
            ``(*value_shape, *target_shape)`` or ``None`` when the family
            declares no order-0 slot (and no size-0 block is given).

    Returns:
        Terms and the concatenated host float64 coefficient array.
    """
    terms: list[frozenset[int]] = []
    blocks: list[np.ndarray] = []
    if empty is not None and 0 not in attributions:
        terms.append(frozenset())
        blocks.append(np.asarray(empty, dtype=np.float64)[..., None])
    for size in sorted(attributions):
        block = np.asarray(attributions[size], dtype=np.float64)
        if size == 0:
            terms.append(frozenset())
            blocks.append(block.reshape(*block.shape[:-1], 1) if block.shape[-1:] == (1,) else block[..., None])
            continue
        terms.extend(frozenset(members) for members in combinations(range(n_players), size))
        blocks.append(block)
    if not blocks:
        return (), np.zeros(0)
    return tuple(terms), np.concatenate(blocks, axis=-1)


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
