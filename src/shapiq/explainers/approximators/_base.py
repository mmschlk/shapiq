from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NoReturn

from shapiq._shape import ensure_bool, logical_size, validate_int
from shapiq.errors import InsufficientSamplesError
from shapiq.explainers._base import validate_index_binding
from shapiq.explainers.approximators._engine import banked_at, grow
from shapiq.explainers.approximators._estimate import leading_blocks_to_terms
from shapiq.games import BasisGame, Estimate, Game, MoebiusBasis, Provenance
from shapiq.sampling import Evidence, NoEvidence, Sampler

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from shapiq.interactions import InteractionIndex


class Approximator(ABC):
    """Base abstraction for sampling-based explainers: frozen config, verbs.

    An approximator is a frozen policy: a sampler, the family's expansion
    from draws to coalition rows, and the interaction index it estimates.
    It holds no process state — the verbs ``estimate``, ``refine``, and
    ``at_evidence`` run the engine's pure loop over ``(evidence, bank)``
    and return an :class:`Estimate`, a game with provenance. Everything
    not in the evidence is recomputed from it, so any estimate is an
    exact resume point.

    Budgets are denominated in game evaluations; whole-unit spending with
    a banked remainder keeps sampled streams invariant under budget
    splits, and evaluations that could not inform an estimate are never
    made. With deduplication enabled, every distinct coalition is
    evaluated on the game at most once: repeated coalitions reuse stored
    values, only novel evaluations are charged, and the final unit may
    overshoot into the bank (negative = borrowed, repaid by the next
    call). If the sampler stops producing novel coalitions, the remaining
    budget stays banked and a ``SamplingStallWarning`` is issued.
    """

    game: Game[Array]
    index: InteractionIndex
    order: int
    sampler: Sampler
    deduplicate: bool

    def __init__(
        self,
        game: Game[Array],
        sampler: Sampler,
        index: InteractionIndex,
        *,
        render: Callable[[Array], Array],
        unit_length: int,
        prelude_masks: Array | None = None,
        deduplicate: bool = False,
    ) -> None:
        """Initialize an approximator without evaluating the game.

        Args:
            game: Game to explain.
            sampler: Stateless draw source matching the game's players and
                target shape.
            index: The interaction index the subclass estimates.
            render: The family's expansion from draws to per-unit coalition
                masks (identity rows for coalition samplers).
            unit_length: Coalition rows one draw materializes into.
            prelude_masks: Deterministic seed masks following the empty and
                grand coalition, or ``None``.
            deduplicate: Whether to evaluate each distinct coalition at most
                once.
        """
        self.order = validate_index_binding(game, index)
        self.game = game
        self.index = index
        if sampler.n_players != game.n_players:
            msg = "sampler and game use different numbers of players"
            raise ValueError(msg)
        if sampler.target_shape != game.target_shape:
            msg = "sampler and game use different target shapes"
            raise ValueError(msg)
        validate_int("unit_length", unit_length, minimum=1)
        self.sampler = sampler
        self._render = render
        self._unit_length = unit_length
        self._prelude_masks = prelude_masks
        self.deduplicate = ensure_bool("deduplicate", deduplicate)
        if self.deduplicate and logical_size(sampler.shared_target_shape) != 1:
            msg = (
                "deduplicate=True requires the same coalitions to be sampled for every "
                "explanation target; pass share_samples=True (or share the selected axes)"
            )
            raise ValueError(msg)

    @property
    def n_seed_samples(self) -> int:
        """Return the seed block length: empty and grand coalition, then prelude."""
        if self._prelude_masks is None:
            return 2
        return 2 + int(self._prelude_masks.shape[-2])

    @property
    def unit_rows(self) -> int:
        """Return the coalition rows one sampled unit contributes."""
        return self._unit_length * self.sampler.draws_per_unit

    @property
    def min_budget(self) -> int:
        """Return the floor below which ``explain()`` cannot succeed.

        The first explanation needs the seed block plus one completed
        sampled unit. It is a floor, not a guarantee: whether the drawn
        coalitions carry enough evidence is method-specific (interaction
        coverage for permutation walks, identification for the
        regressions), and ``explain()`` raises ``InsufficientSamplesError``
        stating the shortfall while they do not.
        """
        return self.n_seed_samples + self.unit_rows

    def __repr__(self) -> str:
        """Return a concise representation of the frozen configuration."""
        return (
            f"{type(self).__name__}(index={self.index!r}, "
            f"order={self.order!r}, deduplicate={self.deduplicate!r})"
        )

    def estimate(self, budget: int) -> Estimate:
        """Estimate the game from scratch: spend a budget, return the carry.

        The returned :class:`Estimate` is inert — a game-with-provenance;
        continue it with ``refine`` on this (frozen) policy.
        """
        fresh = Estimate(
            self._empty_surrogate(),
            Provenance(
                evidence=NoEvidence(),
                index=self.index,
                deduplicated=self.deduplicate,
                fingerprint=self._fingerprint,
                shortfall="no evidence yet: refine this estimate with a budget first",
            ),
        )
        return self.refine(fresh, budget)

    def refine(self, carry: Estimate, budget: int) -> Estimate:
        """Spend more budget on an estimate and return the grown carry.

        Counters are derived from the carried evidence (the stall counter
        included — it is a pure function of the unit sequence), so any
        estimate is an exact resume point: rollback and resample replay
        bit-identically, stalls included. Refining another policy's carry
        raises: the fingerprint says whose it is.
        """
        self._require_own_carry(carry)
        evidence, bank = grow(self, carry.evidence, carry.bank, budget)
        return self._as_estimate(evidence, bank)

    def at_evidence(self, evidence: Evidence, bank: int | None = None) -> Estimate:
        """Return the estimate a policy derives from given evidence.

        ``bank`` defaults to the banked remainder the evidence's last
        checkpoint recorded; rolling back an estimate is
        ``policy.at_evidence(carry.evidence.rollback(steps))``.
        """
        return self._as_estimate(evidence, banked_at(evidence) if bank is None else bank)

    @property
    def _fingerprint(self) -> tuple[object, ...]:
        """Return the policy's structural identity for carry checks."""
        return (type(self).__name__, self.unit_rows, self.n_seed_samples, self.deduplicate)

    def _require_own_carry(self, carry: Estimate) -> None:
        if carry.fingerprint is not None and carry.fingerprint != self._fingerprint:
            msg = (
                f"this estimate was produced by {carry.fingerprint!r} and cannot be "
                f"continued by {self._fingerprint!r}: evidence rows would be "
                "reinterpreted under the wrong unit convention; build the intended "
                "policy, or derive from raw evidence with at_evidence()"
            )
            raise ValueError(msg)

    @abstractmethod
    def _estimate_parts(
        self,
        evidence: Evidence,
        bank: int,
    ) -> tuple[dict[int, Array], Array | None]:
        """Build per-order coefficient blocks and the empty slot's value."""
        ...

    def _empty_surrogate(self) -> BasisGame:
        return BasisGame(
            MoebiusBasis(),
            {},
            self.game.n_players,
            value_shape=tuple(self.game.value_shape),
            target_shape=tuple(self.game.target_shape),
        )

    def _as_estimate(self, evidence: Evidence, bank: int) -> Estimate:
        reason: str | None = None
        try:
            attributions, empty = self._estimate_parts(evidence, bank)
            terms, coefficients = leading_blocks_to_terms(
                attributions,
                self.game.n_players,
                empty,
            )
            surrogate = BasisGame(
                MoebiusBasis(),
                None,
                self.game.n_players,
                terms=terms,
                values=coefficients,
                value_shape=tuple(self.game.value_shape),
                target_shape=tuple(self.game.target_shape),
            )
        except InsufficientSamplesError as error:
            reason = str(error)
            surrogate = self._empty_surrogate()
        return Estimate(
            surrogate,
            Provenance(
                evidence=evidence,
                bank=bank,
                index=self.index,
                deduplicated=self.deduplicate,
                shortfall=reason,
                fingerprint=self._fingerprint,
            ),
        )

    def _require_no_evidence_yet(self) -> NoReturn:
        """Raise the standard error for explaining without any evidence."""
        msg = (
            f"no evidence yet: estimate with at least {self.min_budget} evaluations first; "
            "note that policies are frozen and return the estimate: "
            "`estimate = policy.estimate(budget)`"
        )
        raise InsufficientSamplesError(msg)
