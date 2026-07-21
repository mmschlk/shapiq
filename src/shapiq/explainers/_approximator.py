from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from typing import TYPE_CHECKING

from shapiq._shape import ensure_bool, validate_int
from shapiq.errors import HistoryError
from shapiq.explainers._base import Explainer
from shapiq.games import Game
from shapiq.sampling import ApproximationState, Sampler

if TYPE_CHECKING:
    from shapiq.coalitions import CoalitionArray
    from shapiq.explanations import ExplanationArray
    from shapiq.interactions import InteractionIndex


class Approximator[
    ValueT,
    GameT: Game,
    StateT: ApproximationState,
    SamplerT: Sampler,
](Explainer[ValueT, GameT], ABC):
    """Base abstraction for sampling-based explainers."""

    state: StateT
    sampler: SamplerT
    _sampler_history: tuple[SamplerT, ...] | None

    def __init__(
        self,
        game: GameT,
        sampler: SamplerT,
        state: StateT,
        index: InteractionIndex,
    ) -> None:
        """Initialize an approximator."""
        super().__init__(game, index)
        if sampler.n_players != game.n_players:
            msg = "sampler and game use different numbers of players"
            raise ValueError(msg)
        if sampler.target_shape != game.target_shape:
            msg = "sampler and game use different target shapes"
            raise ValueError(msg)
        if state.track_history and (state.mutable or sampler.mutable):
            msg = "history cannot be enabled with mutable state or sampler"
            raise HistoryError(msg)
        self.sampler = sampler
        self.state = state
        self._sampler_history = (sampler,) if state.track_history else None

    def sample(self, budget: int) -> Approximator[ValueT, GameT, StateT, SamplerT]:
        """Sample and evaluate additional coalitions."""
        validate_int("budget", budget)
        if budget == 0:
            return self
        coalitions, next_sampler = self.sampler.sample(self.state, budget)
        values = self._evaluate(coalitions)
        next_state = self._append_state(coalitions, values)
        next_history = self._next_sampler_history(next_sampler)
        return self._replace(state=next_state, sampler=next_sampler, sampler_history=next_history)

    def _evaluate(self, coalitions: CoalitionArray) -> ValueT:
        """Evaluate the game; subclasses may normalize the value layout here."""
        return self.game(coalitions)

    def approximate(self, budget: int) -> ExplanationArray[ValueT]:
        """Sample a budget and return explanations."""
        return self.sample(budget).explain()

    def rollback(self, steps: int = 1) -> Approximator[ValueT, GameT, StateT, SamplerT]:
        """Return a previous approximator state."""
        rolled_state = self.state.rollback(steps)
        history = self._require_sampler_history()
        if steps >= len(history):
            msg = "cannot roll back past the initial approximator"
            raise HistoryError(msg)
        return self._replace(
            state=rolled_state,  # type: ignore[arg-type]
            sampler=history[-1 - steps],
            sampler_history=history[: len(history) - steps],
        )

    def history(
        self,
        *,
        reverse: bool = False,
        include_self: bool = True,
    ) -> list[Approximator[ValueT, GameT, StateT, SamplerT]]:
        """Return value-equivalent approximator history."""
        ensure_bool("reverse", reverse)
        ensure_bool("include_self", include_self)
        state_history = self.state.history(reverse=reverse, include_self=include_self)
        sampler_history = list(self._require_sampler_history())
        if not include_self:
            sampler_history = sampler_history[:-1]
        if reverse:
            sampler_history.reverse()
        if len(state_history) != len(sampler_history):
            msg = "state and sampler history lengths differ"
            raise HistoryError(msg)
        return [
            self._replace(
                state=state,  # type: ignore[arg-type]
                sampler=sampler,
                sampler_history=tuple(sampler_history[: index + 1]),
            )
            for index, (state, sampler) in enumerate(
                zip(state_history, sampler_history, strict=True)
            )
        ]

    @abstractmethod
    def _append_state(self, coalitions: CoalitionArray, values: ValueT) -> StateT:
        """Return the state after incorporating sampled evaluations."""

    def _next_sampler_history(self, next_sampler: SamplerT) -> tuple[SamplerT, ...] | None:
        if not self.state.track_history:
            return None
        history = self._require_sampler_history()
        return (*history, next_sampler)

    def _require_sampler_history(self) -> tuple[SamplerT, ...]:
        if not self.state.track_history or self._sampler_history is None:
            msg = "history is not enabled; construct the approximator with track_history=True"
            raise HistoryError(msg)
        return self._sampler_history

    def _replace(
        self,
        *,
        state: StateT,
        sampler: SamplerT,
        sampler_history: tuple[SamplerT, ...] | None,
    ) -> Approximator[ValueT, GameT, StateT, SamplerT]:
        clone = copy(self)
        clone.state = state
        clone.sampler = sampler
        clone._sampler_history = sampler_history  # noqa: SLF001
        return clone
