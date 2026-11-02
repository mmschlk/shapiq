"""Games with boolean and packed finite-coalition evaluation hooks."""

from abc import abstractmethod
from typing import override

import jax
import jax.numpy as jnp
from beartype.door import is_bearable
from jaxtyping import Bool, Shaped, UInt8

from shapiq.coalition import FiniteCoalition
from shapiq.coalition.coalition import (
    pack_bool_coalition,
    unpack_bool_coalition,
)
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value import Value

from .game import Game


class FinitePlayerGame[P, V: Value](Game[FiniteCoalition, V]):
    """A game accepting boolean and little-endian packed finite coalitions.

    Implement both evaluation hooks for native support of both representations,
    or derive from ``BoolFinitePlayerGame`` or ``PackedFinitePlayerGame`` to
    implement just the corresponding native hook and inherit a conversion for
    the other representation. Hooks own value-batch broadcasting; public
    ``eval`` validates coalitions against the players and game batch and checks
    that the returned value's logical shape is the broadcasted batch. Packed
    implementations must ignore unused bits in the final input byte.
    Evaluation checks these contracts directly; ``is_valid`` is an explicit
    predicate and is not called during evaluation.
    """

    @property
    @abstractmethod
    @override
    def players(self) -> FinitePlayers[P]:
        """Return finite player sets whose logical shape broadcasts to ``shape``."""

    @property
    def n_players(self) -> int:
        """Return the number of players in each game, which is constant across the batch."""
        return self.players.n_players

    @typechecked
    @override
    def _eval(
        self,
        coalition: Bool[jax.Array, "#*batch {self.n_players}"]
        | UInt8[jax.Array, "#*batch ({self.n_players}+7)//8"],
    ) -> Shaped[V, "*batch"]:
        """Validate and dispatch finite coalitions to the matching evaluation hook.

        Raises:
            ValueError: The coalition batch is inadmissible for this game.
            TypeCheckError: The coalition representation or returned value is invalid.
        """
        if not is_bearable(self, Shaped[FinitePlayerGame[P, V], "#*batch"]):
            msg = "Inadmissible coalition batch."
            raise ValueError(msg)
        if coalition.dtype == jnp.bool_:
            return self._eval_bool(coalition)
        return self._eval_packed(coalition)

    @abstractmethod
    def _eval_bool(
        self, coalition: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Evaluate boolean membership, broadcasting with the game batch."""

    @abstractmethod
    def _eval_packed(
        self, coalition: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        """Evaluate packed membership, ignoring unused input padding bits."""


class BoolFinitePlayerGame[P, V: Value](FinitePlayerGame[P, V]):
    """A finite-player game with a native ``_eval_bool`` implementation.

    Packed input is unpacked automatically. Subclasses may also override
    ``_eval_packed`` to support both representations natively.
    """

    @override
    def _eval_packed(
        self, coalition: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        """Unpack exactly the local player count and defer to boolean evaluation."""
        return self._eval_bool(unpack_bool_coalition(coalition, self.n_players))


class PackedFinitePlayerGame[P, V: Value](FinitePlayerGame[P, V]):
    """A finite-player game with a native ``_eval_packed`` implementation.

    Boolean input is packed automatically. Subclasses may also override
    ``_eval_bool`` to support both representations natively.
    """

    @override
    def _eval_bool(
        self, coalition: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Pack boolean membership and defer to packed evaluation."""
        return self._eval_packed(pack_bool_coalition(coalition))
