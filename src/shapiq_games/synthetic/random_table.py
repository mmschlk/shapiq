"""A game with random but fixed values for every coalition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, GameValues

__all__ = ["RandomTableGame"]

_MAX_PLAYERS = 20


class RandomTableGame(Game):
    r"""A game whose value table is drawn once at random.

    The values of all :math:`2^n` coalitions are drawn uniformly from ``[low, high)`` at
    construction, so the game is a proper set function: the value of a coalition never depends on
    which other coalitions are evaluated with it or in which order. The game has no structure and
    is useful to test methods on a generic game. It supports at most 20 players.

    Attributes:
        values: The value table, indexed by the binary encoding of the coalitions
            (player ``i`` contributes ``2**i``).

    Examples:
        >>> game = RandomTableGame(4, random_state=0)
        >>> bool(game(np.ones((1, 4), dtype=bool))[0] == game.values[-1])
        True
    """

    def __init__(
        self,
        n: int,
        *,
        low: float = 0.0,
        high: float = 1.0,
        random_state: int | None = 42,
        normalize: bool = False,
    ) -> None:
        """Initialize the random table game.

        Args:
            n: The number of players, at most 20.
            low: The lower bound of the values. Defaults to ``0``.
            high: The upper bound of the values. Defaults to ``1``.
            random_state: The seed of the value table. Defaults to ``42``.
            normalize: Whether to center the game such that the value of the empty coalition is zero.
        """
        if not 1 <= n <= _MAX_PLAYERS:
            msg = f"RandomTableGame supports 1 to {_MAX_PLAYERS} players, got {n}."
            raise ValueError(msg)
        rng = np.random.default_rng(random_state)
        self.values: GameValues = rng.uniform(low, high, size=2**n)
        self._powers = 2 ** np.arange(n)
        super().__init__(n, normalize=normalize, normalization_value=float(self.values[0]))

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Look up the values of the coalitions in the value table."""
        coalitions = as_bool_coalitions(coalitions)
        return self.values[coalitions.astype(np.int64) @ self._powers]
