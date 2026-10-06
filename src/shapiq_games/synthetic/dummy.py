"""The dummy game: an additive game with an optional unanimity interaction."""

from __future__ import annotations

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions


class DummyGame(Game):
    r"""An additive game plus an optional interaction, with closed-form Shapley values.

    The value of a coalition :math:`S` is

    .. math::
        v(S) = \frac{|S|}{n} + \mathbb{1}[I \subseteq S],

    where :math:`I` is the (optional) interaction. The Shapley value of a player :math:`i` is
    :math:`1/n + 1/|I|` if :math:`i \in I` and :math:`1/n` otherwise. The game is not normalized:
    :math:`v(\emptyset) = 0` unless the interaction is empty, in which case it is ``1``.

    Attributes:
        n: The number of players.
        N: The set of players ``{0, ..., n - 1}``.
        interaction: The interaction as a sorted tuple of player indices.
        access_counter: The number of coalitions evaluated so far (used by tests to check that
            approximators respect their budget).

    Examples:
        >>> game = DummyGame(4, interaction=(1, 2))
        >>> game(np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]], dtype=bool))
        array([0.  , 0.25, 1.5 , 2.  ])
    """

    def __init__(self, n: int, interaction: set[int] | tuple[int, ...] = ()) -> None:
        """Initialize the dummy game.

        Args:
            n: The number of players.
            interaction: The interaction as a set or tuple of player indices. Defaults to no
                interaction.
        """
        self.n = n
        self.N = set(range(n))
        self.interaction: tuple[int, ...] = tuple(sorted(interaction))
        super().__init__(n, normalize=False)
        self.access_counter = 0

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return ``|S| / n`` plus one if the coalition contains the interaction."""
        coalitions = as_bool_coalitions(coalitions)
        worth = np.sum(coalitions, axis=1) / self.n
        if self.interaction:
            worth = worth + np.all(coalitions[:, self.interaction], axis=1)
        self.access_counter += coalitions.shape[0]
        return worth
