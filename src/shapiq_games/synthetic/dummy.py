"""The dummy game: an additive game with an optional unanimity interaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions

from ._moebius import moebius_values

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues
    from shapiq.typing import CoalitionMatrix, GameValues

__all__ = ["DummyGame"]


class DummyGame(Game):
    r"""An additive game plus an optional interaction, with closed-form Shapley values.

    The value of a coalition :math:`S` is

    .. math::
        v(S) = \frac{|S|}{n} + \mathbb{1}[I \subseteq S],

    where :math:`I` is the (optional) interaction; without an interaction the second term is
    omitted. The Shapley value of a player :math:`i` is :math:`1/n + 1/|I|` if :math:`i \in I` and
    :math:`1/n` otherwise, and :math:`v(\emptyset) = 0`.

    Attributes:
        n: The number of players.
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
        self.interaction: tuple[int, ...] = tuple(sorted(interaction))
        super().__init__(n, normalize=False)
        self.access_counter = 0

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return ``|S| / n`` plus one if the coalition contains the interaction."""
        coalitions = as_bool_coalitions(coalitions)
        worth = np.sum(coalitions, axis=1) / self.n
        if self.interaction:
            worth = worth + np.all(coalitions[:, self.interaction], axis=1)
        self.access_counter += coalitions.shape[0]
        return worth

    @property
    def moebius_coefficients(self) -> InteractionValues:
        """The Möbius transform: ``1 / n`` for every player and one for the interaction."""
        coefficients: dict[tuple[int, ...], float] = {(i,): 1.0 / self.n for i in range(self.n)}
        if self.interaction:
            coefficients[self.interaction] = coefficients.get(self.interaction, 0.0) + 1.0
        return moebius_values(coefficients, self.n)
