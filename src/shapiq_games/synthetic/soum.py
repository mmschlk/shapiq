"""Unanimity games and sums of unanimity games (SOUM).

Both games have a known Möbius representation, which makes them the analytic ground truth for
interaction indices: every index can be computed from the Möbius coefficients exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions

from ._moebius import moebius_values

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from shapiq.interaction_values import InteractionValues
    from shapiq.typing import BoolVector, CoalitionMatrix, GameValues

__all__ = ["SOUM", "UnanimityGame"]


class UnanimityGame(Game):
    r"""The unanimity game of an interaction :math:`T`.

    .. math::
        v(S) = \mathbb{1}[T \subseteq S]

    Its only non-zero Möbius coefficient is :math:`m(T) = 1`, and its Shapley value is
    :math:`1/|T|` for every player in :math:`T` and zero otherwise.

    Attributes:
        interaction_binary: The interaction :math:`T` as a binary vector of length ``n``.
        interaction: The interaction as a tuple of player indices.

    Examples:
        >>> game = UnanimityGame(np.array([0, 1, 0, 1]))
        >>> game(np.array([[0, 0, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1]], dtype=bool))
        array([0., 1., 1.])
    """

    def __init__(self, interaction_binary: ArrayLike) -> None:
        """Initialize the unanimity game.

        Args:
            interaction_binary: The interaction encoded as a binary vector of shape ``(n,)``.
        """
        self.interaction_binary: BoolVector = np.asarray(interaction_binary).astype(bool)
        self.interaction: tuple[int, ...] = tuple(
            int(i) for i in np.flatnonzero(self.interaction_binary)
        )
        super().__init__(n_players=len(self.interaction_binary), normalize=False)

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return one if the coalition contains the interaction, zero otherwise."""
        coalitions = as_bool_coalitions(coalitions)
        return np.all(coalitions[:, self.interaction_binary], axis=1).astype(float)

    @property
    def moebius_coefficients(self) -> InteractionValues:
        """The Möbius transform: one for the interaction, zero for every other coalition."""
        return moebius_values({self.interaction: 1.0}, self.n_players)


class SOUM(Game):
    r"""A sum of unanimity games (SOUM) with random coefficients.

    .. math::
        v(S) = \sum_{k=1}^{K} c_k \cdot \mathbb{1}[T_k \subseteq S]

    The coefficients :math:`c_k` are drawn uniformly from :math:`[-1, 1]` and the interactions
    :math:`T_k` uniformly with sizes between ``min_interaction_size`` and
    ``max_interaction_size``. The Möbius representation of the game is available as
    :attr:`moebius_coefficients`, from which every interaction index follows exactly.

    Attributes:
        n_basis_games: The number of unanimity games :math:`K`.
        unanimity_games: The unanimity games, keyed by their position.
        linear_coefficients: The coefficients :math:`c_k`.
        min_interaction_size: The smallest interaction size that can be drawn.
        max_interaction_size: The largest interaction size that can be drawn.

    Examples:
        >>> game = SOUM(n=6, n_basis_games=10, max_interaction_size=3, random_state=0)
        >>> game.moebius_coefficients.index
        'Moebius'
    """

    def __init__(
        self,
        n: int,
        n_basis_games: int,
        *,
        min_interaction_size: int | None = None,
        max_interaction_size: int | None = None,
        random_state: int | None = 42,
        normalize: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize the SOUM.

        Args:
            n: The number of players.
            n_basis_games: The number of unanimity games.
            min_interaction_size: The smallest interaction size. Defaults to ``0``, which allows
                a constant term (an interaction with the empty set).
            max_interaction_size: The largest interaction size. Defaults to ``n``.
            random_state: The seed of the coefficients and interactions. Defaults to ``42``.
            normalize: Whether to center the game such that the value of the empty coalition is zero. Defaults
                to ``False``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        rng = np.random.default_rng(random_state)
        self.min_interaction_size = 0 if min_interaction_size is None else min_interaction_size
        self.max_interaction_size = n if max_interaction_size is None else max_interaction_size
        self.n_basis_games: int = n_basis_games
        self.linear_coefficients = rng.random(size=n_basis_games) * 2 - 1
        sizes = rng.integers(
            low=self.min_interaction_size,
            high=self.max_interaction_size,
            size=n_basis_games,
            endpoint=True,
        )
        self.unanimity_games: dict[int, UnanimityGame] = {}
        for k, size in enumerate(sizes):
            interaction = rng.choice(n, size, replace=False)
            interaction_binary = np.zeros(n, dtype=bool)
            interaction_binary[interaction] = True
            self.unanimity_games[k] = UnanimityGame(interaction_binary)
        self._moebius_coefficients: InteractionValues | None = None

        empty_value = float(self.value_function(np.zeros((1, n), dtype=bool))[0])
        super().__init__(
            n_players=n,
            normalize=normalize,
            normalization_value=empty_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Sum the coefficients of the unanimity games whose interaction is in the coalition."""
        coalitions = as_bool_coalitions(coalitions)
        worth = np.zeros(coalitions.shape[0])
        for k, game in self.unanimity_games.items():
            worth += self.linear_coefficients[k] * game.value_function(coalitions)
        return worth

    @property
    def moebius_coefficients(self) -> InteractionValues:
        """The (sparse) Möbius transform of the unnormalized game."""
        if self._moebius_coefficients is None:
            self._moebius_coefficients = self.moebius_transform()
        return self._moebius_coefficients

    def moebius_transform(self) -> InteractionValues:
        """Compute the Möbius transform of the unnormalized game from its unanimity games.

        Returns:
            The non-zero Möbius coefficients. The coefficient of the empty set, if any, is also
            the baseline value.
        """
        coefficients: dict[tuple[int, ...], float] = {}
        for k, game in self.unanimity_games.items():
            coefficients[game.interaction] = (
                coefficients.get(game.interaction, 0.0) + self.linear_coefficients[k]
            )
        return moebius_values(coefficients, self.n_players)
