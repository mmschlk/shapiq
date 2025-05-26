"""Synthetic benchmark games based on unanimity games.

This module implements synthetic cooperative games for benchmarking interaction value methods.
It includes the Unanimity game -- a fundamental building block in cooperative game theory --
and the Sum of Unanimity Games (SOUM), a more complex game constructed as a linear
combination of multiple Unanimity games.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.games.base import Game
from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    from shapiq.game_theory.moebius_converter import MoebiusConverter


class UnanimityGame(Game):
    """The Unanimity basis game.

    Unanimity games are basis games from cooperative game theory. They are based on a single
    interaction and return 1 if the coalition contains the interaction and 0 otherwise.

    # TODO(mmshlk): add a reference and a formal description of the Unanimity game
    # https://github.com/mmschlk/shapiq/issues/387

    Attributes:
        interaction_binary: The interaction encoded in a binary vector of length ``n``.
        interaction: The interaction encoded as a tuple.

    Examples:
        >>> game = UnanimityGame(np.array([0, 1, 0, 1]))
        >>> game.n_players
        4
        >>> coalitions = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1]]
        >>> coalitions = np.array(coalitions).astype(bool)
        >>> game(coalitions)
        array([0., 0., 1., 1.])

    """

    def __init__(self, interaction_binary: np.ndarray) -> None:
        """Initializes the Unanimity game.

        Args:
            interaction_binary: The interaction encoded as a one-hot vector of shape ``(n,)``.
        """
        n = len(interaction_binary)
        self.interaction_binary: np.ndarray = interaction_binary
        self.interaction: tuple[int, ...] = tuple(np.where(self.interaction_binary == 1)[0])
        super().__init__(n_players=n, normalize=False)  # call super class which handles calls

    def value_function(self, coalitions: np.ndarray) -> np.ndarray[float]:
        """Returns 1, if the coalition contains the UnanimityGame's interaction, and zero otherwise.

        Args:
            coalitions: The coalition as a binary vector of shape (coalition_size, n_players).

        Returns:
            The worth of the coalition.
        """
        return np.prod(coalitions >= self.interaction_binary, 1)


class SOUM(Game):
    """The Sum of Unanimity Game (SOUM) game class.

    A Sum of Unanimity Game (SOUM) constructs a game based on linear combinations of so-called
    unanimity games (:class:`~shapiq.games.benchmark.synthetic.soum.UnanimityGame`).

    # TODO(mmshlk): extend description of the SOUM, add a reference and a formal description
    # https://github.com/mmschlk/shapiq/issues/387

    Attributes:
        n_players: The number of players.

        n_basis_games: The number of Unanimity gams

        unanimity_games: A dictionary containing instances of :class:`~shapiq.games.benchmark.synthetic.soum.UnanimityGame`.

        linear_coefficients: A numpy array with coefficients between -1 and 1 for the unanimity
            games.

        min_interaction_size: The smallest interaction size

        max_interaction_size: The highest interaction size.

        converter: The MoebiusConverter object to convert the SOUM to a Möbius transform. If no
            moebius transform is computed, this the convert is ``None``.

    Properties:
        moebius_coefficients: The (sparse) Möbius transform of the SOUM.

    Examples:
        >>> game = SOUM(4, 3)
        >>> game.n_players
        4
        >>> game.n_basis_games
        3
        coalitions = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1]]
        >>> coalitions = np.array(coalitions).astype(bool)
        >>> game(coalitions)
        array([0., 0.25, 1.5, 2.])  # depending on the random linear coefficients this can vary
        >>> game.moebius_coefficients
        InteractionValues(values=array([0.25, 0.25, 0.25]), index='Moebius', max_order=4, min_order=0, ...)
    """

    def __init__(
        self,
        n: int,
        n_basis_games: int,
        *,
        min_interaction_size: int | None = None,
        max_interaction_size: int | None = None,
        random_state: int | None = None,
        normalize: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initializes the SOUM game.

        Args:
            n: The number of players in the game.

            n_basis_games: The number of :class:`~shapiq.games.benchmark.synthetic.soum.UnanimityGame`
                basis games to use. The higher the number, the more complex the SOUM becomes.

            min_interaction_size: The minimum size of the interactions in the SOUM. If ``None``,
                then the default value is used. The default value is ``0``. Defaults to ``None``.

            max_interaction_size: The maximum size of the interactions in the SOUM. If ``None``,
                then the default value is used. The default value is ``n``. Defaults to ``None``.

            random_state: The random state to use for the game. If ``None``, then the default value
                is used. The default value is ``42``. Defaults to ``None``.

            normalize: A boolean flag to normalize/center the game values. Defaults to ``False``.

            verbose: A flag to print information from the game. Defaults to ``False``.
        """
        self._rng = np.random.default_rng(random_state)

        # set min_interaction_size and max_interaction_size to 0 and n if not specified
        self.min_interaction_size = min_interaction_size if min_interaction_size is not None else 0
        self.max_interaction_size = max_interaction_size if max_interaction_size is not None else n

        # setup basis games
        self.n_basis_games: int = n_basis_games
        self.unanimity_games = {}
        self.linear_coefficients = self._rng.random(size=self.n_basis_games) * 2 - 1
        # Compute interaction sizes (exclude size 0)
        interaction_sizes = self._rng.integers(
            low=self.min_interaction_size,
            high=self.max_interaction_size,
            size=self.n_basis_games,
            endpoint=True,
        )
        for i, size in enumerate(interaction_sizes):
            interaction = self._rng.choice(tuple(range(n)), size, replace=False)
            interaction_binary = np.zeros(n, dtype=int)
            interaction_binary[interaction] = 1
            self.unanimity_games[i] = UnanimityGame(interaction_binary)

        # will store the Möbius transform
        self._moebius_coefficients: InteractionValues | None = None
        self.converter: MoebiusConverter | None = None

        # init base game
        empty_value = float(self.value_function(np.zeros((1, n)))[0])
        super().__init__(
            n_players=n,
            normalize=normalize,
            verbose=verbose,
            normalization_value=empty_value,
        )

    @property
    def moebius_coefficients(self) -> InteractionValues:
        """Return the (sparse) Möbius transform of the SOUM."""
        if self._moebius_coefficients is None:
            self._moebius_coefficients = self.moebius_transform()
        return self._moebius_coefficients

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Computes the worth of the coalition for the SOUM.

        The worth of a coalition for the SOUM sums up all linear coefficients, if a coalition
        contains the interaction of a corresponding unanimity game.

        Args:
            coalitions: The coalition as a binary vector of shape ``(coalition_size, n)``.

        Returns:
            The worth of the coalition.
        """
        worth = np.zeros(coalitions.shape[0])
        for i, game in self.unanimity_games.items():
            worth += self.linear_coefficients[i] * game(coalitions)
        return worth

    def exact_values(self, index: str, order: int) -> InteractionValues:
        """Computes the exact values for the given index and order.

        Args:
            index: The index to compute the values for.
            order: The order to compute the values for.

        Returns:
            The exact values for the given index and order.

        """
        from shapiq.game_theory.moebius_converter import MoebiusConverter

        if self.converter is None:
            self.converter = MoebiusConverter(self.moebius_coefficients)
        return self.converter(index, order)

    def moebius_transform(self) -> InteractionValues:
        """Computes the (sparse) Möbius transform of the SOUM.

        The Möbius transform is computed for the SOUM via its UnanimityGames. This is helpful for
        ground truth calculations of interaction indices.

        Returns:
            An InteractionValues object containing all non-zero Möbius coefficients of the SOUM.

        """
        # fill the moebius coefficients dict from the game
        moebius_coefficients_dict = {}
        for i, game in self.unanimity_games.items():
            if game.interaction in moebius_coefficients_dict:
                moebius_coefficients_dict[game.interaction] += self.linear_coefficients[i]
            else:
                moebius_coefficients_dict[game.interaction] = self.linear_coefficients[i]

        # generate the lookup for the moebius values
        moebius_coefficients_values = np.zeros(len(moebius_coefficients_dict))
        moebius_coefficients_lookup = {}
        for i, (key, val) in enumerate(moebius_coefficients_dict.items()):
            moebius_coefficients_values[i] = val
            moebius_coefficients_lookup[key] = i

        # handle baseline value and set to 0 if no empty set is present
        baseline_value = moebius_coefficients_dict.get((), 0.0)

        return InteractionValues(
            values=moebius_coefficients_values,
            index="Moebius",
            max_order=self.n_players,
            min_order=0,
            n_players=self.n_players,
            interaction_lookup=moebius_coefficients_lookup,
            estimated=False,
            baseline_value=baseline_value,
        )
