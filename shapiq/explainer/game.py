"""This module contains the game explainer class for the shapiq package."""

import warnings
from collections.abc import Callable

import numpy as np

from .. import InteractionValues
from ..approximator import Approximator
from ..games.base import Game
from ._base import Explainer


class GameExplainer(Explainer):

    def __init__(
        self,
        game: Game | Callable[[np.ndarray], np.ndarray],
        n_players: int | None = None,
        approximator: str | Approximator = "auto",
        index: str = "k-SII",
        max_order: int = 2,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """

        Args:
            game: The game object to be explained as an instance of a Game class or a callable
            function which maps coalitions of players to the game value.
        """
        self.game = game
        try:
            n_players_game = game.n_players
            if n_players is None and n_players_game != n_players:
                warnings.warn(
                    f"The number of players provided ({n_players}) does not match the number of "
                    f"players in the game object ({n_players_game}). Using "
                    f"n_players={n_players_game}."
                )
            n_players = n_players_game
        except AttributeError:
            if n_players is None:
                raise ValueError(
                    "The number of players must be provided if the game object does not have "
                    "an attribute `n_players`."
                )
        self.n_players = n_players

        self._approximator = self._init_approximator(approximator, self.index, self._max_order)

        super().__init__(**kwargs)

    def explain_function(self, *args, **kwargs) -> InteractionValues:
        """Explain the game function."""
