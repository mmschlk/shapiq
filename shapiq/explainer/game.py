"""This module contains the game explainer class for the shapiq package."""

import warnings
from collections.abc import Callable

import numpy as np

from .. import InteractionValues
from ..approximator import Approximator
from ..games.base import Game
from ._base import Explainer
from .setup import setup_approximator
from .validation import set_random_state, validate_budget


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

        super().__init__(
            model=game,
            data=None,
            class_index=None,
            index=index,
            max_order=max_order,
            random_state=random_state,
            verbose=verbose,
        )

        self._approximator = setup_approximator(
            approximator,
            index=self.index,
            max_order=self._max_order,
            n_players=self.n_players,
            random_state=self._random_state,
        )

    def explain_function(
        self, budget: int | None = None, random_state: int | None = None, *args, **kwargs
    ) -> InteractionValues:
        """Explain the game function."""
        budget = validate_budget(budget, n_players=self.n_players)
        set_random_state(random_state, object_with_rng=self)
        approximation = self._approximator(game=self.game, budget=budget)
        return approximation
