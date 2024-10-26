"""Game Explainer class for the shapiq package."""

import warnings
from typing import Optional

import numpy as np

from shapiq.explainer._base import Explainer
from shapiq.approximator._base import Approximator
from shapiq.games.base import Game
from shapiq.interaction_values import InteractionValues


class GameExplainer(Explainer):
    """The game explainer as the advanced interface for the shapiq package.
    Args:
        game: The model to be explained as a callable function expecting data points as input and
            returning 1-dimensional predictions.
        approximator: An approximator object to use for the explainer. 
        **kwargs: Additional keyword-only arguments.

    Attributes:
        baseline_value: A baseline value of the explainer.
    """

    def __init__(
        self,
        game: Game,
        approximator: Approximator,
        **kwargs,
    ) -> None:
        self._game = game
        self._approximator = approximator

    @property
    def baseline_value(self) -> float:
        """Returns the baseline value of the explainer."""
        return self._game.empty_prediction

    def explain(
        self, x: np.ndarray, 
        budget: Optional[int] = None, 
        random_state: Optional[int] = None,
        **kwargs
    ) -> InteractionValues:
        """Explains the model's predictions.

        Args:
            x: The data point to explain as a 2-dimensional array with shape (1, n_features).
            budget: The budget to use for the approximation. Defaults to `None`, which will
                set the budget to 2**n_features based on the number of features.
            **kwargs: Additional keyword-only arguments.
        """
        if budget is None:
            budget = 2**self._game.n_players
            if budget > 2048:
                warnings.warn(
                    f"Using the budget of 2**n_features={budget}, which might take long\
                              to compute. Set the `budget` parameter to suppress this warning."
                )
        if random_state is not None:
            self._approximator._rng = np.random.default_rng(random_state)
            self._approximator._sampler._rng = np.random.default_rng(random_state)

        # initialize the game with the explanation point
        self._game._x = x

        # explain
        interaction_values = self._approximator.approximate(budget=budget, game=self._game, **kwargs)
        interaction_values.baseline_value = self.baseline_value

        return interaction_values