"""This module contains the Owen Sampling approximation method for the Shapley value (SV) by Okhrati and Lipani (2020).
It estimates the Shapley values in its integral representation by sampling random marginal contributions."""

from typing import Callable, Optional

import numpy as np

from shapiq.approximator._base import Approximator
from shapiq.interaction_values import InteractionValues

AVAILABLE_INDICES_SHAPIQ = {"SV"}


class OwenSamplingSV(Approximator):
    """The Owen Sampling algorithm estimates the Shapley values (SV) by sampling random marginal contributions
    for each player and each coalition size. The marginal contributions are used to update an integral representation of the SV.
    For more information, see [Okhrati and Lipani (2020)](https://www.computer.org/csdl/proceedings-article/icpr/2021/09412511/1tmicWxYo2Q).
    The number of points M at which the integral is to be palpated share the avilable budget for each player equally.
    A higher M increases the resolution of the integral reducing bias while reducing the accuracy of the estimation at each point.

    Args:
        n: The number of players.
        random_state: The random state to use for the permutation sampling. Defaults to `None`.
        M: Number of points at which the integral is to be palpated.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        N_arr: The array of players (starting from 0 to n).
        iteration_cost: The cost of a single iteration of the approximator.

    Examples:
        >>> from shapiq.approximator import StratifiedSamplingSV
        >>> from shapiq.games import DummyGame
        >>> game = DummyGame(5, (1, 2))
        >>> approximator = OwenSamplingSV(game.n_players, 10, random_state=42)
        >>> sv_estimates = approximator.approximate(100, game)
        >>> print(sv_estimates.values)
        [0.2 0.7 0.7 0.2 0.2]
    """

    def __init__(
        self,
        n: int,
        M: int,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(n, max_order=1, index="SV", top_order=False, random_state=random_state)
        self.iteration_cost: int = 2 * n * M
        self.M = M

    def approximate(
        self, budget: int, game: Callable[[np.ndarray], np.ndarray], batch_size: Optional[int] = 5
    ) -> InteractionValues:
        """Approximates the Shapley values using Owen Sampling.

        Args:
            budget: The number of game evaluations for approximation
            game: The game function as a callable that takes a set of players and returns the value.
            batch_size: The size of the batch. If None, the batch size is set to 1. Defaults to 5.

        Returns:
            The estimated interaction values.
        """

        used_budget = 0
        batch_size = 1 if batch_size is None else batch_size

        # compute the number of iterations and size of the last batch (can be smaller than original)
        n_iterations, last_batch_size = self._calc_iteration_count(
            budget - 2, batch_size, self.iteration_cost
        )

        #anchors = np.zeros((self.n, self.M), dtype=float)
        #counts = np.zeros((self.n, self.M), dtype=int)

        result = np.zeros(self.n)

        return self._finalize_result(result, budget=used_budget, estimated=True)

    def get_anchor_points(self, M: int):
        if M <= 0:
            raise ValueError("The number of anchor points needs to be greater than 0.")

        if M == 1:
            return np.array([0.5])

        step_size = 1.0 / (M - 1.0)
        return np.arange(0.0, step_size, 1.0 + step_size)
