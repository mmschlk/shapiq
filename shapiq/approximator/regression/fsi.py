"""This module contains the regression algorithms to estimate FSI scores."""
from typing import Optional, Callable

import numpy as np
from scipy.special import binom

from approximator._base import Approximator, ShapleySamplingMixin, InteractionValues
from utils import powerset


class RegressionFSI(Approximator, ShapleySamplingMixin):
    """Estimates the FSI values using the weighted least square approach.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        random_state: The random state of the estimator. Defaults to `None`.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        max_order: The interaction order of the approximation.
        min_order: The minimum order of the approximation. For FSI, min_order is equal to 1.
        iteration_cost: The cost of a single iteration of the regression FSI.

    Example:
        >>> from games import DummyGame
        >>> from approximator import RegressionFSI
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = RegressionFSI(n=5, max_order=2)
        >>> approximator.approximate(budget=100, game=game)
        InteractionValues(
            index=FSI, order=2, estimated=False, estimation_budget=32,
            values={
                (0,): 0.2,
                (1,): 0.2,
                (2,): 0.2,
                (3,): 0.2,
                (4,): 0.2,
                (0, 1): 0,
                (0, 2): 0,
                (0, 3): 0,
                (0, 4): 0,
                (1, 2): 1.0,
                (1, 3): 0,
                (1, 4): 0,
                (2, 3): 0,
                (2, 4): 0,
                (3, 4): 0
            }
        )
    """

    def __init__(
        self,
        n: int,
        max_order: int,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            n, max_order=max_order, index="FSI", top_order=False, random_state=random_state
        )
        self.iteration_cost: int = 1

    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
        batch_size: Optional[int] = None,
        replacement: bool = False,
        pairing: bool = True,
    ) -> InteractionValues:
        """Approximates the interaction values.

        Args:
            budget: The budget of the approximation (how many times the game is queried). The game
                is always queried for the empty and full set (`budget += 2`).
            game: The game to be approximated.
            batch_size: The batch size for the approximation. Defaults to `None`. If `None` the
                batch size is set to the approximation budget.
            replacement: Whether to sample subsets with replacement (`True`) or without replacement
                (`False`). Defaults to `False`.
            pairing: Whether to use the pairing sampling strategy or not. If paired sampling
                (`True`) is used a subset is always paired with its complement subset and sampled
                together. This may increase approximation quality. Defaults to `True`.

        Returns:
            The interaction values.

        Raises:
            np.linalg.LinAlgError: If the regression fails.
        """
        # validate input parameters
        batch_size = budget + 2 if batch_size is None else batch_size
        used_budget = 0

        # generate the dataset containing explicit and sampled subsets
        all_subsets, estimation_flag, n_explicit_subsets = self._generate_shapley_dataset(
            budget, pairing, replacement
        )
        n_subsets = all_subsets.shape[0]

        # calculate the number of iterations and the last batch size
        n_iterations, last_batch_size = self._calc_iteration_count(
            n_subsets, batch_size, iteration_cost=self.iteration_cost
        )

        # get the fsi representation of the subsets
        regression_subsets, num_players = self._get_fsi_subset_representation(all_subsets)  # S, m
        regression_weights = self._get_ksh_subset_weights(all_subsets)  # W(|S|)

        # initialize the regression variables
        game_values: np.ndarray[float] = np.zeros(shape=(n_subsets,), dtype=float)  # \nu(S)
        fsi_values: np.ndarray[float] = np.zeros(shape=(num_players,), dtype=float)

        # main regression loop computing the FSI values
        for iteration in range(1, n_iterations + 1):
            batch_size = batch_size if iteration != n_iterations else last_batch_size
            batch_index = (iteration - 1) * batch_size

            # query the game for the batch of subsets
            batch_subsets = all_subsets[batch_index : batch_index + batch_size]
            game_values[batch_index : batch_index + batch_size] = game(batch_subsets)

            # compute the FSI values up to now
            A = regression_subsets[0 : batch_index + batch_size]
            B = game_values[0 : batch_index + batch_size]
            W = regression_weights[0 : batch_index + batch_size]
            W = np.sqrt(np.diag(W))
            Aw = np.dot(W, A)
            Bw = np.dot(W, B)

            fsi_values = np.linalg.lstsq(Aw, Bw, rcond=None)[0]  # \phi_i

            used_budget += batch_size

        return self._finalize_result(fsi_values, budget=used_budget, estimated=estimation_flag)

    def _get_fsi_subset_representation(
        self, all_subsets: np.ndarray[bool]
    ) -> tuple[np.ndarray[bool], int]:
        """Transforms a subset matrix into the FSI representation.

        The FSI representation is a matrix of shape (n_subsets, num_players) where each interaction
        up to the maximum order is an individual player.

        Args:
            all_subsets: subset matrix in shape (n_subsets, n).

        Returns:
            FSI representation of the subset matrix in shape (n_subsets, num_players).
        """
        n_subsets = all_subsets.shape[0]
        num_players = sum(int(binom(self.n, order)) for order in range(1, self.max_order + 1))
        regression_subsets = np.zeros(shape=(n_subsets, num_players), dtype=bool)
        for interaction_index, interaction in enumerate(
            powerset(self.N, min_size=1, max_size=self.max_order)
        ):
            regression_subsets[:, interaction_index] = all_subsets[:, interaction].all(axis=1)
        return regression_subsets, num_players


if __name__ == "__main__":
    import doctest

    doctest.testmod()
