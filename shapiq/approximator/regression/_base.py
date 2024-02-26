"""This module contains the regression algorithms to estimate FSI and SII scores."""

from typing import Callable, Optional

import numpy as np
from approximator._base import Approximator
from approximator.sampling import ShapleySamplingMixin
from interaction_values import InteractionValues
from scipy.special import bernoulli, binom
from utils import powerset

AVAILABLE_INDICES_REGRESSION = ["FSI", "SII", "SV"]


class Regression(Approximator, ShapleySamplingMixin):
    """Estimates the InteractionScores values using the weighted least square approach.

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
        >>> from approximator import RegressionSII
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = RegressionSII(n=5, max_order=2)
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
        index: str = "FSI",
        random_state: Optional[int] = None,
    ) -> None:
        if index not in AVAILABLE_INDICES_REGRESSION:
            raise ValueError(
                f"Index {index} not available for regression. Choose from "
                f"{AVAILABLE_INDICES_REGRESSION}."
            )
        super().__init__(
            n, max_order=max_order, index=index, top_order=False, random_state=random_state
        )
        self.iteration_cost: int = 1
        self._bernoulli_numbers = bernoulli(self.n)  # used for SII

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
        regression_weights = self._get_ksh_subset_weights(all_subsets)  # W(|S|)

        # if SII is used regression_subsets needs to be changed
        if self.index == "SII":
            regression_subsets, num_players = self._get_sii_subset_representation(all_subsets)  # A
        else:  # FSI or SV
            regression_subsets, num_players = self._get_fsi_subset_representation(all_subsets)  # A

        # initialize the regression variables
        game_values: np.ndarray[float] = np.zeros(shape=(n_subsets,), dtype=float)  # \nu(S)
        result: np.ndarray[float] = np.zeros(shape=(num_players,), dtype=float)

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

            result = np.linalg.lstsq(Aw, Bw, rcond=None)[0]  # \phi_i

            used_budget += batch_size

        return self._finalize_result(result, budget=used_budget, estimated=estimation_flag)

    def _get_fsi_subset_representation(
        self, all_subsets: np.ndarray[bool]
    ) -> tuple[np.ndarray[bool], int]:
        """Transforms a subset matrix into the FSI representation.

        The FSI representation is a matrix of shape (n_subsets, num_players) where each interaction
        up to the maximum order is an individual player.

        Args:
            all_subsets: subset matrix in shape (n_subsets, n).

        Returns:
            FSI representation of the subset matrix in shape (n_subsets, num_players) and the number
            of players.
        """
        n_subsets = all_subsets.shape[0]
        num_players = sum(int(binom(self.n, order)) for order in range(1, self.max_order + 1))
        regression_subsets = np.zeros(shape=(n_subsets, num_players), dtype=bool)
        for interaction_index, interaction in enumerate(
            powerset(self.N, min_size=1, max_size=self.max_order)
        ):
            regression_subsets[:, interaction_index] = all_subsets[:, interaction].all(axis=1)
        return regression_subsets, num_players

    def _get_sii_subset_representation(
        self, all_subsets: np.ndarray[bool]
    ) -> tuple[np.ndarray[bool], int]:
        """Transforms a subset matrix into the SII representation.

        The SII representation is a matrix of shape (n_subsets, num_players) where each interaction
        up to the maximum order is an individual player.

        Args:
            all_subsets: subset matrix in shape (n_subsets, n).

        Returns:
            SII representation of the subset matrix in shape (n_subsets, num_players) and the number
            of players.
        """
        n_subsets = all_subsets.shape[0]
        num_players = sum(int(binom(self.n, order)) for order in range(1, self.max_order + 1))
        regression_subsets = np.zeros(shape=(n_subsets, num_players), dtype=float)
        for interaction_index, interaction in enumerate(
            powerset(self.N, min_size=1, max_size=self.max_order)
        ):
            intersection_size = np.sum(all_subsets[:, interaction], axis=1)
            r_prime = np.full(shape=(n_subsets,), fill_value=len(interaction))
            weights = self._get_bernoulli_weights(intersection_size, r_prime)
            regression_subsets[:, interaction_index] = weights
        return regression_subsets, num_players

    def _get_bernoulli_weight(self, intersection_size: int, r_prime: int) -> float:
        """Calculates the Bernoulli weights for the SII.

        Args:
            intersection_size: The orders of the interactions.
            r_prime: The orders of the interactions.

        Returns:
            The Bernoulli weights.
        """
        weight = 0
        for size in range(1, intersection_size + 1):
            weight += binom(intersection_size, size) * self._bernoulli_numbers[r_prime - size]
        return weight

    def _get_bernoulli_weights(
        self, intersection_size: np.ndarray[int], r_prime: np.ndarray[int]
    ) -> np.ndarray[float]:
        """Calculates the Bernoulli weights for the SII.

        Args:
            intersection_size: The orders of the interactions.
            r_prime: The orders of the interactions.

        Returns:
            The Bernoulli weights.
        """
        weights = np.zeros(shape=(intersection_size.shape[0],), dtype=float)
        for index, (intersection_size_i, r_prime_i) in enumerate(zip(intersection_size, r_prime)):
            weights[index] = self._get_bernoulli_weight(intersection_size_i, r_prime_i)
        return weights
