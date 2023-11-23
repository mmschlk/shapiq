"""This module contains the regression algorithms to estimate FSI scores."""
import copy
from typing import Optional, Callable

import numpy as np
from scipy.special import binom

from approximator._base import Approximator, ShapleySamplingMixin, InteractionValues
from utils import split_subsets_budget, powerset


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

    Properties:
        iteration_cost: The cost of a single iteration of the regression FSI.

    Example:
        >>> from games import DummyGame
        >>> from approximator import RegressionFSI
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = RegressionFSI(n=5, max_order=2)
        >>> approximator.approximate(budget=100, game=game)
        InteractionValues(
            index=FSI, order=2, estimated=True, estimation_budget=102,
            values={
                1: [0.2 0.2 0.2 0.2 0.2]
                2: [[ 0.  0.  0.  0.  0.]
                    [ 0.  0.  1.  0.  0.]
                    [ 0.  0.  0.  0.  0.]
                    [ 0.  0.  0.  0.  0.]
                    [ 0.  0.  0.  0.  0.]]
            }
        )
    """

    def __init__(
        self,
        n: int,
        max_order: int,
        random_state: Optional[int] = None,
    ) -> None:
        # init approximator and shapley weights mixin
        super().__init__(n, max_order, index="FSI", top_order=False, random_state=random_state)
        ShapleySamplingMixin.__init__(self)

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
        n_iterations, last_batch_size = self._get_n_iterations(
            budget + 2, batch_size, iteration_cost=1
        )

        # create storage array for given budget
        all_subsets: np.ndarray[bool] = np.zeros(shape=(budget, self.n), dtype=bool)

        # split the subset sizes into explicit and sampling parts
        sampling_weights: np.ndarray[float] = self._init_ksh_sampling_weights()
        explicit_sizes, sampling_sizes, remaining_budget = split_subsets_budget(
            order=1, n=self.n, budget=budget, sampling_weights=sampling_weights
        )

        # enumerate all explicit subsets
        explicit_subsets: np.ndarray[bool] = self._get_explicit_subsets(self.n, explicit_sizes)
        all_subsets[: len(explicit_subsets)] = explicit_subsets
        # zero out the sampling weights for the explicit sizes
        sampling_weights[explicit_sizes] = 0.0

        # sample the remaining subsets with the remaining budget
        if len(sampling_sizes) > 0 and remaining_budget > 0:
            sampling_subsets: np.ndarray[bool] = self._sample_subsets(
                budget=remaining_budget,
                sampling_weights=sampling_weights,
                replacement=replacement,
                pairing=pairing,
            )
            all_subsets[len(explicit_subsets) :] = sampling_subsets

        # add empty and full set to all_subsets in the beginning
        all_subsets = np.concatenate(
            (
                np.zeros(shape=(1, self.n), dtype=bool),  # empty set
                np.ones(shape=(1, self.n), dtype=bool),  # full set
                all_subsets,  # explicit and sampled subsets
            )
        )
        n_subsets = all_subsets.shape[0]

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
            batch_subsets = all_subsets[0 : batch_index + batch_size]
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

        return self._finalize_fsi_result(fsi_values, budget=used_budget)

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

    def _finalize_fsi_result(
        self, fsi_values: np.ndarray[float], budget: Optional[int] = None
    ) -> InteractionValues:
        """Transforms the FSI values into the output interaction values.

        Args:
            fsi_values: FSI values in shape (num_players,).
            budget: The budget of the approximation. Defaults to `None`.

        Returns:
            InteractionValues: The estimated interaction values.
        """
        result = self._init_result()
        fsi_index = 0
        for interaction in powerset(self.N, min_size=1, max_size=self.max_order):
            result[len(interaction)][interaction] = fsi_values[fsi_index]
            fsi_index += 1
        return self._finalize_result(result, budget=budget)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
