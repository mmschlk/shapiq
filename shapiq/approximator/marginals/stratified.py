"""This module contains the Stratified Sampling approximation method for the Shapley value (SV) by Maleki et al. (2013).
It estimates the Shapley values by sampling random marginal contributions grouped by size."""

from typing import Callable, Optional

import numpy as np

from shapiq.approximator._base import Approximator
from shapiq.interaction_values import InteractionValues


class StratifiedSamplingSV(Approximator):
    """The Stratified Sampling algorithm estimates the Shapley values (SV) by sampling random marginal contributions
    for each player and each coalition size. The marginal contributions are grouped into strata by size.
    The strata are aggregated for each player after sampling to obtain the final estimate.
    For more information, see [Maleki et al. (2009)](http://arxiv.org/abs/1306.4265).

    Args:
        n: The number of players.
        random_state: The random state to use for the permutation sampling. Defaults to `None`.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        _grand_coalition_array: The array of players (starting from 0 to n).
        iteration_cost: The cost of a single iteration of the approximator.
    """

    def __init__(
        self,
        n: int,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(n, max_order=1, index="SV", top_order=False, random_state=random_state)
        self.iteration_cost: int = 2 * n * n

    def approximate(
        self, budget: int, game: Callable[[np.ndarray], np.ndarray], batch_size: Optional[int] = 5
    ) -> InteractionValues:
        """Approximates the Shapley values using ApproShapley.

        Args:
            budget: The number of game evaluations for approximation
            game: The game function as a callable that takes a set of players and returns the value.
            batch_size: The size of the batch. If None, the batch size is set to 1. Defaults to 5.

        Returns:
            The estimated interaction values.
        """

        used_budget = 0
        batch_size = 1 if batch_size is None else batch_size

        # get empty value
        empty_value = float(game(np.zeros(self.n, dtype=bool)))
        used_budget += 1

        # compute the number of iterations and size of the last batch (can be smaller than original)
        n_iterations, last_batch_size = self._calc_iteration_count(
            budget - used_budget, batch_size, self.iteration_cost
        )

        strata = np.zeros((self.n, self.n), dtype=float)
        counts = np.zeros((self.n, self.n), dtype=int)

        # main sampling loop going through all strata of all players with each segment
        for iteration in range(1, n_iterations + 1):
            batch_size = batch_size if iteration != n_iterations else last_batch_size
            n_segments = batch_size
            n_coalitions = n_segments * self.iteration_cost
            coalitions = np.zeros(shape=(n_coalitions, self.n), dtype=bool)
            coalition_index = 0
            # iterate through each segment
            for segment in range(n_segments):
                # iterate through each player
                for player in range(self.n):
                    available_players = list(self._grand_coalition_set)
                    available_players.remove(player)
                    # iterate through each stratum
                    for stratum in range(self.n):
                        # draw a subset of the player set without player of size stratum uniformly at random
                        coalition = list(
                            np.random.choice(available_players, stratum, replace=False)
                        )
                        # add the coalition and coalition with the player, both form a marginal contribution
                        coalitions[coalition_index, tuple(coalition)] = True
                        coalition.append(player)
                        coalitions[coalition_index + 1, tuple(coalition)] = True
                        coalition_index += 2

            # evaluate the collected coalitions
            game_values: np.ndarray[float] = game(coalitions)
            used_budget += len(coalitions)

            # update the strata estimates
            coalition_index = 0
            # iterate through each segment
            for segment in range(n_segments):
                for player in range(self.n):
                    for stratum in range(self.n):
                        # calculate the marginal contribution and update the stratum estimate
                        marginal_con = (
                            game_values[coalition_index + 1] - game_values[coalition_index]
                        )
                        strata[player][stratum] += marginal_con
                        counts[player][stratum] += 1
                        coalition_index += 2

        # aggregate the stratum estimates: divide each stratum sum by its sample number, sum up the means, divide by the number of valid stratum estimates
        strata = np.divide(strata, counts, out=strata, where=counts != 0)
        result = np.sum(strata, axis=1)
        non_zeros = np.count_nonzero(counts, axis=1)
        result = np.divide(result, non_zeros, out=result, where=non_zeros != 0)

        # create vector of interaction values with correct length and order
        result_to_finalize = self._init_result(dtype=float)
        result_to_finalize[self._interaction_lookup[()]] = empty_value
        for player in range(self.n):
            idx = self._interaction_lookup[(player,)]
            result_to_finalize[idx] = result[player]

        return self._finalize_result(
            result_to_finalize, baseline_value=empty_value, budget=used_budget, estimated=True
        )
