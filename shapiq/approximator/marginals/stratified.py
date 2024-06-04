"""This module contains the Stratified Sampling approximation method for the Shapley value
by Maleki et al. (2013). It estimates the Shapley values by sampling random marginal contributions
grouped by size."""

from typing import Callable, Optional

import numpy as np

from shapiq.approximator._base import Approximator
from shapiq.interaction_values import InteractionValues


class StratifiedSamplingSV(Approximator):
    """The Stratified Sampling algorithm estimates the Shapley values (SV) by sampling random
    marginal contributions for each player and each coalition size. The marginal contributions are
    grouped into strata by size. The strata are aggregated for each player after sampling to obtain
    the final estimate. For more information, see `Maleki et al. (2009) <http://arxiv.org/abs/1306.4265>`_.

    Args:
        n: The number of players.
        random_state: The random state to use for the permutation sampling. Defaults to ``None``.

    Attributes:
        n: The number of players.
        _grand_coalition_array: The array of players (starting from ``0`` to ``n``).
        iteration_cost: The cost of a single iteration of the approximator.
    """

    def __init__(
        self,
        n: int,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(n, max_order=1, index="SV", top_order=False, random_state=random_state)
        self.iteration_cost: int = 2

    def approximate(
        self, budget: int, game: Callable[[np.ndarray], np.ndarray]
    ) -> InteractionValues:
        """Approximates the Shapley values using ApproShapley.

        Args:
            budget: The number of game evaluations for approximation
            game: The game function as a callable that takes a set of players and returns the value.
            batch_size: The size of the batch. If ``None``, the batch size is set to ``1``. Defaults to ``5``.

        Returns:
            The estimated interaction values.
        """

        used_budget = 0

        # get value of empty coalition and grand coalition
        empty_value = game(np.zeros(self.n, dtype=bool))[0]
        full_value = game(np.ones(self.n, dtype=bool))[0]
        used_budget += 2

        strata = np.zeros((self.n, self.n), dtype=float)
        counts = np.zeros((self.n, self.n), dtype=int)

        # main sampling loop
        while used_budget < budget:
            # iterate over coalition size to which a marginal contribution can be drawn
            for size in range(0, self.n):
                # iterate over players for whom a marginal contribution is to be drawn
                for player in range(self.n):
                    # check if enough budget is available to sample a marginal contribution
                    if ((size == 0 or size == self.n - 1) and used_budget < budget) or (
                        size in range(1, self.n - 1) and used_budget + 2 <= budget
                    ):
                        marginal_con = 0
                        # if coalition size is 0 or n-1, empty or grand coalition value can be reuse
                        if size == 0:
                            coalition = np.zeros(self.n, dtype=bool)
                            coalition[player] = True
                            marginal_con = game(coalition)[0] - empty_value
                            used_budget += 1
                        elif size == self.n - 1:
                            coalition = np.ones(self.n, dtype=bool)
                            coalition[player] = False
                            marginal_con = full_value - game(coalition)[0]
                            used_budget += 1
                        # otherwise both coalitions that make up the marginal contribution have to eb evaluated
                        else:
                            available_players = list(self._grand_coalition_set)
                            available_players.remove(player)
                            # draw a subset of the player set without player of size stratum uniformly at random
                            coalition_list = list(
                                self._rng.choice(available_players, size, replace=False)
                            )
                            coalition = np.zeros(self.n, dtype=bool)
                            coalition[coalition_list] = True
                            marginal_con = -game(coalition)[0]
                            coalition[player] = True
                            marginal_con += game(coalition)[0]
                            used_budget += 2
                        # update the affected strata estimate
                        strata[player][size] += marginal_con
                        counts[player][size] += 1

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
