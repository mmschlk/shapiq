"""This module contains the Stratified Sampling approximation method for the Shapley values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.games.base import Game


class StratifiedSamplingSV(Approximator):
    """The Stratifield Sampling algorithm for estimating the Shapley values.

    The Stratified Sampling algorithm estimates the Shapley values (SV) by sampling random
    marginal contributions for each player and each coalition size. The marginal contributions are
    grouped into strata by size. The strata are aggregated for each player after sampling to obtain
    the final estimate. For more information, see Maleki et al. (2013) [Mal13]_.

    See Also:
        - :class:`~shapiq.approximator.montecarlo.svarmiq.SVARM`: The SVARM approximator
        - :class:`~shapiq.approximator.montecarlo.svarmiq.SVARMIQ`: The SVARMIQ approximator

    References:
        .. [Mal13] Maleki, S., Tran-Thanh, L., Hines, G., Rahwan, T., and Rogers, A, (2013). Bounding the Estimation Error of Sampling-based Shapley Value Approximation With/Without Stratifying

    """

    valid_indices: tuple[Literal["SV"]] = ("SV",)

    def __init__(
        self,
        n: int,
        random_state: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the Stratified Sampling SV approximator.

        Args:
            n: The number of players.

            random_state: The random state to use for the permutation sampling. Defaults to
            ``None``.

            **kwargs: Additional arguments not used.

        """
        super().__init__(n, max_order=1, index="SV", top_order=False, random_state=random_state)
        self.iteration_cost: int = 2

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximates the Shapley values using ApproShapley.

        Args:
            budget: The number of game evaluations for approximation

            game: The game function as a callable that takes a set of players and returns the value.

            *args: Additional positional arguments (not used).

            **kwargs: Additional keyword arguments (not used).

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
            for size in range(self.n):
                # iterate over players for whom a marginal contribution is to be drawn
                for player in range(self.n):
                    # check if enough budget is available to sample a marginal contribution
                    if ((size == 0 or size == self.n - 1) and used_budget < budget) or (
                        size in range(1, self.n - 1) and used_budget + 2 <= budget
                    ):
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
                        # otherwise both coalitions that make up the marginal contribution have
                        # to eb evaluated
                        else:
                            available_players = list(self._grand_coalition_set)
                            available_players.remove(player)
                            # draw a subset of the player set without player of size stratum
                            # uniformly at random
                            coalition_list = list(
                                self._rng.choice(available_players, size, replace=False),
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

        # aggregate the stratum estimates: divide each stratum sum by its sample number, sum up
        # the means, divide by the number of valid stratum estimates
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

        interactions = InteractionValues(
            n_players=self.n,
            values=result_to_finalize,
            index=self.approximation_index,
            interaction_lookup=self._interaction_lookup,
            baseline_value=float(empty_value),
            min_order=self.min_order,
            max_order=self.max_order,
            estimated=True,
            estimation_budget=used_budget,
        )

        return finalize_computed_interactions(interactions, target_index=self.index)
