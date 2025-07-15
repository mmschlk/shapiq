"""The Owen Sampling approximator for the Shapley value."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.games.base import Game


class OwenSamplingSV(Approximator):
    """Owen Sampling approximator for the Shapley values.

    The Owen Sampling algorithm estimates the Shapley values (SV) by sampling random marginal
    contributions for each player and each coalition size. The marginal contributions are used to
    update an integral representation of the SV. For more information, see [Okh20]_.
    The number of anchor points M at which the integral is to be palpated share the available budget
    for each player equally. A higher `n_anchor_points` increases the resolution of the integral
    reducing bias while reducing the accuracy of the estimation at each point.

    Attributes:
        n: The number of players.
        iteration_cost: The cost of a single iteration of the approximator.

    References:
        .. [Okh20] Ramin Okhrati, Aldo Lipani (2020). A Multilinear Sampling Algorithm to Estimate Shapley Values. arXiv preprint arXiv:2010.12082. https://doi.org/10.48550/arXiv.2010.12082

    """

    valid_indices: tuple[Literal["SV"]] = ("SV",)
    """The valid indices for this approximator."""

    def __init__(
        self,
        n: int,
        n_anchor_points: int = 10,
        random_state: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the Owen Sampling SV approximator.

        Args:
            n: The number of players.

            n_anchor_points: The number of anchor points at which the integral is to be palpated.
                Defaults to ``10``.

            random_state: The random state to use for the permutation sampling. Defaults to
                ``None``.

            **kwargs: Additional arguments not used.

        """
        super().__init__(n, max_order=1, index="SV", top_order=False, random_state=random_state)
        self.iteration_cost: int = 2
        self.n_anchor_points = n_anchor_points

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximates the Shapley values using Owen Sampling.

        Args:
            budget: The number of game evaluations for approximation

            game: The game function as a callable that takes a set of players and returns the value.

            *args: Additional positional arguments not used in this method.

            **kwargs: Additional keyword arguments not used in this method.

        Returns:
            The estimated interaction values.

        """
        used_budget = 0

        empty_value = game(np.zeros(self.n, dtype=bool))[0]
        used_budget += 1

        anchors = self.get_anchor_points(self.n_anchor_points)
        estimates = np.zeros((self.n, self.n_anchor_points), dtype=float)
        counts = np.zeros((self.n, self.n_anchor_points), dtype=int)

        # main sampling loop
        while used_budget + 2 <= budget:
            # iterate over anchor points from which a marginal contribution can be drawn
            for m in range(len(anchors)):
                q = anchors[m]
                # iterate over players for whom a marginal contribution is to be drawn
                for player in range(self.n):
                    if used_budget + 2 <= budget:
                        # draw a subset of players without player: all are inserted independently
                        # with probability q
                        coalition = self._rng.choice(
                            [True, False],
                            self.n - 1,
                            replace=True,
                            p=[q, 1 - q],
                        )
                        # add information that player is absent
                        coalition = np.insert(coalition, player, False)
                        marginal_con = -game(coalition)[0]
                        # add information that player is present to complete marginal contribution
                        coalition[player] = True
                        marginal_con += game(coalition)[0]
                        used_budget += 2
                        # update the affected strata estimate
                        estimates[player][m] += marginal_con
                        counts[player][m] += 1

        # aggregate the anchor estimates: divide each anchor sum by its sample number, sum up the
        # means, divide by the number of valid anchor estimates
        estimates = np.divide(estimates, counts, out=estimates, where=counts != 0)
        result = np.sum(estimates, axis=1)
        non_zeros = np.count_nonzero(counts, axis=1)
        result = np.divide(result, non_zeros, out=result, where=non_zeros != 0)

        # create vector of interaction values with correct length and order
        result_to_finalize = self._init_result(dtype=float)
        result_to_finalize[self._interaction_lookup[()]] = empty_value
        for player in range(self.n):
            idx = self._interaction_lookup[(player,)]
            result_to_finalize[idx] = result[player]

        interaction = InteractionValues(
            n_players=self.n,
            values=result_to_finalize,
            index=self.approximation_index,
            interaction_lookup=self._interaction_lookup,
            baseline_value=empty_value,
            min_order=self.min_order,
            max_order=self.max_order,
            estimated=True,
            estimation_budget=used_budget,
        )

        return finalize_computed_interactions(
            interaction,
            target_index=self.index,
        )

    @staticmethod
    def get_anchor_points(n_anchor_points: int) -> np.ndarray:
        """Returns the anchor points for the Owen Sampling approximation.

        Args:
            n_anchor_points: The number of anchor points.

        Returns:
            An array of anchor points.

        Raises:
            ValueError: If the number of anchor points is less than or equal to 0.

        """
        if n_anchor_points <= 0:
            msg = "The number of anchor points needs to be greater than 0."
            raise ValueError(msg)
        if n_anchor_points == 1:
            return np.array([0.5])
        return np.linspace(0.0, 1.0, num=n_anchor_points)
