"""This module contains the permutation sampling approximation method for the Shapley value (SV)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.games.base import Game


class PermutationSamplingSV(Approximator):
    """The Permutation Sampling algorithm for estimating the Shapley values.

    Permutation Sampling [1]_ (also known as ApproShapley) estimates the Shapley values by drawing
    random permutations of the player set and extracting all marginal contributions from each
    permutation. For details, see Castro et al. (2009)[1]_.

    Args:
        n: The number of players.
        random_state: The random state to use for the permutation sampling. Defaults to ``None``.

    See Also:
        - :class:`~shapiq.approximator.permutation.sii.PermutationSamplingSII`: The Permutation
            Sampling approximator for the SII index
        - :class:`~shapiq.approximator.permutation.stii.PermutationSamplingSTII`: The Permutation
            Sampling approximator for the STII index

    References:
        .. [1] Castro, J., Gómez, D., and Tejada, J. (2009) Polynomial calculation of the Shapley value based on sampling. In Computers & Operations Research 36(5), 1726-1730. doi: https://doi.org/10.1016/j.cor.2008.04.004

    """

    valid_indices: tuple[Literal["SV"]] = ("SV",)

    def __init__(
        self,
        n: int,
        random_state: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the Permutation Sampling approximator for Shapley values.

        Args:
            n: The number of players.

            random_state: The random state to use for the permutation sampling. Defaults to
                ``None``.

            **kwargs: Additional keyword arguments (not used for compatibility)
        """
        super().__init__(n=n, max_order=1, index="SV", top_order=False, random_state=random_state)
        self.iteration_cost: int = n - 1

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        batch_size: int | None = 5,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximates the Shapley values using ApproShapley.

        Args:
            budget: The number of game evaluations for approximation

            game: The game function as a callable that takes a set of players and returns the value.

            batch_size: The size of the batch. If ``None``, the batch size is set to ``1``.
                Defaults to ``5``.

            *args: Additional positional arguments (not used, only for compatibility).

            **kwargs: Additional keyword arguments (not used, only for compatibility).

        Returns:
            The estimated interaction values.

        """
        result: np.ndarray[float] = self._init_result()
        counts: np.ndarray[int] = self._init_result(dtype=int)

        batch_size = 1 if batch_size is None else batch_size

        # store the values of the empty and full coalition
        # this saves 2 evaluations per permutation
        empty_val = float(game(np.zeros(self.n, dtype=bool))[0])
        full_val = float(game(np.ones(self.n, dtype=bool))[0])
        used_budget = 2

        # catch special case of single player game, otherwise iteration through permutations fails
        if self.n == 1:
            interaction_index = self._interaction_lookup[self._grand_coalition_tuple]
            result[interaction_index] = full_val - empty_val
            counts[interaction_index] = 1

            interactions = InteractionValues(
                values=result,
                interaction_lookup=self._interaction_lookup,
                baseline_value=empty_val,
                min_order=self.min_order,
                max_order=self.max_order,
                n_players=self.n,
                index=self.approximation_index,
                estimated=True,
                estimation_budget=used_budget,
            )

            return finalize_computed_interactions(interactions, target_index=self.index)

        # compute the number of iterations and size of the last batch (can be smaller than original)
        n_iterations, last_batch_size = self._calc_iteration_count(
            budget - 2,
            batch_size,
            self.iteration_cost,
        )

        # main permutation sampling loop
        for iteration in range(1, n_iterations + 1):
            batch_size = batch_size if iteration != n_iterations else last_batch_size

            # create batch_size many permutations: two-dimensional matrix of shape (batch_size, n)
            # each row is a permutation
            permutations = np.tile(np.arange(self.n), (batch_size, 1))
            self._rng.permuted(permutations, axis=1, out=permutations)
            n_permutations = batch_size
            n_coalitions = n_permutations * self.iteration_cost

            # generate all coalitions to evaluate from permutations
            coalitions = np.zeros(shape=(n_coalitions, self.n), dtype=bool)
            coalition_index = 0
            for permutation_id in range(n_permutations):
                permutation = permutations[permutation_id]
                coalition = set()
                for i in range(self.n - 1):
                    coalition.add(permutation[i])
                    coalitions[coalition_index, tuple(coalition)] = True
                    coalition_index += 1

            # evaluate the collected coalitions
            game_values = game(coalitions)
            used_budget += len(coalitions)

            # update the estimates
            coalition_index = 0
            for permutation_id in range(n_permutations):
                # update the first player in the permutation
                permutation = permutations[permutation_id]
                marginal_con = game_values[coalition_index] - empty_val
                permutation_idx = self._interaction_lookup[(permutation[0],)]
                result[permutation_idx] += marginal_con
                counts[permutation_idx] += 1
                # update the players in the middle of the permutation
                for i in range(1, self.n - 1):
                    marginal_con = game_values[coalition_index + 1] - game_values[coalition_index]
                    permutation_idx = self._interaction_lookup[(permutation[i],)]
                    result[permutation_idx] += marginal_con
                    counts[permutation_idx] += 1
                    coalition_index += 1
                # update the last player in the permutation
                marginal_con = full_val - game_values[coalition_index]
                permutation_idx = self._interaction_lookup[(permutation[self.n - 1],)]
                result[permutation_idx] += marginal_con
                counts[permutation_idx] += 1
                coalition_index += 1

        result = np.divide(result, counts, out=result, where=counts != 0)

        interactions = InteractionValues(
            values=result,
            interaction_lookup=self._interaction_lookup,
            baseline_value=empty_val,
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            index=self.approximation_index,
            estimated=True,
            estimation_budget=used_budget,
        )

        return finalize_computed_interactions(interactions, target_index=self.index)
