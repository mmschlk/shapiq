"""This module contains the Base Regression approximator to compute SII and k-SII of arbitrary max_order."""

from typing import Callable, Optional

import numpy as np
from scipy.special import binom, factorial

from shapiq.approximator._base import Approximator
from shapiq.approximator.k_sii import KShapleyMixin
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.interaction_values import InteractionValues

AVAILABLE_INDICES_REGRESSION = ["k-SII", "SII", "kADD-SHAP", "FSII", "kADD-SHAP"]


class MonteCarlo(Approximator, KShapleyMixin):
    """This class is the base class for all MonteCarlo approximators, e.g. SHAP-IQ and SVARM-IQ.

    MonteCarlo approximators are based on a representation of the interaction index as a weighted sum over discrete
    derivatives. The sum is re-written and approximated using Monte Carlo sampling.
    The sum may be stratified by coalition size or by the intersection size of the coalition and the interaction.
    The standard form for approximation is based on Theorem 1 in
    https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The interaction index to be estimated. Available indices are 'SII', 'kSII', 'STII',
            and 'FSII'.
        stratify_coalition_size: If True, then each coalition size is estimated separately
        stratify_intersection_size: If True, then each coalition is stratified by number of interseection elements
        random_state: The random state to use for the approximation. Defaults to None.
    """

    def __init__(
        self,
        n: int,
        max_order: int,
        index: str,
        stratify_coalition_size: bool = True,
        stratify_intersection_size: bool = True,
        random_state: Optional[int] = None,
    ):
        if index not in AVAILABLE_INDICES_REGRESSION:
            raise ValueError(
                f"Index {index} not available for Regression Approximator. Choose from "
                f"{AVAILABLE_INDICES_REGRESSION}."
            )
        super().__init__(
            n,
            min_order=0,
            max_order=max_order,
            index=index,
            top_order=False,
            random_state=random_state,
        )
        self._big_M: int = 10e7

    def _init_sampling_weights(self) -> np.ndarray:
        """Initializes the weights for sampling subsets.

        The sampling weights are of size n + 1 and indexed by the size of the subset. The edges
        All weights are set to _big_M, if size < order or size > n - order to ensure efficiency.

        Returns:
            The weights for sampling subsets of size s in shape (n + 1,).
        """
        weight_vector = np.zeros(shape=self.n + 1)
        for coalition_size in range(0, self.n + 1):
            if (coalition_size == 0) or (coalition_size == self.n):
                # prioritize these subsets
                weight_vector[coalition_size] = self._big_M**2
            elif (coalition_size < self.max_order) or (coalition_size > self.n - self.max_order):
                # prioritize these subsets
                weight_vector[coalition_size] = self._big_M
            else:
                # KernelSHAP sampling weights
                weight_vector[coalition_size] = 1 / (coalition_size * (self.n - coalition_size))
        sampling_weight = weight_vector / np.sum(weight_vector)
        return sampling_weight

    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
        batch_size: Optional[int] = None,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray = None,
    ) -> InteractionValues:
        # validate input parameters
        batch_size = budget if batch_size is None else batch_size

        if sampling_weights is None:
            # Initialize default sampling weights
            sampling_weights = self._init_sampling_weights()

        sampler = CoalitionSampler(
            n_players=self.n,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=self._random_state,
        )

        sampler.sample(budget)

        coalitions_matrix = sampler.coalitions_matrix
        sampling_adjustment_weights = sampler.sampling_adjustment_weights
        n_coalitions = sampler.n_coalitions

        # calculate the number of iterations and the last batch size
        n_iterations, last_batch_size = self._calc_iteration_count(
            n_coalitions, batch_size, iteration_cost=self.iteration_cost
        )

        game_values: np.ndarray[float] = np.zeros(shape=(n_coalitions,), dtype=float)

        # main regression loop computing the FSII values
        for iteration in range(1, n_iterations + 1):
            current_batch_size = batch_size if iteration != n_iterations else last_batch_size
            batch_index = (iteration - 1) * batch_size

            batch_coalitions_size = np.sum(
                coalitions_matrix[0 : batch_index + current_batch_size, :], 1
            )
            batch_coalitions_matrix = coalitions_matrix[0 : batch_index + current_batch_size, :]
            batch_sampling_adjustment_weights = sampling_adjustment_weights[
                0 : batch_index + current_batch_size
            ]

            # query the game for the current batch of coalitions
            game_values[batch_index : batch_index + current_batch_size] = game(
                batch_coalitions_matrix[batch_index : batch_index + current_batch_size, :]
            )

            batch_game_values = game_values[0 : batch_index + current_batch_size]

            shapley_interactions_values = self.montecarlo_routine(
                {},
                batch_game_values,
                batch_coalitions_matrix,
                batch_coalitions_size,
                batch_sampling_adjustment_weights,
                self.index,
            )

            if np.shape(coalitions_matrix)[0] >= 2**self.n:
                estimated_indicator = False
            else:
                estimated_indicator = True

        return self._finalize_result(
            result=shapley_interactions_values, estimated=estimated_indicator, budget=budget
        )

    def montecarlo_routine(
        self,
        kernel_weights_dict: dict,
        batch_game_values: np.ndarray,
        batch_coalitions_matrix: np.ndarray,
        batch_coalitions_size: np.ndarray,
        batch_sampling_adjustment_weights: np.ndarray,
        index_approximation: str,
    ):
        return {}

    def _sii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Returns the SII discrete derivative weight given the coalition size and interaction size.

        Args:
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        return 1 / (
            (self.n - interaction_size + 1) * binom(self.n - interaction_size, coalition_size)
        )

    def _stii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Returns the STII discrete derivative weight given the coalition size and interaction size.

        Representation according to https://proceedings.mlr.press/v119/sundararajan20a.html

        Args:
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        if interaction_size == self.max_order:
            return self.max_order / (self.n * binom(self.n - 1, coalition_size))
        else:
            return 1.0 * (coalition_size == 0)

    def _fsii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Returns the FSII discrete derivative weight given the coalition size and interaction size.

        Representation according to Theorem 19 in https://www.jmlr.org/papers/volume24/22-0202/22-0202.pdf
        Args:
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        if interaction_size == self.max_order:
            return (
                factorial(2 * self.max_order - 1)
                / factorial(self.max_order - 1) ** 2
                * factorial(self.n - coalition_size - 1)
                * factorial(coalition_size + self.max_order - 1)
                / factorial(self.n + self.max_order - 1)
            )
        else:
            raise ValueError("Lower order interactions are not supported.")

    def _weight_kernel(self, coalition_size: int, interaction_size: int) -> float:
        """Returns the weight for each interaction type given coalition and interaction size.

        Args:
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        if self.index == "SII" or self.index == "k-SII" or self.index == "SV":  # SII default
            return self._sii_weight(coalition_size, interaction_size)
        elif self.index == "STII":
            return self._stii_weight(coalition_size, interaction_size)
        elif self.index == "FSII":
            return self._fsii_weight(coalition_size, interaction_size)
        else:
            raise ValueError(f"Unknown index {self.index}.")

    def _init_discrete_derivative_weights(self) -> dict[int, np.ndarray[float]]:
        """Initializes the weights for the interaction index re-written from discrete derivatives to standard form.
         Standard form according to Theorem 1 in https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html

        Returns:
            The standard form weights.
        """
        # init data structure
        weights = {}
        for order in self._order_iterator:
            weights[order] = np.zeros((self.n + 1, order + 1))
        # fill with values specific to each index
        for t in range(0, self.n + 1):
            for order in self._order_iterator:
                for k in range(max(0, order + t - self.n), min(order, t) + 1):
                    weights[order][t, k] = (-1) ** (order - k) * self._weight_kernel(t - k, order)
        return weights
