"""This module contains the KernelSHAPIQ approximator to compute SII and k-SII of arbitrary order."""

import copy
from typing import Callable, Optional

import numpy as np
from scipy.special import bernoulli, binom

from shapiq.approximator._base import Approximator
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset

AVAILABLE_INDICES_KERNELSHAPIQ = ["SII"]


class KernelSHAPIQ(Approximator):
    def __init__(self, n: int, order: int, index: str = "SII", random_state: Optional[int] = None):
        if index not in AVAILABLE_INDICES_KERNELSHAPIQ:
            raise ValueError(
                f"Index {index} not available for KernelSHAP-IQ. Choose from "
                f"{AVAILABLE_INDICES_KERNELSHAPIQ}."
            )
        super().__init__(
            n, max_order=order, index=index, top_order=False, random_state=random_state
        )
        self._big_M = 10e7
        self._bernoulli_numbers = bernoulli(self.n)  # used for SII

    def _init_kernel_weights(self, interaction_size: int) -> np.ndarray:
        """Initializes the kernel weights for the regression in KernelSHAP-IQ.

        The kernel weights are of size n + 1 and indexed by the size of the coalition.
        The kernel weights depend on the size of the interactions.
        The kernel weights are set to _big_M for the edges and adjusted by the number of coalitions

        Returns:
            The weights for sampling subsets of size s in shape (n + 1,).
        """
        # vector that determines the kernel weights for KernelSHAPIQ
        weight_vector = np.zeros(shape=self.n + 1)
        for coalition_size in range(0, self.n + 1):
            if (coalition_size < interaction_size) or (coalition_size > self.n - interaction_size):
                weight_vector[coalition_size] = self._big_M  # * binom(self.n, coalition_size)
            else:
                weight_vector[coalition_size] = 1 / (
                    (self.n - interaction_size + 1)
                    * binom(self.n - 2 * interaction_size, coalition_size - interaction_size)
                )
        kernel_weight = weight_vector
        return kernel_weight

    def _init_sampling_weights(self) -> np.ndarray:
        """Initializes the weights for sampling subsets.

        The sampling weights are of size n + 1 and indexed by the size of the subset. The edges
        All weights are set to _big_M, if size < order or size > n - order to ensure efficiency.

        Returns:
            The weights for sampling subsets of size s in shape (n + 1,).
        """
        weight_vector = np.zeros(shape=self.n + 1)
        for subset_size in range(0, self.n + 1):
            if (subset_size < self.max_order) or (subset_size > self.n - self.max_order):
                # prioritize these subsets
                weight_vector[subset_size] = self._big_M
            else:
                # KernelSHAP sampling weights
                weight_vector[subset_size] = 1 / (subset_size * (self.n - subset_size))
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
        if sampling_weights is None:
            # Initialize default sampling weights
            sampling_weights = self._init_sampling_weights()

        kernel_weights_dict = {}
        for interaction_size in range(1, self.max_order + 1):
            kernel_weights_dict[interaction_size] = self._init_kernel_weights(interaction_size)
        sampler = CoalitionSampler(
            n_players=self.n,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=self._random_state,
        )

        sampler.sample(budget)

        coalitions_matrix = sampler.coalitions_matrix
        coalitions_size = np.sum(coalitions_matrix, 1)
        sampling_adjustment_weights = sampler.sampling_adjustment_weights
        n_coalitions = sampler.n_coalitions

        game_values = game(coalitions_matrix)
        emptycoalition_value = game_values[0]
        game_values -= emptycoalition_value

        sii_values = np.array([])
        residual_game_values = {}
        residual_game_values[1] = copy.copy(game_values)

        for interaction_size in range(1, self.max_order + 1):
            regression_coefficient_weight = self._get_regression_coefficient_weights(
                interaction_size=interaction_size, index=self.index
            )
            regression_matrix = np.zeros((n_coalitions, int(binom(self.n, interaction_size))))
            for coalition_pos, coalition in enumerate(coalitions_matrix):
                for interaction_pos, interaction in enumerate(
                    powerset(self.N, min_size=interaction_size, max_size=interaction_size)
                ):
                    intersection_size = np.sum(coalition[list(interaction)])
                    regression_matrix[
                        coalition_pos, interaction_pos
                    ] = regression_coefficient_weight[intersection_size]

            # Regression weights adjusted by sampling weights
            regression_weights = (
                kernel_weights_dict[interaction_size][coalitions_size] * sampling_adjustment_weights
            )

            if interaction_size <= 2:
                regression_weights_sqrt_matrix = np.diag(np.sqrt(regression_weights))
                regression_lhs = np.dot(regression_weights_sqrt_matrix, regression_matrix)
                regression_rhs = np.dot(
                    regression_weights_sqrt_matrix, residual_game_values[interaction_size]
                )
                wlsq_solution = np.linalg.lstsq(regression_lhs, regression_rhs, rcond=None)[
                    0
                ]  # \phi_i
                approximations = np.dot(regression_matrix, wlsq_solution)
                sii_values = np.hstack((sii_values, wlsq_solution))


            residual_game_values[interaction_size + 1] = (
                residual_game_values[interaction_size] - approximations
            )

        sii = InteractionValues(
            baseline_value=emptycoalition_value,
            values=sii_values,
            interaction_lookup=self.interaction_lookup,
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            index=self.index,
        )

        return sii

    def _get_regression_coefficient_weights(self, interaction_size: int, index: str) -> float:
        """Returns the weights for the regression coefficients based on the interaction index.

        Args:
            index: The interaction index

        Returns:
            An array with weights based on the interaction index
        """
        if index == "SII":
            weights = self._get_bernoulli_weights(interaction_size)
        if index == "k-SII":
            weights = self._get_aggregation_weights(interaction_size)

        return weights

    def _get_aggregation_weights(self, interaction_size: int) -> np.ndarray:
        """Pre-computes and array of weights for k-SII. Returns 1 if interaction is fully contained, 0 otherwise.

        Returns:
            An array of regression coefficient weights for k-SII
        """
        weights = np.zeros(interaction_size + 1)
        weights[-1] = 1
        return weights

    def _get_bernoulli_weights(self, interaction_size: int) -> np.ndarray:
        """Pre-computes and array of Bernoulli weights for the current interaction size.

        Args:
            interaction_size: The size of the interaction

        Returns:
            An array of the (regression coefficient) Bernoulli weights for the current interaction size.
        """
        bernoulli_weights = np.zeros(interaction_size + 1)
        for intersection_size in range(interaction_size + 1):
            bernoulli_weights[intersection_size] = self._bernoulli_weights(
                intersection_size, interaction_size
            )
        return bernoulli_weights

    def _bernoulli_weights(self, intersection_size: int, interaction_size: int) -> float:
        """Computes the weights of SII in the k-additive approximation.

        The weights are based on the size of the interaction and
        the size of the intersection of the interaction and the coalition.

        Args:
            intersection_size: The size of the intersection
            interaction_size: The size of the interaction

        Returns:
            The weight of SII in the k-additive approximation.
        """
        weight = 0
        for sum_index in range(1, intersection_size + 1):
            weight += (
                binom(intersection_size, sum_index)
                * self._bernoulli_numbers[interaction_size - sum_index]
            )
        return weight
