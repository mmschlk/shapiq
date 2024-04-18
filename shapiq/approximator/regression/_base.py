"""This module contains the Base Regression approximator to compute SII and k-SII of arbitrary max_order."""

import copy
import warnings
from typing import Callable, Optional

import numpy as np
from scipy.special import bernoulli, binom

from shapiq.approximator._base import Approximator
from shapiq.approximator.k_sii import KShapleyMixin
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset

AVAILABLE_INDICES_REGRESSION = ["k-SII", "SII", "kADD-SHAP", "FSII", "kADD-SHAP"]


class Regression(Approximator, KShapleyMixin):
    def __init__(
        self,
        n: int,
        max_order: int,
        index: str,
        sii_consistent: bool = True,
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
        self._big_M = 10e7
        self._bernoulli_numbers = bernoulli(self.n)  # used for SII
        self._sii_consistent = (
            sii_consistent  # used for SII, if False, then Inconsistent KernelSHAP-IQ is used
        )

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
                    (self.n - 2 * interaction_size + 1)
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
        sampling_adjustment_weights = sampler.sampling_adjustment_weights
        n_coalitions = sampler.n_coalitions

        # calculate the number of iterations and the last batch size
        n_iterations, last_batch_size = self._calc_iteration_count(
            n_coalitions, batch_size, iteration_cost=self.iteration_cost
        )

        game_values: np.ndarray[float] = np.zeros(shape=(n_coalitions,), dtype=float)

        # main regression loop computing the FSII values
        batch_index = 0
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

            if self.index == "k-SII":
                # For k-SII the SII values are approximated and then aggregated
                index_approximation = "SII"
            else:
                index_approximation = self.index

            if index_approximation == "SII" and self._sii_consistent:
                shapley_interactions_values = self.kernelshapiq_routine(
                    kernel_weights_dict=kernel_weights_dict,
                    batch_game_values=batch_game_values,
                    batch_coalitions_matrix=batch_coalitions_matrix,
                    batch_coalitions_size=batch_coalitions_size,
                    batch_sampling_adjustment_weights=batch_sampling_adjustment_weights,
                    index_approximation=index_approximation,
                )
            else:
                shapley_interactions_values = self.regression_routine(
                    kernel_weights=kernel_weights_dict[1],
                    batch_game_values=batch_game_values,
                    batch_coalitions_matrix=batch_coalitions_matrix,
                    batch_coalitions_size=batch_coalitions_size,
                    batch_sampling_adjustment_weights=batch_sampling_adjustment_weights,
                    index_approximation=index_approximation,
                )

            if self.index == "k-SII":
                baseline_value = shapley_interactions_values[0]
                # Aggregate SII to k-SII, will change SII -> k-SII
                shapley_interactions_values = self.transforms_sii_to_ksii(
                    shapley_interactions_values
                )
                shapley_interactions_values[0] = baseline_value
            if np.shape(coalitions_matrix)[0] >= 2**self.n:
                estimated_indicator = False
            else:
                estimated_indicator = True
        return self._finalize_result(
            result=shapley_interactions_values, estimated=estimated_indicator, budget=budget
        )

    def kernelshapiq_routine(
        self,
        kernel_weights_dict: dict,
        batch_game_values: np.ndarray,
        batch_coalitions_matrix: np.ndarray,
        batch_coalitions_size: np.ndarray,
        batch_sampling_adjustment_weights: np.ndarray,
        index_approximation: str,
    ):
        residual_game_values = {}
        residual_game_values[1] = copy.copy(batch_game_values)
        emptycoalition_value = residual_game_values[1][batch_coalitions_size == 0][0]
        residual_game_values[1] -= emptycoalition_value

        sii_values = np.array([emptycoalition_value])

        regression_coefficient_weight = self._get_regression_coefficient_weights(
            max_order=self.max_order, index=index_approximation
        )

        for interaction_size in range(1, self.max_order + 1):
            regression_matrix = np.zeros(
                (np.shape(batch_coalitions_matrix)[0], int(binom(self.n, interaction_size)))
            )
            for coalition_pos, coalition in enumerate(batch_coalitions_matrix):
                for interaction_pos, interaction in enumerate(
                    powerset(self.N, min_size=interaction_size, max_size=interaction_size)
                ):
                    intersection_size = np.sum(coalition[list(interaction)])
                    regression_matrix[
                        coalition_pos, interaction_pos
                    ] = regression_coefficient_weight[interaction_size, intersection_size]

            # Regression weights adjusted by sampling weights
            regression_weights = (
                kernel_weights_dict[interaction_size][batch_coalitions_size]
                * batch_sampling_adjustment_weights
            )

            weighted_regression_matrix = regression_weights[:, None] * regression_matrix

            if interaction_size <= 2:
                try:
                    # Try solving via solve function
                    sii_values_current_size = np.linalg.solve(
                        regression_matrix.T @ weighted_regression_matrix,
                        weighted_regression_matrix.T @ residual_game_values[interaction_size],
                    )
                except np.linalg.LinAlgError:
                    # Solve WLSQ via lstsq function and throw warning
                    regression_weights_sqrt_matrix = np.diag(np.sqrt(regression_weights))
                    regression_lhs = np.dot(regression_weights_sqrt_matrix, regression_matrix)
                    regression_rhs = np.dot(
                        regression_weights_sqrt_matrix, residual_game_values[interaction_size]
                    )
                    warnings.warn(
                        "Linear regression equation is singular, a least squares solutions is used instead.\n"
                    )
                    sii_values_current_size = np.linalg.lstsq(
                        regression_lhs, regression_rhs, rcond=None
                    )[
                        0
                    ]  # \phi_i

            else:
                # For order > 2 use ground truth weights for sizes < interaction_size and > n - interaction_size
                ground_truth_weights_indicator = (batch_coalitions_size < interaction_size) + (
                    batch_coalitions_size > self.n - interaction_size
                )
                weights_from_ground_truth = self._get_ground_truth_sii_weights(
                    batch_coalitions_matrix[ground_truth_weights_indicator], interaction_size
                )
                sii_values_current_size_minus = np.dot(
                    weights_from_ground_truth.T,
                    residual_game_values[interaction_size][ground_truth_weights_indicator],
                )

                # For interaction_size <= coalition size <= n-interaction_size solve WLSQ problem
                game_values_plus = copy.deepcopy(residual_game_values[interaction_size])
                game_values_plus[ground_truth_weights_indicator] = 0

                try:
                    # Try solving via solve function
                    sii_values_current_size_plus = np.linalg.solve(
                        regression_matrix.T @ weighted_regression_matrix,
                        weighted_regression_matrix.T @ game_values_plus,
                    )
                except np.linalg.LinAlgError:
                    warnings.warn(
                        "Linear regression equation is singular, a least squares solutions is used instead.\n"
                    )
                    regression_weights_sqrt_matrix = np.diag(np.sqrt(regression_weights))
                    regression_lhs = np.dot(regression_weights_sqrt_matrix, regression_matrix)

                    regression_rhs = np.dot(regression_weights_sqrt_matrix, game_values_plus)
                    sii_values_current_size_plus = np.linalg.lstsq(
                        regression_lhs, regression_rhs, rcond=None
                    )[
                        0
                    ]  # \phi_i

                sii_values_current_size = (
                    sii_values_current_size_minus + sii_values_current_size_plus
                )

            approximations = np.dot(regression_matrix, sii_values_current_size)
            sii_values = np.hstack((sii_values, sii_values_current_size))
            residual_game_values[interaction_size + 1] = (
                residual_game_values[interaction_size] - approximations
            )

        return sii_values

    def regression_routine(
        self,
        kernel_weights: np.ndarray,
        batch_game_values: np.ndarray,
        batch_coalitions_matrix: np.ndarray,
        batch_coalitions_size: np.ndarray,
        batch_sampling_adjustment_weights: np.ndarray,
        index_approximation: str,
    ):
        regression_response = copy.copy(batch_game_values)
        emptycoalition_value = regression_response[batch_coalitions_size == 0][0]
        regression_response -= emptycoalition_value
        regression_coefficient_weight = self._get_regression_coefficient_weights(
            max_order=self.max_order, index=index_approximation
        )
        n_interactions = np.sum(
            [int(binom(self.n, interaction_size)) for interaction_size in range(self.max_order + 1)]
        )
        regression_matrix = np.zeros((np.shape(batch_coalitions_matrix)[0], n_interactions))

        for coalition_pos, coalition in enumerate(batch_coalitions_matrix):
            for interaction_pos, interaction in enumerate(
                powerset(self.N, max_size=self.max_order)
            ):
                interaction_size = len(interaction)
                intersection_size = np.sum(coalition[list(interaction)])
                regression_matrix[coalition_pos, interaction_pos] = regression_coefficient_weight[
                    interaction_size, intersection_size
                ]

        # Regression weights adjusted by sampling weights
        regression_weights = (
            kernel_weights[batch_coalitions_size] * batch_sampling_adjustment_weights
        )
        weighted_regression_matrix = regression_weights[:, None] * regression_matrix

        try:
            # Try solving via solve function
            shapley_interactions_values = np.linalg.solve(
                regression_matrix.T @ weighted_regression_matrix,
                weighted_regression_matrix.T @ regression_response,
            )
        except np.linalg.LinAlgError:
            # Solve WLSQ via lstsq function and throw warning
            regression_weights_sqrt_matrix = np.diag(np.sqrt(regression_weights))
            regression_lhs = np.dot(regression_weights_sqrt_matrix, regression_matrix)
            regression_rhs = np.dot(regression_weights_sqrt_matrix, regression_response)
            warnings.warn(
                "Linear regression equation is singular, a least squares solutions is used instead.\n"
            )
            shapley_interactions_values = np.linalg.lstsq(
                regression_lhs, regression_rhs, rcond=None
            )[
                0
            ]  # \phi_i

        shapley_interactions_values[0] = emptycoalition_value

        return shapley_interactions_values

    def _get_regression_coefficient_weights(self, max_order: int, index: str) -> np.ndarray:
        """Pre-computes the regression coefficient weights based on the index and the max_order.
        Bernoulli weights for SII and kADD-SHAP. Binary weights for FSI.

           Args:
                max_order: The highest interaction size considered
                index: The interaction index

           Returns:
               An array of the regression coefficient weights.
        """
        if index in ["SII", "kADD-SHAP"]:
            weights = self._get_bernoulli_weights(max_order=max_order)
        if index == "FSII":
            # Default weights for FSI
            weights = np.zeros((max_order + 1, max_order + 1))
            for interaction_size in range(1, max_order + 1):
                # 1 if interaction is fully contained, else 0.
                weights[interaction_size, interaction_size] = 1
        return weights

    def _get_bernoulli_weights(self, max_order: int) -> np.ndarray:
        """Pre-computes and array of Bernoulli weights for a given max_order.

        Args:
            max_order: The highest interaction size considered

        Returns:
            An array of the (regression coefficient) Bernoulli weights for all interaction sizes up to the max_order.
        """
        bernoulli_weights = np.zeros((max_order + 1, max_order + 1))
        for interaction_size in range(1, max_order + 1):
            for intersection_size in range(interaction_size + 1):
                bernoulli_weights[interaction_size, intersection_size] = self._bernoulli_weights(
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

    def _get_ground_truth_sii_weights(self, coalitions, interaction_size: int) -> np.ndarray:
        """Returns the ground truth SII weights for the coalitions per interaction.

        Args:
            coalitions: A binary coalition matrix for which the ground truth weights should be computed

        Returns:
            An array of weights with weights per coalition and per interaction
        """

        coalition_sizes = np.unique(np.sum(coalitions, axis=1))

        ground_truth_sii_weights = np.zeros((len(coalition_sizes), interaction_size + 1))

        # Pre-compute weights
        for coalition_size_pos, coalition_size in enumerate(coalition_sizes):
            for intersection_size in range(min(coalition_size, interaction_size) + 1):
                ground_truth_sii_weights[
                    coalition_size_pos, intersection_size
                ] = self._ground_truth_sii_weight(
                    coalition_size, interaction_size, intersection_size
                )

        # Compute ground truth weights
        coalitions_sii_weights = np.zeros(
            (np.shape(coalitions)[0], int(binom(self.n, interaction_size))), dtype=float
        )

        for coalition_pos, coalition in enumerate(coalitions):
            coalition_size = np.sum(coalition)
            for interaction_pos, interaction in enumerate(
                powerset(
                    self._grand_coalition_set, min_size=interaction_size, max_size=interaction_size
                )
            ):
                intersection_size = np.sum(coalition[list(interaction)])
                coalitions_sii_weights[coalition_pos, interaction_pos] = ground_truth_sii_weights[
                    list(coalition_sizes).index(coalition_size), intersection_size
                ]

        return coalitions_sii_weights

    def _ground_truth_sii_weight(
        self, coalition_size: int, interaction_size: int, intersection_size: int
    ) -> float:
        """Returns the ground truth SII weight for a given coalition size, interaction size and its intersection size.

        Args:
            coalition_size: The size of the coalition
            interaction_size: The size of the interaction
            intersection_size: The size of the intersection

        Returns:
            The ground truth SII weight
        """
        return (-1) ** (interaction_size - intersection_size) / (
            (self.n - interaction_size + 1)
            * binom(self.n - interaction_size, coalition_size - intersection_size)
        )

    @staticmethod
    def _calc_iteration_count(budget: int, batch_size: int, iteration_cost: int) -> tuple[int, int]:
        """Computes the number of iterations and the size of the last batch given the batch size and
        the budget.

        Args:
            budget: The budget for the approximation.
            batch_size: The size of the batch.
            iteration_cost: The cost of a single iteration.

        Returns:
            int, int: The number of iterations and the size of the last batch.
        """
        n_iterations = budget // (iteration_cost * batch_size)
        last_batch_size = batch_size
        remaining_budget = budget - n_iterations * iteration_cost * batch_size
        if remaining_budget > 0 and remaining_budget // iteration_cost > 0:
            last_batch_size = remaining_budget // iteration_cost
            n_iterations += 1
        return n_iterations, last_batch_size

    def base_aggregation(
        self, base_interactions: InteractionValues, order: int
    ) -> InteractionValues:
        """Transform Base Interactions into Interactions satisfying efficiency, e.g. SII to k-SII

        Args:
            base_interactions: InteractionValues object containing interactions up to order "order"
            order: The highest order of interactions considered

        Returns:
            InteractionValues object containing transformed base_interactions
        """
        # TODO: move to central location
        transformed_values = np.zeros(
            np.sum(
                [
                    int(binom(self.n, interaction_size))
                    for interaction_size in range(self.max_order + 1)
                ]
            )
        )
        transformed_lookup = {}
        bernoulli_numbers = bernoulli(order)  # lookup Bernoulli numbers
        for i, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order)):
            transformed_lookup[interaction] = i
            if len(interaction) == 0:
                # Initialize emptyset baseline value
                transformed_values[i] = base_interactions.baseline_value
            else:
                interaction_effect = base_interactions[interaction]
                subset_size = len(interaction)
                # go over all subsets S_tilde of length |S| + 1, ..., n that contain S
                for interaction_higher_order in powerset(
                    self._grand_coalition_set, min_size=subset_size + 1, max_size=order
                ):
                    if not set(interaction).issubset(interaction_higher_order):
                        continue
                    # get the effect of T
                    interaction_tilde_effect = base_interactions[interaction_higher_order]
                    # normalization with bernoulli numbers
                    interaction_effect += (
                        bernoulli_numbers[len(interaction_higher_order) - subset_size]
                        * interaction_tilde_effect
                    )
                transformed_values[i] = interaction_effect

        # setup interaction values
        transformed_index = base_interactions.index  # raname the index (e.g. SII -> k-SII)
        if transformed_index not in ["SV", "BV"]:
            transformed_index = "k-" + transformed_index
        transformed_interactions = InteractionValues(
            values=transformed_values,
            index=transformed_index,
            min_order=0,
            max_order=order,
            interaction_lookup=transformed_lookup,
            n_players=self.n,
            estimated=False,
        )
        return copy.deepcopy(transformed_interactions)
