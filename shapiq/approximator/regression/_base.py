"""This module contains the Base Regression approximator to compute SII and k-SII of arbitrary max_order."""

import copy
import warnings
from typing import Callable, Optional

import numpy as np
from scipy.special import bernoulli, binom

from shapiq.approximator._base import Approximator
from shapiq.indices import AVAILABLE_INDICES_REGRESSION
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset


class Regression(Approximator):
    """This class is the base class for all regression approximators.

    Regression approximators are based on a representation of the interaction index as a solution
    to a weighted least square problem. The objective of this optimization problem is approximated
    and then solved exactly. For the Shapley value this method is known as KernelSHAP.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The interaction index to be estimated. Available indices are ``['SII', 'k-SII', 'STII',
            'FSII']``.
        sii_consistent: If ``True``, the KernelSHAP-IQ method is used for SII, else Inconsistent
            KernelSHAP-IQ. Defaults to ``True``.
        pairing_trick: If `True`, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a coalition
            of a certain size. Defaults to ``None``.
        random_state: The random state to use for the approximation. Defaults to ``None``.
    """

    def __init__(
        self,
        n: int,
        max_order: int,
        index: str,
        sii_consistent: bool = True,
        pairing_trick: bool = False,
        sampling_weights: Optional[np.ndarray] = None,
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
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
        self._bernoulli_numbers = bernoulli(self.n)  # used for SII
        self._sii_consistent = (
            sii_consistent  # used for SII, if False, then Inconsistent KernelSHAP-IQ is used
        )

    def _init_kernel_weights(self, interaction_size: int) -> np.ndarray:
        """Initializes the kernel weights for the regression in KernelSHAP-IQ.

        The kernel weights are of size n + 1 and indexed by the size of the coalition. The kernel
        weights depend on the size of the interactions and are set to a large number for the edges.

        Args:
            interaction_size: The size of the interaction.

        Returns:
            The weights for sampling subsets of size s in shape (n + 1,).
        """
        # vector that determines the kernel weights for KernelSHAPIQ
        weight_vector = np.zeros(shape=self.n + 1)
        for coalition_size in range(0, self.n + 1):
            if (coalition_size < interaction_size) or (coalition_size > self.n - interaction_size):
                weight_vector[coalition_size] = self._big_M
            else:
                weight_vector[coalition_size] = 1 / (
                    (self.n - 2 * interaction_size + 1)
                    * binom(self.n - 2 * interaction_size, coalition_size - interaction_size)
                )
        kernel_weight = weight_vector
        return kernel_weight

    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
    ) -> InteractionValues:
        """The main approximation routine for the regression approximators.
        The regression estimators approximate Shapley Interactions based on a representation as a
        weighted least-square (WLSQ) problem. The regression approximator first approximates the
        objective of the WLSQ problem by Monte Carlo sampling and then computes an exact WLSQ
        solution based on the approximated objective. This approximation is an extension of
        KernelSHAP with different variants of kernel weights and regression settings.
        For details on KernelSHAP, refer to `Lundberg and Lee (2017) <https://doi.org/10.48550/arXiv.1705.07874>`_.

        Args:
            budget: The budget of the approximation.
            game: The game to be approximated.

        Returns:
            The `InteractionValues` object containing the estimated interaction values.
        """

        # initialize the kernel weights
        kernel_weights_dict = {}
        for interaction_size in range(1, self.max_order + 1):
            kernel_weights_dict[interaction_size] = self._init_kernel_weights(interaction_size)

        # get the coalitions
        self._sampler.sample(budget)

        # query the game for the coalitions
        game_values = game(self._sampler.coalitions_matrix)

        index_approximation = self.index
        if self.index == "k-SII":
            index_approximation = "SII"  # for k-SII, SII values are approximated and aggregated

        if index_approximation == "SII" and self._sii_consistent:
            shapley_interactions_values = self.kernel_shap_iq_routine(
                kernel_weights_dict=kernel_weights_dict, game_values=game_values
            )
        else:
            shapley_interactions_values = self.regression_routine(
                kernel_weights=kernel_weights_dict[1],
                game_values=game_values,
                index_approximation=index_approximation,
            )

        baseline_value = float(game_values[self._sampler.empty_coalition_index])

        return self._finalize_result(
            result=shapley_interactions_values, baseline_value=baseline_value, budget=budget
        )

    def kernel_shap_iq_routine(
        self, kernel_weights_dict: dict, game_values: np.ndarray
    ) -> np.ndarray[float]:
        """The main regression routine for the KernelSHAP-IQ approximator.

        This method solves for each interaction_size up to self.max_order separate regression
        problems iteratively. The following regression is thereby fit on the residuals of the
        previous regression problem.
        For details, refer to `Fumagalli et al. (2024) <https://doi.org/10.48550/arXiv.2405.10852>`_.

        Args:
            kernel_weights_dict: The weights of the regression problem as a dictionary per
                interaction size containing a numpy array with the regression weights per
                coalition size.
            game_values: The computed game values for the sampled coalitions.

        Returns:
            The approximated SII values of the KernelSHAP-IQ routine
        """

        coalitions_matrix = self._sampler.coalitions_matrix
        sampling_adjustment_weights = self._sampler.sampling_adjustment_weights
        coalitions_size = np.sum(coalitions_matrix, axis=1)
        sampling_adjustment_weights = sampling_adjustment_weights

        # set up the storage mechanisms
        empty_coalition_value = float(game_values[coalitions_size == 0][0])
        residual_game_values = {1: copy.copy(game_values)}
        residual_game_values[1] -= empty_coalition_value
        sii_values = np.array([empty_coalition_value])

        regression_coefficient_weight = self._get_regression_coefficient_weights(
            max_order=self.max_order, index="SII"
        )

        # iterate over the interaction sizes and compute the sii values via the WLSQ regression
        for interaction_size in range(1, self.max_order + 1):
            regression_matrix = np.zeros(
                (np.shape(coalitions_matrix)[0], int(binom(self.n, interaction_size)))
            )
            for coalition_pos, coalition in enumerate(coalitions_matrix):
                for interaction_pos, interaction in enumerate(
                    powerset(
                        self._grand_coalition_set,
                        min_size=interaction_size,
                        max_size=interaction_size,
                    )
                ):
                    intersection_size = np.sum(coalition[list(interaction)])
                    regression_matrix[coalition_pos, interaction_pos] = (
                        regression_coefficient_weight[interaction_size, intersection_size]
                    )

            # Regression weights adjusted by sampling weights
            regression_weights = (
                kernel_weights_dict[interaction_size][coalitions_size] * sampling_adjustment_weights
            )

            if interaction_size <= 2:
                # get \phi_i via solving the regression problem
                sii_values_current_size = self._solve_regression(
                    regression_matrix=regression_matrix,
                    regression_response=residual_game_values[interaction_size],
                    regression_weights=regression_weights,
                )
            else:
                # for order > 2 use ground truth weights for sizes < interaction_size and > n -
                # interaction_size
                ground_truth_weights_indicator = (coalitions_size < interaction_size) + (
                    coalitions_size > self.n - interaction_size
                )
                weights_from_ground_truth = self._get_ground_truth_sii_weights(
                    coalitions_matrix[ground_truth_weights_indicator], interaction_size
                )
                sii_values_current_size_minus = np.dot(
                    weights_from_ground_truth.T,
                    residual_game_values[interaction_size][ground_truth_weights_indicator],
                )

                # for interaction_size <= coalition size <= n-interaction_size solve the WLSQ
                game_values_plus = copy.deepcopy(residual_game_values[interaction_size])
                game_values_plus[ground_truth_weights_indicator] = 0

                # get \phi_i via solving the regression problem
                sii_values_current_size_plus = self._solve_regression(
                    regression_matrix=regression_matrix,
                    regression_response=game_values_plus,
                    regression_weights=regression_weights,
                )

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
        game_values: np.ndarray,
        index_approximation: str,
    ) -> np.ndarray[float]:
        """The main regression routine for the regression approximators.

        Solves a weighted least-square problem based on the representation of the target index.
        First, approximates the objective of the regression problem and then solves the regression
        problem using the approximated objective. Extends on KernelSHAP in different forms and
        computes all interactions using a single regression problem.

        Args:
            kernel_weights: An array of the regression weights associated with each coalition size.
            game_values: The computed game values for the sampled coalitions.
            index_approximation: The current index that is approximated.

        Returns:
            A numpy array of the approximated interaction values.
        """

        coalitions_matrix = self._sampler.coalitions_matrix
        sampling_adjustment_weights = self._sampler.sampling_adjustment_weights
        coalitions_size = np.sum(coalitions_matrix, axis=1)
        sampling_adjustment_weights = sampling_adjustment_weights

        empty_coalition_value = float(game_values[coalitions_size == 0][0])
        regression_response = game_values - empty_coalition_value
        regression_coefficient_weight = self._get_regression_coefficient_weights(
            max_order=self.max_order, index=index_approximation
        )
        n_interactions = np.sum(
            [int(binom(self.n, interaction_size)) for interaction_size in range(self.max_order + 1)]
        )
        regression_matrix = np.zeros((np.shape(coalitions_matrix)[0], n_interactions))

        for coalition_pos, coalition in enumerate(coalitions_matrix):
            for interaction_pos, interaction in enumerate(
                powerset(self._grand_coalition_set, max_size=self.max_order)
            ):
                interaction_size = len(interaction)
                intersection_size = np.sum(coalition[list(interaction)])
                regression_matrix[coalition_pos, interaction_pos] = regression_coefficient_weight[
                    interaction_size, intersection_size
                ]

        # Regression weights adjusted by sampling weights
        regression_weights = kernel_weights[coalitions_size] * sampling_adjustment_weights
        shapley_interactions_values = self._solve_regression(
            regression_matrix=regression_matrix,
            regression_response=regression_response,
            regression_weights=regression_weights,
        )

        if index_approximation == "kADD-SHAP":
            shapley_interactions_values[0] += empty_coalition_value
        else:
            shapley_interactions_values[0] = empty_coalition_value

        return shapley_interactions_values

    @staticmethod
    def _solve_regression(
        regression_matrix: np.ndarray,
        regression_response: np.ndarray,
        regression_weights: np.ndarray,
    ) -> np.ndarray[float]:
        """Solves the regression problem using the weighted least squares method. Returns all
        approximated interactions.

        Args:
            regression_matrix: The regression matrix of shape ``[n_coalitions, n_interactions]``.
                Depends on the index.
            regression_response: The response vector for each coalition.
            regression_weights: The weights for the regression problem for each coalition.

        Returns:
            The solution to the regression problem.
        """
        # regression weights adjusted by sampling weights
        weighted_regression_matrix = regression_weights[:, None] * regression_matrix

        try:
            # try solving via solve function
            shapley_interactions_values = np.linalg.solve(
                regression_matrix.T @ weighted_regression_matrix,
                weighted_regression_matrix.T @ regression_response,
            )
        except np.linalg.LinAlgError:
            # solve WLSQ via lstsq function and throw warning
            regression_weights_sqrt_matrix = np.diag(np.sqrt(regression_weights))
            regression_lhs = np.dot(regression_weights_sqrt_matrix, regression_matrix)
            regression_rhs = np.dot(regression_weights_sqrt_matrix, regression_response)
            warnings.warn(
                UserWarning(
                    "Linear regression equation is singular, a least squares solutions is used "
                    "instead.\n"
                )
            )
            shapley_interactions_values = np.linalg.lstsq(
                regression_lhs, regression_rhs, rcond=None
            )[0]

        return shapley_interactions_values.astype(dtype=float)

    def _get_regression_coefficient_weights(self, max_order: int, index: str) -> np.ndarray:
        """Pre-computes the regression coefficient weights based on the index and the max_order.
        Bernoulli weights for SII and kADD-SHAP. Binary weights for FSI.

           Args:
                max_order: The highest interaction size considered
                index: The interaction index

           Returns:
               An array of the regression coefficient weights.
        """
        if index in ["SII"]:
            weights = self._get_bernoulli_weights(max_order=max_order)
        elif index in ["kADD-SHAP"]:
            weights = self._get_kadd_weights(max_order=max_order)
        elif index == "FSII":
            # Default weights for FSI
            weights = np.zeros((max_order + 1, max_order + 1))
            for interaction_size in range(1, max_order + 1):
                # 1 if interaction is fully contained, else 0.
                weights[interaction_size, interaction_size] = 1
        else:
            raise ValueError(f"Index {index} not available for Regression Approximator.")
        return weights

    def _get_bernoulli_weights(self, max_order: int) -> np.ndarray:
        """Pre-computes and array of Bernoulli weights for a given max_order.

        Args:
            max_order: The highest interaction size considered

        Returns:
            An array of the (regression coefficient) Bernoulli weights for all interaction sizes up
                to the max_order.
        """
        bernoulli_weights = np.zeros((max_order + 1, max_order + 1))
        for interaction_size in range(1, max_order + 1):
            for intersection_size in range(interaction_size + 1):
                bernoulli_weights[interaction_size, intersection_size] = self._bernoulli_weights(
                    intersection_size, interaction_size
                )
        return bernoulli_weights

    def _get_kadd_weights(self, max_order: int) -> np.ndarray:
        """Pre-computes and array of Bernoulli weights for a given max_order.

        Args:
            max_order: The highest interaction size considered

        Returns:
            An array of the (regression coefficient) Bernoulli weights for all interaction sizes up
                to the max_order.
        """
        bernoulli_weights = np.zeros((max_order + 1, max_order + 1))
        for interaction_size in range(max_order + 1):
            for intersection_size in range(interaction_size + 1):
                bernoulli_weights[interaction_size, intersection_size] = self._kadd_weights(
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

    def _kadd_weights(self, intersection_size: int, interaction_size: int) -> float:
        """Computes the weights of SII in the k-additive approximation.
        Similar to _bernoulli_weights but sum ranges from zero.

        The weights are based on the size of the interaction and
        the size of the intersection of the interaction and the coalition.

        Args:
            intersection_size: The size of the intersection
            interaction_size: The size of the interaction

        Returns:
            The weight of SII in the k-additive approximation.
        """
        weight = 0
        for sum_index in range(intersection_size + 1):
            weight += (
                binom(intersection_size, sum_index)
                * self._bernoulli_numbers[interaction_size - sum_index]
            )
        return weight

    def _get_ground_truth_sii_weights(self, coalitions, interaction_size: int) -> np.ndarray:
        """Returns the ground truth SII weights for the coalitions per interaction.

        Args:
            coalitions: A binary coalition matrix for which the ground truth weights should be
                computed

        Returns:
            An array of weights with weights per coalition and per interaction
        """

        coalition_sizes = np.unique(np.sum(coalitions, axis=1))

        ground_truth_sii_weights = np.zeros((len(coalition_sizes), interaction_size + 1))

        # Pre-compute weights
        for coalition_size_pos, coalition_size in enumerate(coalition_sizes):
            for intersection_size in range(
                max(0, coalition_size + interaction_size - self.n),
                min(coalition_size, interaction_size) + 1,
            ):
                ground_truth_sii_weights[coalition_size_pos, intersection_size] = (
                    self._ground_truth_sii_weight(
                        coalition_size, interaction_size, intersection_size
                    )
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
        """Returns the ground truth SII weight for a given coalition size, interaction size and
            its intersection size.

        Args:
            coalition_size: The size of the coalition
            interaction_size: The size of the interaction
            intersection_size: The size of the intersection  TODO add more details here what intersection size is

        Returns:
            The ground truth SII weight
        """
        return (-1) ** (interaction_size - intersection_size) / (
            (self.n - interaction_size + 1)
            * binom(self.n - interaction_size, coalition_size - intersection_size)
        )
