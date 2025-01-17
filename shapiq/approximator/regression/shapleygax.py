"""This module contains the Faithful Shapley-GAX approximator to compute the SV"""

import warnings
from typing import Callable, Optional

import numpy as np
from scipy.special import bernoulli, binom

from shapiq.approximator._base import Approximator
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset


class ShapleyGAX(Approximator):
    """Estimates the Faithful Shapley-GAX using pre-defined interactions and computes the SV`_.

    Args:
        n: The number of players.
        output_order: The order of the explanation that is output.
        interaction_lookup: A dictionary containing all interactions and their ordering.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a coalition
            of a certain size. Defaults to ``None``.
        random_state: The random state of the estimator. Defaults to ``None``.

    Attributes:
        n: The number of players.
        N: The set of players (starting from ``0`` to ``n - 1``).
        max_order: The interaction order of the approximation.
        min_order: The minimum order of the approximation. For the regression estimator, min_order
            is equal to ``1``.
        iteration_cost: The cost of a single iteration of the regression SII.
    """

    def __init__(
        self,
        n: int,
        explanation_basis: dict = None,
        pairing_trick: bool = False,
        sampling_weights: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n,
            min_order=0,
            max_order=1,
            index="SV",
            top_order=False,
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )

        self.gax_interactions = explanation_basis
        # Verify gax_interactions
        for i in self._grand_coalition_set:
            if (i,) not in explanation_basis:
                raise ValueError("Shapley-GAX requires all main effects in the interaction lookup.")
        # Extend interaction_lookup with pre-defined interactions
        for S, pos in explanation_basis.items():
            self.interaction_lookup[S] = pos

    def _init_kernel_weights(self) -> np.ndarray:
        """Initializes the kernel weights for the regression in KernelSHAP-IQ.

        The kernel weights are of size n + 1 and indexed by the size of the coalition. The kernel
        weights are set to a large number for the empty coalition and grand coalition.

        Returns:
            The weights for sampling subsets of size s in shape (n + 1,).
        """
        # vector that determines the kernel weights for KernelSHAPIQ
        weight_vector = np.zeros(shape=self.n + 1)
        for coalition_size in range(0, self.n + 1):
            if (coalition_size < 1) or (coalition_size > self.n - 1):
                # Constraints
                weight_vector[coalition_size] = self._big_M
            else:
                weight_vector[coalition_size] = 1 / (
                    (self.n - 1) * binom(self.n - 2, coalition_size - 1)
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
        kernel_weights = self._init_kernel_weights()

        # get the coalitions
        self._sampler.sample(budget)

        # query the game for the coalitions
        game_values = game(self._sampler.coalitions_matrix)

        shapley_interactions_values = self.regression_routine(
            kernel_weights=kernel_weights,
            game_values=game_values,
        )
        baseline_value = float(game_values[self._sampler.empty_coalition_index])

        # test = self._transform_moebius_to_interactions(shapley_interactions_values)
        # test2 = self._transform_interactions_to_moebius(test)
        fsii_output_order = self._transform_fsii_to_shap(shapley_interactions_values)

        return self._finalize_result(
            result=fsii_output_order, baseline_value=baseline_value, budget=budget
        )

    def regression_routine(
        self,
        kernel_weights: np.ndarray,
        game_values: np.ndarray,
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
        regression_response = game_values  # - empty_coalition_value

        n_interactions = len(self.interaction_lookup)

        regression_matrix = np.zeros((np.shape(coalitions_matrix)[0], n_interactions))

        for coalition_pos, coalition in enumerate(coalitions_matrix):
            for interaction, interaction_pos in self.interaction_lookup.items():
                interaction_size = len(interaction)
                intersection_size = np.sum(coalition[list(interaction)])
                regression_matrix[coalition_pos, interaction_pos] = int(
                    interaction_size == intersection_size
                )

        # Regression weights adjusted by sampling weights
        regression_weights = kernel_weights[coalitions_size] * sampling_adjustment_weights
        shapley_interactions_values = self._solve_regression(
            regression_matrix=regression_matrix,
            regression_response=regression_response,
            regression_weights=regression_weights,
        )

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

    def _transform_moebius_to_interactions(self, input_values, max_order):
        transformed_values = np.zeros_like(input_values)
        for S, S_pos in self.gax_interactions.items():
            for S_tilde, S_tilde_pos in self.gax_interactions.items():
                if set(S).issubset(S_tilde):
                    S_tilde_effect = input_values[S_tilde_pos]
                    # normalization with bernoulli numbers
                    transformed_values[S_pos] += S_tilde_effect / (len(S_tilde) - len(S) + 1)
        return transformed_values

    def _transform_interactions_to_moebius(self, input_values):
        transformed_values = np.zeros_like(input_values)
        bernoulli_numbers = bernoulli(self.n)  # all subsets S with 1 <= |S| <= n
        for S, S_pos in self.gax_interactions.items():
            for S_tilde, S_tilde_pos in self.gax_interactions.items():
                if set(S).issubset(S_tilde):
                    S_tilde_effect = input_values[S_tilde_pos]
                    # normalization with bernoulli numbers
                    transformed_values[S_pos] += (
                        bernoulli_numbers[len(S_tilde) - len(S)] * S_tilde_effect
                    )
        return transformed_values

    def _transform_fsii(self, input_values):
        transformed_values = np.zeros_like(input_values)
        bernoulli_numbers = bernoulli(self.max_order)  # all subsets S with 1 <= |S| <= n
        for i, S in enumerate(
            reversed(list(powerset(self._grand_coalition_set, min_size=1, max_size=self.max_order)))
        ):
            S_pos = self.interaction_lookup[S]
            S_effect = input_values[S_pos]
            subset_size = len(S)
            # go over all subsets S_tilde of length |S| + 1, ..., n that contain S
            for S_tilde in powerset(
                self._grand_coalition_set, min_size=subset_size + 1, max_size=self.max_order
            ):
                if not set(S).issubset(S_tilde):
                    continue
                # get the effect of T
                S_tilde_effect = transformed_values[self.interaction_lookup[S_tilde]]
                # normalization with bernoulli numbers
                S_effect -= bernoulli_numbers[len(S_tilde) - subset_size] * S_tilde_effect
            transformed_values[S_pos] = S_effect
        return transformed_values

    def _transform_fsii_to_shap(self, input_values):
        transformed_values = np.zeros_like(input_values)
        for interaction, interaction_pos in self.interaction_lookup.items():
            for i in interaction:
                transformed_values[self.interaction_lookup[(i,)]] += input_values[
                    interaction_pos
                ] / len(interaction)
        return transformed_values
