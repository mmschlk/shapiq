"""This module contains the Faithful Shapley-GAX approximator to compute the SV"""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable

import numpy as np
from scipy.special import bernoulli, binom
from sklearn.linear_model import LassoLarsIC

from shapiq.approximator.base import Approximator
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions
from shapiq.utils.sets import powerset

import time


class ExplanationFrontierGenerator:
    def __init__(self, N: set):
        self.N = N
        self.n = len(N)

    def generate_kadd(self, max_order, sizes_to_exclude=None):
        explanation_basis = {}
        pos = 0
        for S in powerset(self.N, max_size=max_order):
            if sizes_to_exclude is None or len(S) not in sizes_to_exclude:
                explanation_basis[S] = pos
                pos += 1
        return explanation_basis

    def generate_prior(self, Q_prior):
        explanation_basis = {}
        pos = 0
        for S in Q_prior:
            explanation_basis[S] = pos
            pos += 1
        return explanation_basis

    def generate_partial(self, n_explanation_terms, sizes_to_exclude=None):
        perm = list(self.N)
        np.random.shuffle(perm)
        explanation_basis = {}
        S_pos = 0
        for S in powerset(self.N):
            if sizes_to_exclude is not None and len(S) in sizes_to_exclude:
                continue
            if S_pos < n_explanation_terms:
                explanation_basis[tuple(sorted([perm[i] for i in S]))] = S_pos
                S_pos += 1
            else:
                break
        return explanation_basis


class PolySHAP(Approximator):
    """Estimates the Shapley values using polynomial regression. Extends KernelSHAP.`_.

    Args:
        n: The number of players.
        explanation_frontier: A dictionary containing all interactions and their position.
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
        explanation_frontier: dict = None,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        replacement: bool = True,
        random_state: int | None = None,
    ):
        super().__init__(
            n,
            min_order=0,
            max_order=1,
            index="SV",
            top_order=False,
            random_state=random_state,
            pairing_trick=pairing_trick,
            replacement=replacement,
            sampling_weights=sampling_weights,
        )

        # Verify explanation set
        for i in self._grand_coalition_set:
            if (i,) not in explanation_frontier:
                raise ValueError(
                    "Poly SHAP requires all main effects in the interaction lookup."
                )
        # Extend interaction_lookup with pre-defined interactions
        for S, pos in explanation_frontier.items():
            self.interaction_lookup[S] = pos

        # init runtime dictionary of type float
        self.runtime_last_approximate_run: dict[str, float] = {}

    def _init_kernel_weights(self) -> np.ndarray:
        """Initializes the kernel weights for the regression in KernelSHAP-IQ.

        The kernel weights are of size n + 1 and indexed by the size of the coalition. The kernel
        weights are set to a large number for the empty coalition and grand coalition.

        Returns:
            The weights for sampling subsets of size s in shape (n + 1,).
        """
        # vector that determines the kernel weights for KernelSHAPIQ
        weight_vector = np.zeros(shape=self.n + 1)
        for coalition_size in range(self.n + 1):
            if (coalition_size < 1) or (coalition_size > self.n - 1):
                # Constraints
                weight_vector[coalition_size] = 0
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
        approximate_start_time = time.time()
        # initialize the kernel weights
        kernel_weights = self._init_kernel_weights()

        # get the coalitions
        self._sampler.sample(budget)
        sampling_end_time = time.time()
        self.runtime_last_approximate_run["sampling"] = (
            sampling_end_time - approximate_start_time
        )
        # query the game for the coalitions
        game_values = game(self._sampler.coalitions_matrix)
        game_evaluation_end_time = time.time()
        self.runtime_last_approximate_run["evaluations"] = (
            game_evaluation_end_time - sampling_end_time
        )
        interaction_representation = self.regression_routine(
            kernel_weights=kernel_weights,
            game_values=game_values,
        )
        baseline_value = float(game_values[self._sampler.empty_coalition_index])

        sv = self._transform_to_shapley(interaction_representation)
        regression_end_time = time.time()
        self.runtime_last_approximate_run["regression"] = (
            regression_end_time - game_evaluation_end_time
        )

        sv_numpy = np.zeros(self.n + 1, dtype=float)
        sv_lookup = {}
        new_pos = 0
        for key, pos in self.interaction_lookup.items():
            if len(key) <= 1:
                sv_numpy[new_pos] = sv[pos]
                sv_lookup[key] = new_pos
                new_pos += 1

        # Transform the output to the interaction values
        result = InteractionValues(
            values=sv_numpy,
            index="SV",
            interaction_lookup=sv_lookup,
            baseline_value=baseline_value,
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            estimated=not budget >= 2**self.n,
            estimation_budget=budget,
        )

        final_result = finalize_computed_interactions(result, target_index=self.index)
        shapiq_post_processing_end_time = time.time()
        self.runtime_last_approximate_run["shapiq_post_processing"] = (
            shapiq_post_processing_end_time - regression_end_time
        )
        self.runtime_last_approximate_run["total"] = (
            shapiq_post_processing_end_time - approximate_start_time
        )
        return final_result

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

        empty_set_prediction = float(game_values[coalitions_size == 0][0])
        regression_response = game_values - empty_set_prediction

        n_regression_variables = len(self.interaction_lookup)
        regression_matrix = np.zeros(
            (np.shape(coalitions_matrix)[0], n_regression_variables)
        )

        for coalition_pos, coalition in enumerate(coalitions_matrix):
            for interaction, interaction_pos in self.interaction_lookup.items():
                interaction_size = len(interaction)
                intersection_size = np.sum(coalition[list(interaction)])
                regression_matrix[coalition_pos, interaction_pos] = int(
                    interaction_size == intersection_size
                )

        # Regression weights adjusted by sampling weights
        regression_weights = (
            kernel_weights[coalitions_size] * sampling_adjustment_weights
        )
        # Solve the regression problem
        shapley_interactions_values = self._solve_regression_unconstrained(
            regression_matrix=regression_matrix,
            regression_response=regression_response,
            regression_weights=regression_weights,
        )
        shapley_interactions_values[0] = empty_set_prediction
        return shapley_interactions_values.astype(dtype=float)

    def _solve_regression_unconstrained(
        self,
        regression_matrix: np.ndarray,
        regression_response: np.ndarray,
        regression_weights: np.ndarray,
    ) -> np.ndarray[float]:
        """Solves the regression problem using an unconstrained weighted least squares method by computing a projection matrix first. Assumes empty and full prediction correspond to the first two rows. Returns all
        approximated interactions.

        Args:
            regression_matrix: The regression matrix of shape ``[n_coalitions, n_interactions]``.
                Depends on the index.
            regression_response: The response vector for each coalition.
            regression_weights: The weights for the regression problem for each coalition.

        Returns:
            The solution to the regression problem.
        """
        # Retrieve empty and full set response
        empty_set_prediction = regression_response[0]
        full_set_prediction = regression_response[1]
        assert empty_set_prediction == 0
        # exclude the empty and full coalition
        regression_matrix_truncated = regression_matrix[2:, 1:]
        regression_response_truncated = regression_response[2:]
        regression_weights_truncated = regression_weights[2:]
        n_variables = np.shape(regression_matrix_truncated)[1]
        sum_of_rows = np.sum(regression_matrix_truncated, axis=1)
        response_modified = (
            regression_response_truncated
            - sum_of_rows * full_set_prediction / n_variables
        )
        projection_matrix = np.identity(n_variables) - 1 / n_variables

        # compute new regression matrices
        regression_response_modified = (
            projection_matrix
            @ regression_matrix_truncated.T
            @ np.diag(regression_weights_truncated)
            @ response_modified
        )
        regression_matrix_modified = (
            projection_matrix
            @ regression_matrix_truncated.T
            @ np.diag(regression_weights_truncated)
            @ regression_matrix_truncated
            @ projection_matrix
        )
        # compute solution
        regression_solution = np.linalg.lstsq(
            regression_matrix_modified, regression_response_modified, rcond=None
        )[0]
        # create shapley interaction values, add empty coalition variable back
        interaction_values = np.zeros(n_variables + 1, dtype=float)
        interaction_values[1:] = regression_solution + full_set_prediction / n_variables
        return interaction_values

    def _transform_to_shapley(self, input_values):
        transformed_values = np.zeros_like(input_values)
        for interaction, interaction_pos in self.interaction_lookup.items():
            for i in interaction:
                transformed_values[self.interaction_lookup[(i,)]] += input_values[
                    interaction_pos
                ] / len(interaction)
        return transformed_values
