"""This module contains the Faithful Shapley-GAX approximator to compute the SV"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from scipy.special import binom

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions
from shapiq.utils.sets import powerset


class ExplanationFrontierGenerator:
    def __init__(self, N: set, random_state: int | None = None):
        self.N = N
        self.n = len(N)
        self.random_state = random_state

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
        explanation_basis = {}
        S_pos = 0
        # generate individual's basis
        for S in powerset(self.N, max_size=1):
            explanation_basis[S] = S_pos
            S_pos += 1

        # add interactions in random order
        perm = list(self.N)
        # set random state of numpy
        if self.random_state is not None:
            np.random.seed(self.random_state)
        np.random.shuffle(perm)
        for S in powerset(self.N, min_size=2):
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
                raise ValueError("Poly SHAP requires all main effects in the interaction lookup.")
        # Extend interaction_lookup with pre-defined interactions
        self.interaction_matrix_binary = np.zeros((len(explanation_frontier), self.n), dtype=bool)
        self.explanation_frontier = explanation_frontier
        for S, pos in explanation_frontier.items():
            self.interaction_lookup[S] = pos
            self.interaction_matrix_binary[pos, S] = True

        # init runtime dictionary of type float
        self.runtime_last_approximate_run: dict[str, float] = {}

        self.n_variables = len(explanation_frontier) - 1  # exclude empty coalition

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
        self.projection_matrix = np.identity(self.n_variables) - 1 / self.n_variables

        # get the coalitions
        self._sampler.sample(budget)
        sampling_end_time = time.time()
        self.runtime_last_approximate_run["sampling"] = sampling_end_time - approximate_start_time
        # query the game for the coalitions
        game_values = game(self._sampler.coalitions_matrix)
        game_evaluation_end_time = time.time()
        self.runtime_last_approximate_run["evaluations"] = (
            game_evaluation_end_time - sampling_end_time
        )

        # compute polyshap representation
        empty_set_value = game_values[0]
        game_values -= empty_set_value
        full_set_value = game_values[1]

        sampling_normalization = np.sqrt(
            kernel_weights[self._sampler.coalitions_size[2:]]
            * self._sampler.sampling_adjustment_weights[2:]
        )

        if self.n_variables>self.n: # interactions available
            X_tilde= np. zeros((budget-2,self.n_variables))
            for pos,row in enumerate(self.interaction_matrix_binary[1:,:]):
                X_tilde[:,pos] = np.all(row<= self._sampler.coalitions_matrix[2:,:],axis=1)*sampling_normalization
        else:
            X_tilde = sampling_normalization[:,np.newaxis]*self._sampler.coalitions_matrix[2:,:]


        y_tilde = game_values[2:] * sampling_normalization

        lstsq_solution = np.linalg.lstsq(
            X_tilde @ self.projection_matrix,
            y_tilde - full_set_value / self.n_variables * np.sum(X_tilde, axis=1),
            rcond=None,
        )[0]
        interaction_representation = np.zeros(self.n_variables + 1, dtype=float)
        interaction_representation[0] = empty_set_value
        interaction_representation[1:] = full_set_value / self.n_variables + lstsq_solution

        # Transform to Shapley values
        sv, sv_lookup = self._transform_to_shapley(interaction_representation)

        regression_end_time = time.time()
        self.runtime_last_approximate_run["regression"] = (
            regression_end_time - game_evaluation_end_time
        )

        # Transform the output to the interaction values
        result = InteractionValues(
            values=sv,
            index="SV",
            interaction_lookup=sv_lookup,
            baseline_value=float(empty_set_value),
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

    def _transform_to_shapley(self, input_values):
        # transformed_values = np.zeros_like(input_values)
        sv = np.zeros(self.n + 1)
        sv_lookup = {}
        for interaction, interaction_pos in self.interaction_lookup.items():
            if len(interaction) == 0:
                sv[interaction_pos] = input_values[interaction_pos]
                sv_lookup[()] = interaction_pos
            for i in interaction:
                # transformed_values[self.interaction_lookup[(i,)]] += input_values[
                #    interaction_pos
                # ] / len(interaction)
                sv[i + 1] += input_values[interaction_pos] / len(interaction)
                sv_lookup[(i,)] = i + 1
        return sv, sv_lookup
