"""This module contains the Faithful Shapley-GAX approximator to compute the SV"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

from ..base import Regression

import numpy as np
from scipy.special import binom

from shapiq.interaction_values import InteractionValues, finalize_computed_interactions

ValidRegressionPolySHAPIndices = Literal["SV"]


class PolySHAP(Regression[ValidRegressionPolySHAPIndices]):
    """Estimates the Shapley values using polynomial regression. Extends KernelSHAP.

    The explanation frontier — the set of interactions used as basis functions in the regression —
    is supplied as a pre-built dictionary mapping each coalition tuple to its column index.
    Use one of the subclasses (:class:`PolySHAPKAddFrontier`, :class:`PolySHAPPriorFrontier`,
    :class:`PolySHAPPartialFrontier`) to construct a frontier automatically rather than building
    the dictionary by hand.

    Args:
        n: The number of players.
        explanation_frontier: A dictionary mapping coalition tuples to their column positions in
            the regression matrix. Must contain every singleton ``(i,)`` for ``i`` in ``range(n)``.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure.
            Defaults to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights
            must be of shape ``(n + 1,)`` and determine the probability of sampling a coalition
            of a certain size. Defaults to ``None``.
        replacement: Whether to sample coalitions with replacement. Defaults to ``True``.
        random_state: The random state of the estimator. Defaults to ``None``.

    Attributes:
        n: The number of players.
        N: The set of players (``0`` to ``n - 1``).
        max_order: The interaction order of the approximation.
        min_order: The minimum order of the approximation (always ``1`` for regression).
        iteration_cost: The cost of a single iteration of the regression estimator.
        explanation_frontier: The active explanation frontier dictionary.
        runtime_last_approximate_run: Per-phase wall-clock timings from the most recent
            :meth:`approximate` call.
    """

    def __init__(
        self,
        n: int,
        explanation_frontier: dict,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            n,
            max_order=1,
            index=Literal["SV"],
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )

        self.projection_matrix = None

        # Every singleton must be present so that Shapley values can be read off directly.
        for i in self._grand_coalition_set:
            if (i,) not in explanation_frontier:
                raise ValueError("PolySHAP requires all main effects in the explanation frontier.")

        # Build the binary indicator matrix: rows = frontier terms, cols = players.
        self.interaction_matrix_binary = np.zeros((len(explanation_frontier), self.n), dtype=bool)
        self.explanation_frontier = explanation_frontier
        for S, pos in explanation_frontier.items():
            self.interaction_lookup[S] = pos
            self.interaction_matrix_binary[pos, S] = True

        self.runtime_last_approximate_run: dict[str, float] = {}

        # Exclude the empty-coalition term from the variable count.
        self.n_variables = len(explanation_frontier) - 1

    def _init_kernel_weights(self) -> np.ndarray:
        """Initialise the KernelSHAP kernel weights indexed by coalition size.

        Weights are set to zero for the empty and grand coalitions (which are
        handled as hard constraints) and follow the standard KernelSHAP formula
        otherwise.

        Returns:
            Weight vector of shape ``(n + 1,)``.
        """
        weight_vector = np.zeros(shape=self.n + 1)
        for coalition_size in range(self.n + 1):
            if coalition_size < 1 or coalition_size > self.n - 1:
                weight_vector[coalition_size] = 0
            else:
                weight_vector[coalition_size] = 1 / (
                    (self.n - 1) * binom(self.n - 2, coalition_size - 1)
                )
        return weight_vector

    def _transform_to_shapley(self, input_values):
        """Aggregate interaction-level values into Shapley values.

        Each interaction term's value is split equally among its members, so
        higher-order terms contribute a ``1/|S|`` share to each player in ``S``.

        Args:
            input_values: Array of per-frontier-term values (including the
                empty-coalition entry at index 0).

        Returns:
            Tuple ``(sv, sv_lookup)`` where *sv* is a length-``(n + 1)`` array
            of Shapley values (index 0 reserved for the empty coalition) and
            *sv_lookup* maps singleton tuples to their positions in *sv*.
        """
        sv = np.zeros(self.n + 1)
        sv_lookup = {}
        for interaction, interaction_pos in self.interaction_lookup.items():
            if len(interaction) == 0:
                sv[interaction_pos] = input_values[interaction_pos]
                sv_lookup[()] = interaction_pos
            for i in interaction:
                sv[i + 1] += input_values[interaction_pos] / len(interaction)
                sv_lookup[(i,)] = i + 1
        return sv, sv_lookup

    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
        **kwargs: Any,
    ) -> InteractionValues:
        """Approximate Shapley values via weighted least-squares regression.

        The estimator draws random coalitions, queries the game, then solves a
        weighted least-squares problem whose basis functions are the terms in the
        explanation frontier.  This generalises KernelSHAP: when the frontier
        contains only singletons the method reduces exactly to KernelSHAP.

        For background on KernelSHAP see
        `Lundberg and Lee (2017) <https://doi.org/10.48550/arXiv.1705.07874>`_.

        Args:
            budget: Total number of coalition evaluations (including the empty
                and grand coalition which are always queried).
            game: Callable that accepts a binary coalition matrix of shape
                ``(budget, n)`` and returns a value array of shape ``(budget,)``.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` containing the
            estimated Shapley values.
        """
        approximate_start_time = time.time()

        kernel_weights = self._init_kernel_weights()
        self.projection_matrix = np.identity(self.n_variables) - 1 / self.n_variables

        # Sample coalitions and query the game.
        self._sampler.sample(budget)
        sampling_end_time = time.time()
        self.runtime_last_approximate_run["sampling"] = sampling_end_time - approximate_start_time

        game_values = game(self._sampler.coalitions_matrix)
        game_evaluation_end_time = time.time()
        self.runtime_last_approximate_run["evaluations"] = (
            game_evaluation_end_time - sampling_end_time
        )

        # Centre game values on the empty-coalition baseline.
        empty_set_value = game_values[0]
        game_values -= empty_set_value
        full_set_value = game_values[1]

        sampling_normalization = np.sqrt(
            kernel_weights[self._sampler.coalitions_size[2:]]
            * self._sampler.sampling_adjustment_weights[2:]
        )

        # Build the weighted design matrix.
        # When interactions are included (n_variables > n) each column checks
        # whether the corresponding frontier set is a subset of the sampled coalition.
        if self.n_variables > self.n:
            x_tilde = np.zeros((budget - 2, self.n_variables))
            for pos, row in enumerate(self.interaction_matrix_binary[1:, :]):
                x_tilde[:, pos] = (
                    np.all(row <= self._sampler.coalitions_matrix[2:, :], axis=1)
                    * sampling_normalization
                )
        else:
            x_tilde = sampling_normalization[:, np.newaxis] * self._sampler.coalitions_matrix[2:, :]

        y_tilde = game_values[2:] * sampling_normalization

        # Solve the weighted least-squares problem.
        least_squares_solution = np.linalg.lstsq(
            x_tilde @ self.projection_matrix,
            y_tilde - full_set_value / self.n_variables * np.sum(x_tilde, axis=1),
            rcond=None,
        )[0]

        interaction_representation = np.zeros(self.n_variables + 1, dtype=float)
        interaction_representation[0] = empty_set_value
        interaction_representation[1:] = full_set_value / self.n_variables + least_squares_solution

        sv, sv_lookup = self._transform_to_shapley(interaction_representation)

        regression_end_time = time.time()
        self.runtime_last_approximate_run["regression"] = (
            regression_end_time - game_evaluation_end_time
        )

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