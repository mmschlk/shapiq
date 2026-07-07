"""This module contains the PolySHAP approximator to compute Shapley values."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.special import binom

from shapiq.approximator.regression.base import Regression
from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.game import Game

ValidRegressionPolySHAPIndices = Literal["SV"]


class PolySHAP(Regression[ValidRegressionPolySHAPIndices]):
    """Estimate Shapley values using the PolySHAP regression algorithm.

    Generalises KernelSHAP :cite:t:`Lundberg.2017`; the algorithm is described in
    Fumagalli et al. (2026) :cite:t:`Fumagalli.2026a`.

    The explanation frontier — the set of interactions used as basis functions in the
    regression — is supplied as a pre-built dictionary mapping each coalition tuple to
    its column index.  Three approximators with different frontier schemes are available:
    :class:`PolySHAPKAdd` (k-additive), :class:`PolySHAPPartial` (budget-controlled
    random extension), and :class:`PolySHAPPrior` (user-supplied).

    Args:
        n: The number of players.
        explanation_frontier: A dictionary mapping coalition tuples to their column
            positions in the regression matrix.  Must contain every singleton ``(i,)``
            for ``i`` in ``range(n)``.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling
            procedure.  Defaults to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure.
            Must be of shape ``(n + 1,)`` and determines the probability of sampling
            a coalition of a given size.  Defaults to ``None``.
        random_state: The random state of the estimator.  Defaults to ``None``.

    Attributes:
        n: The number of players.
        N: The set of players (``0`` to ``n - 1``).
        max_order: Interaction order of the approximation (always ``1`` — PolySHAP
            targets Shapley values).
        min_order: Minimum interaction order (always ``0``).
        iteration_cost: The cost of a single iteration of the estimator.
        explanation_frontier: The active explanation frontier dictionary.
    """

    def __init__(
        self,
        n: int,
        explanation_frontier: dict,
        *,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the PolySHAP approximator from a pre-built explanation frontier."""
        super().__init__(
            n,
            max_order=1,
            index="SV",
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )

        self.projection_matrix = None

        # Every singleton must be present so that Shapley values can be read off directly.
        for i in self._grand_coalition_set:
            if (i,) not in explanation_frontier:
                msg = "PolySHAP requires all main effects in the explanation frontier."
                raise ValueError(msg)

        # Build the binary indicator matrix: rows = frontier terms, cols = players.
        self.interaction_matrix_binary = np.zeros((len(explanation_frontier), self.n), dtype=bool)
        self.explanation_frontier = explanation_frontier
        for S, pos in explanation_frontier.items():
            self.interaction_lookup[S] = pos
            self.interaction_matrix_binary[pos, S] = True

        # Exclude the empty-coalition term from the variable count.
        self.n_variables = len(explanation_frontier) - 1

    def _warn_if_underdefined(self, n_sampled: int, budget: int) -> None:
        """Emit a UserWarning when the least-squares system has more variables than samples.

        Args:
            n_sampled: Number of sampled coalitions (excluding the border pair).
            budget: The budget originally passed to :meth:`approximate`.
        """
        if n_sampled < self.n_variables:
            warnings.warn(
                f"The least-squares system is underdefined: {n_sampled} sampled coalition(s) "
                f"but {self.n_variables} frontier variable(s). "
                f"Increase the budget (currently {budget}) or reduce the explanation frontier "
                "size for reliable estimates.",
                UserWarning,
                stacklevel=3,
            )

    def _init_sv_kernel_weights(self) -> np.ndarray:
        """Initialise the order-1 (Shapley-value) regression kernel weights by coalition size.

        Weights are zero for the empty and grand coalitions (handled as hard
        constraints) and follow the KernelSHAP formula otherwise.

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

    def _transform_to_shapley(
        self, input_values: np.ndarray
    ) -> tuple[np.ndarray, dict[tuple[int, ...], int]]:
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
        game: Game | Callable[[np.ndarray], np.ndarray],
        *args: Any | None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximate Shapley values via weighted least-squares regression.

        Draws random coalitions, queries the game, then solves a weighted
        least-squares problem whose basis functions are the terms in the explanation
        frontier.  When the frontier contains only singletons the method reduces exactly
        to KernelSHAP :cite:t:`Lundberg.2017`.  For details see Fumagalli et al. (2026)
        :cite:t:`Fumagalli.2026a`.

        Args:
            budget: Total number of coalition evaluations (including the empty and
                grand coalition, which are always queried).
            game: Callable accepting a binary coalition matrix of shape
                ``(budget, n)`` and returning a value array of shape ``(budget,)``.
            *args: Ignored; accepted for API compatibility with other approximators.
            **kwargs: Ignored; accepted for API compatibility with other approximators.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` containing the
            estimated Shapley values.
        """
        kernel_weights = self._init_sv_kernel_weights()
        self.projection_matrix = np.identity(self.n_variables) - 1 / self.n_variables

        # Sample coalitions and query the game.
        self._sampler.sample(budget)
        game_values = game(self._sampler.coalitions_matrix)

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
        # Use the actual sample count: the border-trick may cap evaluations below budget.
        n_sampled = len(self._sampler.coalitions_matrix) - 2
        self._warn_if_underdefined(n_sampled, budget)
        if self.n_variables > self.n:
            x_tilde = np.zeros((n_sampled, self.n_variables))
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

        return InteractionValues(
            values=sv,
            index="SV",
            interaction_lookup=sv_lookup,
            baseline_value=float(empty_set_value),
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            estimated=not budget >= 2**self.n,
            estimation_budget=budget,
            target_index=self.index,
        )
