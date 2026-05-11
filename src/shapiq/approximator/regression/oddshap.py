""" This module contains the OddSHAP approximator for estimating Shapley values.

OddSHAP is a value estimator based on paired sampling, odd-only Fourier regression, and sparse odd interaction detection as introduced in Fumagalli et al. (2026) :cite:t:`Fumagalli.2026`
"""

from __future__ import annotations
import lightgbm as lgb
import time

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import binom

from shapiq.approximator.base import Approximator
from shapiq.approximator.regression._oddshap_proxyspex_adapter import (
    lgboost_to_fourier,
    top_k_interactions,
)
from shapiq.interaction_values import InteractionValues
from shapiq.tree.explainer import TreeExplainer

if TYPE_CHECKING:
    from collections.abc import Callable
    from shapiq.game import Game


class OddSHAP(Approximator):
    """ Args

    """

    valid_indices: tuple[str, ...] = ("SV")

    @staticmethod
    def _init_sampling_weights_static(n: int) -> np.ndarray:
        """ Initialize OddSHAP coalition-size sampling weights.


        Args

        """
        weight_vector = np.zeros(n + 1, dtype=float)

        for coalition_size in range(n + 1):
            if coalition_size == 0 or coalition_size == n:
                weight_vector[coalition_size] = 0.0  # zero for empty and full coalition
            else:
                weight_vector[coalition_size] = 1.0 / (
                        (n - 1) * binom(n - 2, coalition_size - 1)
                )
        return weight_vector / np.sum(weight_vector)  # breaks when dividing by 0!!!!

    def __init__(
            self,
            n: int,
            *,
            pairing_trick: bool = True,  # Theorem 3.2 Alg. 1
            sampling_weights: np.ndarray | None = None,
            random_state: int | None = None,
            regression_basis: str = "Fourier",  # isolate odd terms
            interaction_detection: str = "ProxySPEX",  # screening sparse odd interactions
            odd_only: bool = True,
            interaction_factor: int = 10,  # standard setting according to paper
            tree_params: dict[str, Any] | None = None,
            **kwargs: Any
    ) -> None:
        # OddSHAP uses its own coalition-size distribution.
        # Compute here before calling the base class, because base class creates the sampler during super().__init__()
        if sampling_weights is None:
            sampling_weights = self._init_sampling_weights_static(n)

        super().__init__(
            n=n,
            max_order=1,
            index="SV",
            top_order=False,
            min_order=0,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
            random_state=random_state,
        )

        self.regression_basis = regression_basis
        self.interaction_detection = interaction_detection
        self.odd_only = odd_only
        self.interaction_factor = interaction_factor
        self.tree_params = tree_params

        self.odd_interaction_lookup: dict[tuple[int, ...], int] = {}
        self.odd_interaction_matrix_binary = np.zeros((0, self.n), dtype=bool)
        self.n_active_interactions: int = 0

        # init runtime dict of type float
        self.runtime_last_approximate_run: dict[str, float] = {}

    def approximate(
            self,
            budget: int,
            game: Game | Callable[[np.ndarray], np.ndarray],
            **kwargs: Any
    ) -> InteractionValues:
        """ Approximate first-order Shapley values

        main control flow: (Alg. 1 of OddSHAP paper)
        1. sample coalitions using the inherited CoalitionSampler
        2. evaluate the game on sampled coalitions
        3. extract empty and grand coalition values
        4. decide between fallback mode and odd-regression mode

        Args

        """
        start_time = time.time()

        # 1. Sample coalitions
        # use coalition sampler initialized in base class
        self._sampler.sample(budget)
        sampling_end_time = time.time()
        self.runtime_last_approximate_run["sampling"] = sampling_end_time - start_time

        coalitions = self._sampler.coalitions_matrix

        # 2. Evaluate game on all sampled coalitions
        game_values = np.asarray(game(coalitions), dtype=float)
        evaluation_end_time = time.time()
        self.runtime_last_approximate_run["evaluations"] = evaluation_end_time - sampling_end_time

        # 3. Extract empty and grand coalition values (CoalitionSampler ensures both are present)
        empty_idx = self._sampler.empty_coalition_index
        if empty_idx is None:
            msg = "OddSHAP expected empty coalition to be present in the sampled coalitions"
            raise RuntimeError(msg)

        empty_set_value = float(game_values[empty_idx])

        full_mask = np.sum(coalitions, axis=1) == self.n
        if not np.any(full_mask):
            msg = "OddSHAP expected grand coalition to be present in the sampled coalitions"
            raise RuntimeError(msg)

        full_set_value = float(game_values[np.where(full_mask)[0][0]])

        #centered_game_values = game_values - empty_set_value  # normalize response relative to empty coalition

        # 4. Compute how many higher-order interactions OddSHAP is allowed to consider later
        n_candidate_interactions = max(0, budget // self.interaction_factor - self.n) #TODO: does paper do it with or without subtraction?

        # branch between: low-budget fallback and odd-regressioin path
        if budget < self.n * self.interaction_factor:
            result = self._approximate_via_fallback(
                budget=budget,
                coalitions=coalitions,
                game_values=game_values,
                #centered_game_values=centered_game_values,
                empty_set_value=empty_set_value,
                full_set_value=full_set_value,
            )
        else:
            result = self._approximate_via_odd_regression(
                budget=budget,
                coalitions=coalitions,
                game_values=game_values,
                #centered_game_values=centered_game_values,
                empty_set_value=empty_set_value,
                full_set_value=full_set_value,
                n_candidate_interactions=n_candidate_interactions,
            )

        end_time = time.time()
        self.runtime_last_approximate_run["total"] = end_time - start_time
        return result

    # if budget too small to fit odd regression
    def _approximate_via_fallback(
            self,
            *,
            budget: int,
            coalitions: np.ndarray,
            game_values: np.ndarray,
            #centered_game_values: np.ndarray,
            empty_set_value: float,
            full_set_value: float,
    ) -> InteractionValues:
        """ low-budget OddSHAP fallback branch

        When the budget is too small to stably fit the odd-regression problem, OddSHAP falls back to explaining a fitted tree surrogate.
        (fit a LightGBM surrogate on sampled coalitions and return its first-order Shapley values for the full coalition.

        Args

        """

        # 1. fit tree surrogate on sampled coalitions
        surrogate_start_time = time.time()
        surrogate_model = self._fit_surrogate_model(coalitions=coalitions, game_values=game_values)
        surrogate_end_time = time.time()
        self.runtime_last_approximate_run["proxy_fit"] = surrogate_end_time - surrogate_start_time

        # 2. explain full coalition
        explain_start_time = time.time()
        tree_explainer = TreeExplainer(
            model=surrogate_model,
            max_order=1,
            min_order=0,
            index="SV",
        )
        surrogate_explanation = tree_explainer.explain_function(np.ones(self.n, dtype=float))
        explain_end_time = time.time()
        self.runtime_last_approximate_run["fallback_explain"] = explain_end_time - explain_start_time

        # 3. convert the returned explanation
        sv_values = np.zeros(self.n + 1, dtype=float)
        sv_values[0] = empty_set_value

        for player in range(self.n):
            sv_values[player + 1] = surrogate_explanation[(player,)]

        interaction_lookup = {(): 0}
        for player in range(self.n):
            interaction_lookup[(player,)] = player + 1

        return InteractionValues(
            values=sv_values,
            index="SV",
            max_order=1,
            min_order=0,
            n_players=self.n,
            interaction_lookup=interaction_lookup,
            baseline_value=float(empty_set_value),
            estimated=not budget >= 2 ** self.n,
            estimation_budget=budget,
            target_index="SV",
        )

    def _approximate_via_odd_regression(
            self,
            *,
            budget: int,
            coalitions: np.ndarray,
            game_values: np.ndarray,
            #centered_game_values: np.ndarray,
            empty_set_value: float,
            full_set_value: float,
            n_candidate_interactions: int,
    ) -> InteractionValues:
        """OddSHAP high-budget branch.

        This branch:
        1. fits the tree surrogate
        2. detects higher-order odd interactions
        3. builds the active odd support
        4. constructs the weighted Fourier regression problem
        5. solves the constrained odd regression
        6. transforms the fitted coefficients into Shapley values
        """

        # 1. fit surrogate model
        surrogate_start_time = time.time()
        surrogate_model = self._fit_surrogate_model(coalitions=coalitions, game_values=game_values)
        surrogate_end_time = time.time()
        self.runtime_last_approximate_run["proxy_fit"] = surrogate_end_time - surrogate_start_time

        # 2. detect odd higher-order interactions
        extraction_start_time = time.time()
        detected_interactions = self._select_odd_interactions(
            coalitions=coalitions,
            game_values=game_values,
            n_candidate_interactions=n_candidate_interactions,
            surrogate_model=surrogate_model,
        )
        extraction_end_time = time.time()
        self.runtime_last_approximate_run["extraction"] = extraction_end_time - extraction_start_time

        # 3. build active support
        self._build_support(detected_interactions)

        # 4. build weighted regression objects
        regression_start_time = time.time()
        X_tilde, y_tilde = self._build_weighted_system(
            coalitions=coalitions,
            game_values=game_values,
            empty_set_value=empty_set_value,
            full_set_value=full_set_value,
            drop_boundary_rows=True,
        )

        # 5. solve constrained odd Fourier regression
        odd_fourier_coefficients = self._solve_constrained_regression(
            X_tilde=X_tilde,
            y_tilde=y_tilde,
            empty_set_value=empty_set_value,
            full_set_value=full_set_value,
        )

        # 6. transform coefficients into Shapley values
        sv_values = self._transform_to_shapley(
            odd_fourier_coefficients,
            baseline_value=empty_set_value,
        )

        regression_end_time = time.time()
        self.runtime_last_approximate_run["regression"] = regression_end_time - regression_start_time

        return InteractionValues(
            values=sv_values,
            index="SV",
            max_order=1,
            min_order=0,
            n_players=self.n,
            baseline_value=float(empty_set_value),
            estimated=not budget >= 2 ** self.n,
            estimation_budget=budget,
            target_index="SV",
        )

    # TODO: adjust if necessary later on (Wu)
    def _fit_surrogate_model(
            self,
            coalitions: np.ndarray,
            game_values: np.ndarray,
    ) -> lgb.LGBMRegressor:
        """ Fit the LightGBM surrogate used by the OddSHAP fallback branch.

        Args

        """
        if self.tree_params is None:
            surrogate_model = lgb.LGBMRegressor(
                verbose=-1,
                n_jobs=1,
                random_state=self._random_state,
                max_depth=10,
            )
        else:
            surrogate_model = lgb.LGBMRegressor(
                verbose=-1,
                n_jobs=1,
                random_state=self._random_state,
                **self.tree_params,
            )

        surrogate_model.fit(coalitions.astype(float), game_values)
        return surrogate_model

    def _build_support(
            self,
            selected_interactions: list[tuple[int, ...]] | None,
    ) -> None:
        """Build the active OddSHAP support.

        The support always contains:
        - the empty interaction ()
        - all singleton interactions (i,)
        - selected higher-order odd interactions

        The ordering is deterministic:
        1. ()
        2. (0,), (1,), ..., (n-1,)
        3. sorted selected higher-order odd interactions
        """
        if selected_interactions is None:
            normalized_selected: list[tuple[int, ...]] = []
        else:
            normalized_set: set[tuple[int, ...]] = set()

            for interaction in selected_interactions:
                normalized = tuple(sorted(interaction))

                # skip empty or singleton terms because OddSHAP always includes them
                if len(normalized) <= 1:
                    continue

                # keep only odd-sized higher-order interactions
                if len(normalized) % 2 == 0:
                    continue

                normalized_set.add(normalized)

            normalized_selected = sorted(normalized_set)

        active_interactions: list[tuple[int, ...]] = [()]
        active_interactions.extend((player,) for player in range(self.n))
        active_interactions.extend(normalized_selected)

        self.odd_interaction_lookup = {
            interaction: position for position, interaction in enumerate(active_interactions)
        }

        interaction_matrix_binary = np.zeros((len(active_interactions), self.n), dtype=bool)

        for interaction, position in self.odd_interaction_lookup.items():
            if interaction:
                interaction_matrix_binary[position, list(interaction)] = True

        self.odd_interaction_matrix_binary = interaction_matrix_binary
        self.n_active_interactions = len(active_interactions)

    def _select_odd_interactions(
            self,
            *,
            coalitions: np.ndarray,
            game_values: np.ndarray,
            n_candidate_interactions: int,
            surrogate_model: Any | None = None,
    ) -> list[tuple[int, ...]]:
        """Return selected higher-order odd interactions for the OddSHAP support.

        Uses ProxySPEX-style screening: convert the LightGBM surrogate into its
        sparse Fourier representation, then keep the top-k higher-order
        odd-cardinality interactions by coefficient magnitude. Singletons and
        the empty interaction are excluded here because _build_support adds
        them unconditionally.
        """
        del coalitions, game_values  # already encoded in the fitted surrogate

        if n_candidate_interactions <= 0 or surrogate_model is None:
            return []

        fourier_coeffs = lgboost_to_fourier(surrogate_model.booster_.dump_model())
        higher_order_odd = {
            interaction: coefficient
            for interaction, coefficient in fourier_coeffs.items()
            if len(interaction) >= 3 and len(interaction) % 2 == 1
        }
        selected = top_k_interactions(
            higher_order_odd, k=n_candidate_interactions, odd=False,
        )
        return list(selected.keys())


    def _get_regression_row_weights(self) -> np.ndarray:
        """ Return square-root row weights for the OddSHAP regression problem

        The weights combine:
        - OddSHAP coalition-size kernel weights
        - sampling adjustment weights from the CoalitionSampler

        """
        kernel_weights = self._init_sampling_weights_static(self.n)
        coalition_sizes = self._sampler.coalitions_size
        sampling_adjustment_weights = self._sampler.sampling_adjustment_weights

        regression_weights = (
                kernel_weights[coalition_sizes] * sampling_adjustment_weights
        )
        return np.sqrt(regression_weights)

    def _build_weighted_system(
            self,
            *,
            coalitions: np.ndarray,
            game_values: np.ndarray,
            empty_set_value: float,
            full_set_value: float,
            drop_boundary_rows: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Build the weighted OddSHAP regression system (X_tilde, y_tilde).

        The regression is solved for the non-empty active Fourier coefficients,
        while the empty Fourier coefficient is fixed exactly as

            beta_empty = (f(full) + f(empty)) / 2

        following Appendix C of the OddSHAP paper.
        """
        if self.n_active_interactions == 0:
            msg = (
                "OddSHAP support has not been built yet. Call _build_support(...) first."
            )
            raise RuntimeError(msg)

        row_weights = self._get_regression_row_weights()
        beta_empty = 0.5 * (full_set_value + empty_set_value)

        if drop_boundary_rows:
            coalition_sizes = np.sum(coalitions, axis=1)
            inner_row_mask = (coalition_sizes > 0) & (coalition_sizes < self.n)

            coalitions_used = coalitions[inner_row_mask]
            row_weights_used = row_weights[inner_row_mask]
            centered_values = (game_values - beta_empty)[inner_row_mask]
        else:
            coalitions_used = coalitions
            row_weights_used = row_weights
            centered_values = game_values - beta_empty

        # Drop the empty interaction column because beta_empty is fixed exactly
        interaction_masks = self.odd_interaction_matrix_binary[1:, :]

        coalitions_int = coalitions_used.astype(np.uint8)
        interaction_masks_int = interaction_masks.astype(np.uint8)

        parity_matrix = (coalitions_int @ interaction_masks_int.T) % 2
        design_matrix = 1 - 2 * parity_matrix
        X_tilde = design_matrix.astype(float) * row_weights_used[:, np.newaxis]
        y_tilde = centered_values * row_weights_used

        return X_tilde, y_tilde

    def _build_constraint_system(
            self,
            *,
            full_set_value: float,
            empty_set_value: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """ Build the exact OddSHAP Fourier constraint system.

        Returns:
            beta_empty = (f(full) + f(empty)) / 2
                Exact Fourier coefficient for the empty interaction.
            projection_matrix:
                Projection onto the orthogonal complement of the all-ones vector.
            beta_const:
                A feasible point satisfying the odd-coefficient sum constraint.
        """
        n_nonempty_terms = self.n_active_interactions - 1
        if n_nonempty_terms <= 0:
            msg = (
                "OddSHAP support must contain at least one non-empty interaction before building constraint objects."
            )
            raise RuntimeError(msg)

        beta_empty = 0.5 * (full_set_value + empty_set_value)

        # non-empty odd coefficients
        b = -0.5 * (full_set_value - empty_set_value)

        a = np.ones(n_nonempty_terms, dtype=float)
        denominator = float(a @ a)

        projection_matrix = np.eye(n_nonempty_terms, dtype=float) - np.outer(a, a) / denominator
        beta_const = (b / denominator) * a

        return beta_empty, projection_matrix, beta_const

    def _solve_constrained_regression(
            self,
            *,
            X_tilde: np.ndarray,
            y_tilde: np.ndarray,
            empty_set_value: float,
            full_set_value: float,
    ) -> np.ndarray:
        """ Solve the constrained OddSHAP regression in the Fourier basis.

        This solves the non-empty odd coefficients under the exact sum constraint
        from the OddSHAP paper, then assembles the full coefficient vector
        including the empty interaction coefficient.
        """
        beta_empty, projection_matrix, beta_const = self._build_constraint_system(
            full_set_value=full_set_value,
            empty_set_value=empty_set_value,
        )

        # Project the constrained problem into an unconstrained least-squares problem
        X_projected = X_tilde @ projection_matrix
        y_projected = y_tilde - X_tilde @ beta_const

        # Solve for the free projected coordinates
        z_solution = np.linalg.lstsq(X_projected, y_projected, rcond=None)[0]

        # Reconstruct the constrained non-empty coefficient vector
        beta_nonempty = beta_const + projection_matrix @ z_solution

        # Assemble the full coefficient vector including the empty interaction
        beta_full = np.zeros(self.n_active_interactions, dtype=float)
        beta_full[0] = beta_empty
        beta_full[1:] = beta_nonempty

        return beta_full

    def _transform_to_shapley(
            self,
            odd_fourier_coefficients: np.ndarray,
            *,
            baseline_value: float,
    ) -> np.ndarray:
        """ Transform fitted odd Fourier coefficients into first-order Shapley values.

        The coefficient vector must follow the active support ordering:
        - position 0 corresponds to the empty interaction ()
        - positions 1.. correspond to the active non-empty interactions

        For OddSHAP, only odd-cardinality interactions contribute to the Shapley
        value. For each odd interaction T with coefficient beta_T, the contribution
        is split equally across players in T, and the final Shapley values are
        scaled by -2:

            phi_i = -2 * sum_{T: i in T, |T| odd} beta_T / |T|

        Args

        """
        if odd_fourier_coefficients.shape[0] != self.n_active_interactions:
            msg = (
                "Coefficient vector length does not match the active OddSHAP support. "
                f"Expected {self.n_active_interactions}, got {odd_fourier_coefficients.shape[0]}."
            )
            raise ValueError(msg)

        sv_values = np.zeros(self.n + 1, dtype=float)
        sv_values[0] = baseline_value

        for interaction, position in self.odd_interaction_lookup.items():
            # The empty interaction does not contribute to player attributions
            if len(interaction) == 0:
                continue

            # OddSHAP only uses odd-cardinality Fourier terms for the attribution
            if len(interaction) % 2 == 0:
                continue

            coefficient = odd_fourier_coefficients[position]
            share = coefficient / len(interaction)

            for player in interaction:
                sv_values[player + 1] += share

        # Global Fourier-to-Shapley scaling from the OddSHAP paper
        sv_values[1:] *= -2.0

        return sv_values