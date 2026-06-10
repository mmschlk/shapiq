"""This module contains the OddSHAP approximator for estimating Shapley values.

OddSHAP is a value estimator based on paired sampling, odd-only Fourier regression, and sparse odd interaction detection as introduced in Fumagalli et al. (2026) :cite:t:`Fumagalli.2026`
"""

from __future__ import annotations

import math
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import binom

from shapiq.approximator.base import Approximator
from shapiq.approximator.proxy.proxyspex import ProxySPEX
from shapiq.interaction_values import InteractionValues
from shapiq.tree.conversion import convert_tree_model

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from shapiq.game import Game


def _import_lightgbm() -> ModuleType:
    try:
        lightgbm = import_module("lightgbm")
    except ImportError as err:
        msg = (
            "The 'lightgbm' package is required for OddSHAP but it is not installed. "
            "Install it via the optional extra: pip install 'shapiq[proxy]'."
        )
        raise ImportError(msg) from err
    return lightgbm


class OddSHAP(Approximator):
    """OddSHAP approximator for first-order Shapley values (Fumagalli et al., 2026).

    Note:
        Where Algorithm 1 of the paper falls back to TreeSHAP for budgets below
        ``n * interaction_factor``, this implementation raises ``ValueError`` instead
        (no silent downgrade to another estimator), unless the budget already covers
        the full coalition space (``budget >= 2**n``). It therefore does not reproduce
        the low-budget, high-dimension regime of the paper's Figure 2.
    """

    valid_indices: tuple[str, ...] = ("SV",)

    @staticmethod
    def _init_sampling_weights_static(n: int) -> np.ndarray:
        """Initialize OddSHAP coalition-size sampling weights.

        OddSHAP samples uniformly over non-boundary coalition sizes.
        Empty and full coalitions are handled separately by the sampler.
        """
        if n <= 1:
            msg = "OddSHAP sampling weights are undefined for n <= 1."
            raise ValueError(msg)

        weight_vector = np.zeros(n + 1, dtype=float)
        weight_vector[1:n] = 1.0
        return weight_vector / np.sum(weight_vector)

    @staticmethod
    def _init_regression_kernel_weights_static(n: int) -> np.ndarray:
        """Initialize OddSHAP weighted least-squares kernel weights."""
        if n <= 1:
            msg = "OddSHAP regression weights are undefined for n <= 1."
            raise ValueError(msg)

        weight_vector = np.zeros(n + 1, dtype=float)
        for coalition_size in range(1, n):
            weight_vector[coalition_size] = 1.0 / (
                coalition_size * (n - coalition_size) * binom(n, coalition_size)
            )
        return weight_vector

    def __init__(
        self,
        n: int,
        *,
        pairing_trick: bool = True,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
        odd_only: bool = True,
        interaction_factor: int = 10,  # eta; paper default
        tree_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OddSHAP approximator.

        ``tree_params`` entries override the surrogate defaults — including
        ``random_state``, ``n_jobs``, and ``verbose``; ``max_depth`` defaults to 10
        (the paper's configuration) unless overridden.
        """
        del kwargs
        _import_lightgbm()

        # OddSHAP's own coalition-size distribution; set before super().__init__,
        # which builds the sampler.
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

        if not odd_only:
            msg = "This OddSHAP implementation supports only odd_only=True."
            raise ValueError(msg)
        self.odd_only = True
        if interaction_factor < 1:
            msg = "interaction_factor (eta) must be at least 1."
            raise ValueError(msg)
        self.interaction_factor = interaction_factor
        self.tree_params = tree_params
        # The WLS kernel weights depend only on n; compute them once.
        self._kernel_weights = self._init_regression_kernel_weights_static(n)

        self.odd_interaction_lookup: dict[tuple[int, ...], int] = {}
        self.odd_interaction_matrix_binary = np.zeros((0, self.n), dtype=bool)
        self.n_active_interactions: int = 0

    def approximate(
        self, budget: int, game: Game | Callable[[np.ndarray], np.ndarray], **kwargs: Any
    ) -> InteractionValues:
        """Approximate first-order Shapley values.

        The method samples coalitions, evaluates the game, detects sparse odd
        interactions, solves the constrained odd Fourier regression problem, and
        transforms the fitted coefficients into Shapley values.

        Args:
            budget: Number of game evaluations available to the approximator.
            game: Game or callable that evaluates coalition matrices.
            **kwargs: Additional keyword arguments kept for API compatibility.

        Returns:
            Estimated first-order Shapley values.

        Raises:
            ValueError: If ``budget < min(n * interaction_factor, 2**n)``, i.e. the
                budget is below the eta-based minimum and does not cover the full
                coalition space either. Algorithm 1 of the paper falls back to TreeSHAP
                in this regime; this implementation deliberately raises instead, so an
                under-budgeted call never silently returns a different estimator's
                values.
            RuntimeError: If the sampled coalitions do not contain the empty or grand coalition.
        """
        del kwargs

        # Fail fast before any (possibly expensive) game evaluation. A budget that
        # covers the full coalition space is always sufficient, even when 2**n is
        # smaller than the eta-based minimum (small n).
        minimum_budget = min(self.n * self.interaction_factor, 2**self.n)
        if budget < minimum_budget:
            msg = (
                "The budget is too small for OddSHAP. "
                f"Received budget={budget}, but at least {minimum_budget} evaluations are required. "
                "Please increase the budget."
            )
            raise ValueError(msg)

        self._sampler.sample(budget)
        coalitions = self._sampler.coalitions_matrix
        game_values = np.asarray(game(coalitions), dtype=float)

        # CoalitionSampler guarantees the empty and grand coalitions are present.
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

        # Candidate higher-order support size |T_odd| = ceil(m / eta) (paper Alg. 1);
        # at full budget all coalitions are enumerated, so the candidate support is
        # not truncated.
        if budget >= 2**self.n:
            n_candidate_interactions = 2**self.n
        else:
            n_candidate_interactions = max(0, math.ceil(budget / self.interaction_factor))

        return self._approximate_via_odd_regression(
            budget=budget,
            coalitions=coalitions,
            game_values=game_values,
            empty_set_value=empty_set_value,
            full_set_value=full_set_value,
            n_candidate_interactions=n_candidate_interactions,
        )

    def _approximate_via_odd_regression(
        self,
        *,
        budget: int,
        coalitions: np.ndarray,
        game_values: np.ndarray,
        empty_set_value: float,
        full_set_value: float,
        n_candidate_interactions: int,
    ) -> InteractionValues:
        """Run the OddSHAP odd-regression approximation.

        This branch:
        1. fits the tree surrogate
        2. detects higher-order odd interactions
        3. builds the active odd support
        4. constructs the weighted Fourier regression problem
        5. solves the constrained odd regression
        6. transforms the fitted coefficients into Shapley values
        """
        surrogate_model = self._fit_surrogate_model(coalitions=coalitions, game_values=game_values)
        detected_interactions = self._select_odd_interactions(
            n_candidate_interactions=n_candidate_interactions,
            surrogate_model=surrogate_model,
        )
        self._build_support(detected_interactions)

        X_tilde, y_tilde = self._build_weighted_system(
            coalitions=coalitions,
            game_values=game_values,
            empty_set_value=empty_set_value,
            full_set_value=full_set_value,
            drop_boundary_rows=True,
        )
        odd_fourier_coefficients = self._solve_constrained_regression(
            X_tilde=X_tilde,
            y_tilde=y_tilde,
            empty_set_value=empty_set_value,
            full_set_value=full_set_value,
        )
        sv_values = self._transform_to_shapley(
            odd_fourier_coefficients,
            baseline_value=empty_set_value,
        )

        return InteractionValues(
            values=sv_values,
            index=self.approximation_index,
            max_order=1,
            min_order=0,
            n_players=self.n,
            # the base class populated {(): 0, (0,): 1, ..., (n-1,): n} for
            # min_order=0, max_order=1
            interaction_lookup=self.interaction_lookup,
            baseline_value=float(empty_set_value),
            estimated=not (budget >= 2**self.n),
            estimation_budget=budget,
            target_index=self.index,
        )

    def _fit_surrogate_model(
        self,
        coalitions: np.ndarray,
        game_values: np.ndarray,
    ) -> object:
        """Fit the LightGBM surrogate used for sparse odd-interaction detection."""
        lgb = _import_lightgbm()

        # Paper defaults (depth-10 surrogate); any user-supplied tree_params entry
        # overrides the matching default.
        params = {
            "verbose": -1,
            "n_jobs": 1,
            "random_state": self._random_state,
            "max_depth": 10,
            **(self.tree_params or {}),
        }
        surrogate_model = lgb.LGBMRegressor(**params)
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
        n_candidate_interactions: int,
        surrogate_model: object | None = None,
    ) -> list[tuple[int, ...]]:
        """Screen higher-order odd interactions from the surrogate's Fourier spectrum.

        Implements the paper's ``OddInteractionExtract`` (Fumagalli et al. 2026,
        Algorithm 1 / "Controlling Higher-Order Terms"): following ProxySPEX, the fitted
        GBT is converted to its exact Fourier (Walsh) spectrum and the odd-cardinality
        frequencies with the largest coefficient magnitudes are kept. The Fourier
        extraction is reused directly from ProxySPEX (``_sklearn_to_fourier``) rather than
        re-implemented, so the support is selected in the very Fourier basis the odd
        regression is solved in.
        """
        if surrogate_model is None or n_candidate_interactions <= 0:
            return []

        tree_models = convert_tree_model(surrogate_model)
        # The ProxySPEX instance is only a host for its exact GBT->Fourier extractor;
        # it is never sampled or fit.
        # max_order=1 keeps the base class from building an order-2 interaction
        # lookup that the extractor never uses.
        fourier_extractor = ProxySPEX(
            n=self.n, max_order=1, proxy_model=surrogate_model, random_state=self._random_state
        )
        fourier_coefficients = fourier_extractor._sklearn_to_fourier(tree_models)  # noqa: SLF001

        higher_order_odd: dict[tuple[int, ...], float] = {}
        for interaction, coefficient in fourier_coefficients.items():
            normalized_interaction = tuple(sorted(interaction))
            if len(normalized_interaction) >= 3 and len(normalized_interaction) % 2 == 1:
                higher_order_odd[normalized_interaction] = float(coefficient)

        selected = sorted(
            higher_order_odd.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:n_candidate_interactions]

        return [interaction for interaction, _ in selected]

    def _get_regression_row_weights(self) -> np.ndarray:
        """Return square-root row weights for the OddSHAP regression problem.

        The weights combine:
        - OddSHAP coalition-size kernel weights
        - sampling adjustment weights from the CoalitionSampler

        """
        kernel_weights = self._kernel_weights
        coalition_sizes = self._sampler.coalitions_size
        sampling_adjustment_weights = self._sampler.sampling_adjustment_weights

        regression_weights = kernel_weights[coalition_sizes] * sampling_adjustment_weights
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
        """Build the weighted OddSHAP regression system (X_tilde, y_tilde).

        The regression is solved for the non-empty active Fourier coefficients,
        while the empty Fourier coefficient is fixed exactly as

            beta_empty = (f(full) + f(empty)) / 2

        following Appendix C of the OddSHAP paper.
        """
        if self.n_active_interactions == 0:
            msg = "OddSHAP support has not been built yet. Call _build_support(...) first."
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
        # Cast before the Fourier sign transform: uint8 would underflow
        # 1 - 2 * 1 to 255 instead of -1.
        design_matrix = 1.0 - 2.0 * parity_matrix.astype(float)
        X_tilde = design_matrix.astype(float) * row_weights_used[:, np.newaxis]
        y_tilde = centered_values * row_weights_used

        return X_tilde, y_tilde

    def _build_constraint_system(
        self,
        *,
        full_set_value: float,
        empty_set_value: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Build the exact OddSHAP Fourier constraint system.

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
            msg = "OddSHAP support must contain at least one non-empty interaction before building constraint objects."
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
        """Solve the constrained OddSHAP regression in the Fourier basis.

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
        """Transform fitted odd Fourier coefficients into first-order Shapley values.

        The coefficient vector must follow the active support ordering:
        - position 0 corresponds to the empty interaction ()
        - positions 1.. correspond to the active non-empty interactions

        For OddSHAP, only odd-cardinality interactions contribute to the Shapley
        value. For each odd interaction T with coefficient beta_T, the contribution
        is split equally across players in T, and the final Shapley values are
        scaled by -2:

            phi_i = -2 * sum_{T: i in T, |T| odd} beta_T / |T|

        Args:
            odd_fourier_coefficients: Fitted odd Fourier coefficients in active support order.
            baseline_value: Baseline value assigned to the empty coalition.
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
