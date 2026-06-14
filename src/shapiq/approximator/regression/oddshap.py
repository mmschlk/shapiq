"""This module contains the OddSHAP approximator for estimating Shapley values.

OddSHAP is a value estimator based on paired sampling, odd-only Fourier regression, and sparse odd interaction detection as introduced in Fumagalli et al. (2026) :cite:t:`Fumagalli.2026`
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import binom
from sklearn.tree import DecisionTreeRegressor

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues
from shapiq.tree.conversion import convert_tree_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.game import Game
    from shapiq.tree.base import TreeModel


def _resolve_surrogate_model(
    random_state: int | None,
    tree_params: dict[str, Any] | None,
) -> object:
    """Build the surrogate tree model for OddSHAP's Fourier screening.

    Tries LightGBM first (the paper's configuration), then falls back to
    a DecisionTreeRegressor with a warning — following the same resolution
    pattern as ProxySHAP/ProxySPEX (``_models._select_base_proxy_via_string``).
    """
    params: dict[str, Any] = {
        "verbose": -1,
        "n_jobs": 1,
        "random_state": random_state,
        "max_depth": 10,
    }
    params.update(tree_params or {})

    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**params)
    except ImportError:
        pass
    dt_keys = set(DecisionTreeRegressor().get_params())
    dt_params = {k: v for k, v in params.items() if k in dt_keys}
    user_dropped = sorted(set(tree_params or {}) - dt_keys)
    msg = (
        "LightGBM is not installed. OddSHAP will use a DecisionTreeRegressor "
        "as the surrogate. For best results install LightGBM: "
        "pip install 'shapiq[proxy]'."
    )
    if user_dropped:
        msg += (
            f" The following tree_params were dropped (not supported by "
            f"DecisionTreeRegressor): {user_dropped}."
        )
    warnings.warn(msg, stacklevel=2)
    return DecisionTreeRegressor(**dt_params)


_FourierDict = dict[tuple[int, ...], float]


def _tree_to_fourier(tree_model: TreeModel) -> _FourierDict:
    """Extract exact Fourier (Walsh) coefficients from a single tree by DFS.

    Equivalent to ``ProxySPEX._sklearn_tree_to_fourier`` but standalone — no
    Approximator instantiation required.
    """

    def _combine(
        left: _FourierDict, right: _FourierDict, feature: int
    ) -> _FourierDict:
        combined: _FourierDict = {}
        for interaction in set(left) | set(right):
            left_val = left.get(interaction, 0.0)
            right_val = right.get(interaction, 0.0)
            combined[interaction] = (left_val + right_val) / 2
            combined[tuple(sorted(set(interaction) | {feature}))] = (left_val - right_val) / 2
        return combined

    def _dfs(node: int) -> _FourierDict:
        if tree_model.children_left[node] == -1:
            return {(): tree_model.values[node]}
        return _combine(
            _dfs(tree_model.children_left[node]),
            _dfs(tree_model.children_right[node]),
            tree_model.features[node],
        )

    return _dfs(0)


def _ensemble_to_fourier(tree_models: list[TreeModel]) -> dict[tuple[int, ...], float]:
    """Aggregate Fourier coefficients across an ensemble of trees."""
    aggregated: dict[tuple[int, ...], float] = defaultdict(float)
    for tree_model in tree_models:
        for interaction, value in _tree_to_fourier(tree_model).items():
            aggregated[interaction] += value
    return {k: v for k, v in aggregated.items() if v != 0.0}


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
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
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
            pairing_trick=True,
            sampling_weights=sampling_weights,
            random_state=random_state,
        )

        if interaction_factor < 1:
            msg = "interaction_factor (eta) must be at least 1."
            raise ValueError(msg)
        self.interaction_factor = interaction_factor
        self.tree_params = tree_params
        self._surrogate_template = _resolve_surrogate_model(self._random_state, tree_params)
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

        # Higher-order odd support size |T_odd| = ceil(m / eta) - d (paper Alg. 1,
        # p. 6): the total regression variable count is ceil(m/eta), of which d are
        # singletons (always included), so only the remainder are screened from the
        # surrogate's Fourier spectrum.  At full budget all coalitions are enumerated,
        # so the candidate support is not truncated.
        if budget >= 2**self.n:
            n_candidate_interactions = 2**self.n
        else:
            n_candidate_interactions = max(0, math.ceil(budget / self.interaction_factor) - self.n)

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
            index=self.index,
            max_order=1,
            min_order=0,
            n_players=self.n,
            interaction_lookup=self.interaction_lookup,
            baseline_value=float(empty_set_value),
            estimated=not (budget >= 2**self.n),
            estimation_budget=budget,
        )

    def _fit_surrogate_model(
        self,
        coalitions: np.ndarray,
        game_values: np.ndarray,
    ) -> object:
        """Fit the surrogate tree used for sparse odd-interaction detection."""
        from sklearn.base import clone

        surrogate_model = clone(self._surrogate_template)
        surrogate_model.fit(coalitions, game_values)
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
        Algorithm 1 / "Controlling Higher-Order Terms"): the fitted GBT is
        converted to its exact Fourier (Walsh) spectrum and the odd-cardinality
        frequencies with the largest coefficient magnitudes are kept.
        """
        if surrogate_model is None or n_candidate_interactions <= 0:
            return []

        tree_models = convert_tree_model(surrogate_model)
        if not isinstance(tree_models, list):
            tree_models = [tree_models]
        fourier_coefficients = _ensemble_to_fourier(tree_models)

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
        X_tilde = design_matrix * row_weights_used[:, np.newaxis]
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

        # The active support contains only () and odd-cardinality interactions
        # (singletons + detected higher-order odds), so the membership matrix
        # already encodes the correct containment and sizes.
        interaction_sizes = self.odd_interaction_matrix_binary.sum(axis=1).astype(float)  # (K,)
        # position 0 is () with size 0 — skip it; singletons have size 1
        coeffs = odd_fourier_coefficients[1:]
        sizes = interaction_sizes[1:]
        masks = self.odd_interaction_matrix_binary[1:]  # (K-1, n) bool

        # phi_i = -2 * sum_{T: i in T} beta_T / |T|
        shares = coeffs / sizes  # (K-1,)
        sv_values[1:] = -2.0 * (masks.T @ shares)  # (n,)

        return sv_values
