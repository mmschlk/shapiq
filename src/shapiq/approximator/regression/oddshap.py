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

    def _combine(left: _FourierDict, right: _FourierDict, feature: int) -> _FourierDict:
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
        ``n * interaction_factor``, this implementation expands the selection of
        active terms also to individuals, allowing a minimum budget of
        ``interaction_factor``. Below that, it raises ``ValueError``
        (no silent downgrade to another estimator), unless the budget already covers
        the full coalition space (``budget >= 2**n``). It therefore does not reproduce
        the low-budget, high-dimension regime of the paper's Figure 2.

        The active support's candidate budget (``ceil(budget / interaction_factor)``)
        is shared between individuals and higher-order odd interactions: the most
        relevant individuals (ranked by absolute Fourier coefficient, all ``n`` of
        them always considered) are screened first, and only the remaining candidate
        budget is spent on higher-order odd interactions. Unlike the paper, low
        budgets can therefore leave some individuals out of the active support
        entirely (their Shapley value is then estimated as exactly 0).
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
        self.odd_interaction_matrix_float = np.zeros((0, self.n), dtype=np.float64)
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
            ValueError: If ``budget < min(interaction_factor, 2**n)``, i.e. the
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
        minimum_budget = min(self.interaction_factor, 2**self.n)
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

        # Total active-support candidate budget = ceil(m / eta) (paper Alg. 1, p. 6).
        # Individuals and higher-order odd interactions draw from this shared budget:
        # individuals are screened first (see `_select_active_terms`), so only the
        # remainder goes to higher-order odd interactions. At full budget all
        # coalitions are enumerated, so the candidate support is not truncated.
        if budget >= 2**self.n:
            n_candidate_terms = 2**self.n
        else:
            n_candidate_terms = max(0, math.ceil(budget / self.interaction_factor))

        return self._approximate_via_odd_regression(
            budget=budget,
            coalitions=coalitions,
            game_values=game_values,
            empty_set_value=empty_set_value,
            full_set_value=full_set_value,
            n_candidate_terms=n_candidate_terms,
        )

    def _approximate_via_odd_regression(
        self,
        *,
        budget: int,
        coalitions: np.ndarray,
        game_values: np.ndarray,
        empty_set_value: float,
        full_set_value: float,
        n_candidate_terms: int,
    ) -> InteractionValues:
        """Run the OddSHAP odd-regression approximation.

        This branch:
        1. fits the tree surrogate
        2. selects the active odd support (individuals first, then higher-order
           odd interactions)
        3. builds the active odd support
        4. constructs the weighted Fourier regression problem
        5. solves the constrained odd regression
        6. transforms the fitted coefficients into Shapley values
        """
        surrogate_model = self._fit_surrogate_model(coalitions=coalitions, game_values=game_values)
        selected_individuals, selected_interactions = self._select_active_terms(
            n_candidate_terms=n_candidate_terms,
            surrogate_model=surrogate_model,
        )
        self._build_support(selected_individuals + selected_interactions)

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
        selected_terms: list[tuple[int, ...]] | None,
    ) -> None:
        """Build the active OddSHAP support from a list of selected terms.

        The support always contains the empty interaction ``()`` plus the given
        terms (deduplicated and sorted by size then value). Unlike in Fumagalli et al. (2026),
        singletons are *not* auto-injected here: relevance-based
        individual selection now happens upstream in ``_select_active_terms``, so
        only individuals that were actually selected end up in the support.
        """
        if selected_terms is None:
            normalized_selected: list[tuple[int, ...]] = []
        else:
            normalized_set = {tuple(sorted(term)) for term in selected_terms if len(term) >= 1}
            normalized_selected = sorted(normalized_set, key=lambda term: (len(term), term))

        active_interactions: list[tuple[int, ...]] = [(), *normalized_selected]

        self.odd_interaction_lookup = {
            interaction: position for position, interaction in enumerate(active_interactions)
        }

        interaction_matrix_binary = np.zeros((len(active_interactions), self.n), dtype=bool)

        for interaction, position in self.odd_interaction_lookup.items():
            if interaction:
                interaction_matrix_binary[position, list(interaction)] = True

        self.odd_interaction_matrix_binary = interaction_matrix_binary
        self.odd_interaction_matrix_float = interaction_matrix_binary.astype(np.float64)
        self.n_active_interactions = len(active_interactions)

    @staticmethod
    def _surrogate_to_fourier(surrogate_model: object) -> _FourierDict:
        """Convert the fitted surrogate to its exact Fourier (Walsh) spectrum."""
        tree_models = convert_tree_model(surrogate_model)
        if not isinstance(tree_models, list):
            tree_models = [tree_models]
        return _ensemble_to_fourier(tree_models)

    def _select_active_terms(
        self,
        *,
        n_candidate_terms: int,
        surrogate_model: object | None,
    ) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        """Split the shared candidate budget between individuals and interactions.

        Implements the paper's "Controlling Higher-Order Terms" priority
        (Fumagalli et al. 2026, Algorithm 1): the most relevant individuals are
        screened first from the surrogate's Fourier spectrum (all ``n`` of them
        are always ranked, whether or not the surrogate happened to split on
        them), and only the remaining candidate budget is spent screening
        higher-order odd interactions via ``_select_odd_interactions``.
        """
        if surrogate_model is None or n_candidate_terms <= 0:
            return [], []

        # Convert the surrogate to its Fourier spectrum once and reuse it for both
        # screening steps; individuals and higher-order odds previously each
        # recomputed the full surrogate -> Fourier transform.
        fourier_coefficients = self._surrogate_to_fourier(surrogate_model)

        n_individual_terms = min(self.n, n_candidate_terms)
        selected_individuals = self._select_individual_terms(
            n_individual_terms=n_individual_terms,
            surrogate_model=surrogate_model,
            fourier_coefficients=fourier_coefficients,
        )

        remaining_budget = n_candidate_terms - len(selected_individuals)
        selected_interactions = self._select_odd_interactions(
            n_candidate_interactions=remaining_budget,
            surrogate_model=surrogate_model,
            fourier_coefficients=fourier_coefficients,
        )

        return selected_individuals, selected_interactions

    def _select_individual_terms(
        self,
        *,
        n_individual_terms: int,
        surrogate_model: object | None = None,
        fourier_coefficients: _FourierDict | None = None,
    ) -> list[tuple[int, ...]]:
        """Rank all individuals by absolute Fourier coefficient and keep the top ones.

        Every player is ranked, including players the surrogate never split on
        (their coefficient defaults to 0.0 and they rank last), so a scarce
        budget always prefers players the surrogate found relevant over
        arbitrary index order.

        ``fourier_coefficients`` may be passed in to reuse an already-computed
        spectrum; when omitted it is derived from ``surrogate_model``.
        """
        if surrogate_model is None or n_individual_terms <= 0:
            return []

        if fourier_coefficients is None:
            fourier_coefficients = self._surrogate_to_fourier(surrogate_model)
        singleton_coefficients = {
            player: float(fourier_coefficients.get((player,), 0.0)) for player in range(self.n)
        }
        ranked_players = sorted(
            singleton_coefficients,
            key=lambda player: (-abs(singleton_coefficients[player]), player),
        )

        return [(player,) for player in ranked_players[:n_individual_terms]]

    def _select_odd_interactions(
        self,
        *,
        n_candidate_interactions: int,
        surrogate_model: object | None = None,
        fourier_coefficients: _FourierDict | None = None,
    ) -> list[tuple[int, ...]]:
        """Screen higher-order odd interactions from the surrogate's Fourier spectrum.

        Implements the paper's ``OddInteractionExtract`` (Fumagalli et al. 2026,
        Algorithm 1 / "Controlling Higher-Order Terms"): the fitted GBT is
        converted to its exact Fourier (Walsh) spectrum and the odd-cardinality
        frequencies with the largest coefficient magnitudes are kept.

        ``fourier_coefficients`` may be passed in to reuse an already-computed
        spectrum; when omitted it is derived from ``surrogate_model``.
        """
        if surrogate_model is None or n_candidate_interactions <= 0:
            return []

        if fourier_coefficients is None:
            fourier_coefficients = self._surrogate_to_fourier(surrogate_model)

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
        """Return raw (non-square-root) row weights for the OddSHAP regression problem.

        The weights combine:
        - OddSHAP coalition-size kernel weights
        - sampling adjustment weights from the CoalitionSampler

        Kept un-square-rooted because ``_pair_interior_rows`` combines pairs of
        rows by doubling their raw weight; the square root is taken once, after
        pairing, in ``_build_weighted_system``.
        """
        kernel_weights = self._kernel_weights
        coalition_sizes = self._sampler.coalitions_size
        sampling_adjustment_weights = self._sampler.sampling_adjustment_weights

        return kernel_weights[coalition_sizes] * sampling_adjustment_weights

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

        When ``drop_boundary_rows`` is set (the only path used by
        ``approximate``), complementary coalition pairs (S, S^c) sampled by the
        pairing trick are collapsed into a single combined row each (see
        ``_pair_interior_rows``), halving the number of regression rows without
        changing the least-squares solution.
        """
        if self.n_active_interactions == 0:
            msg = "OddSHAP support has not been built yet. Call _build_support(...) first."
            raise RuntimeError(msg)

        row_weights = self._get_regression_row_weights()
        beta_empty = 0.5 * (full_set_value + empty_set_value)

        if drop_boundary_rows:
            coalition_sizes = np.sum(coalitions, axis=1)
            interior_mask = (coalition_sizes > 0) & (coalition_sizes < self.n)
            coalitions_used, weights_used, targets_used = self._pair_interior_rows(
                coalitions=coalitions,
                game_values=game_values,
                row_weights=row_weights,
                interior_mask=interior_mask,
                beta_empty=beta_empty,
            )
        else:
            coalitions_used = coalitions
            weights_used = row_weights
            targets_used = game_values - beta_empty

        # Drop the empty interaction column because beta_empty is fixed exactly
        interaction_masks = self.odd_interaction_matrix_float[1:, :]
        parity_matrix = np.remainder(coalitions_used.astype(np.float64) @ interaction_masks.T, 2.0)
        design_matrix = 1.0 - 2.0 * parity_matrix
        row_weights_sqrt = np.sqrt(weights_used)
        X_tilde = design_matrix * row_weights_sqrt[:, np.newaxis]
        y_tilde = targets_used * row_weights_sqrt

        return X_tilde, y_tilde

    def _pair_interior_rows(
        self,
        *,
        coalitions: np.ndarray,
        game_values: np.ndarray,
        row_weights: np.ndarray,
        interior_mask: np.ndarray,
        beta_empty: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collapse complementary coalition pairs into single combined rows.

        For every active support column T, |T| is odd (singletons and
        higher-order odd interactions only), so chi_T(S^c) = -chi_T(S). Paired
        coalitions also share identical kernel*sampling weight (the pairing
        trick guarantees the complement is sampled, and OddSHAP's symmetric
        kernel/sampling weights make w(S) == w(S^c)). Under weighted least
        squares, replacing the two rows

            sqrt(w) * chi(S)  ~  sqrt(w) * (f(S)  - beta_empty)
            sqrt(w) * chi(S^c) ~ sqrt(w) * (f(S^c) - beta_empty)

        by the single row

            sqrt(2w) * chi(S)  ~  sqrt(2w) * (f(S) - f(S^c)) / 2

        changes the weighted sum-of-squares objective only by a
        beta-independent constant, so the least-squares solution is identical
        while using half the rows. Coalitions without a sampled complement (a
        rare edge case when the sampling budget is exhausted mid-pair) keep
        their original, beta_empty-centered single-row equation.
        """
        interior_idx = np.where(interior_mask)[0]
        if interior_idx.size == 0:
            return (
                np.zeros((0, self.n), dtype=bool),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
            )

        interior_coalitions = coalitions[interior_idx]
        packed = np.packbits(interior_coalitions, axis=1)
        packed_complement = np.packbits(~interior_coalitions, axis=1)
        local_index_by_key = {row.tobytes(): local for local, row in enumerate(packed)}

        n_interior = interior_idx.size
        processed = np.zeros(n_interior, dtype=bool)
        pair_a: list[int] = []
        pair_b: list[int] = []
        unpaired: list[int] = []
        for local in range(n_interior):
            if processed[local]:
                continue
            partner = local_index_by_key.get(packed_complement[local].tobytes())
            if partner is None:
                unpaired.append(local)
            else:
                pair_a.append(local)
                pair_b.append(partner)
                processed[partner] = True
            processed[local] = True

        idx_a = interior_idx[pair_a]
        idx_b = interior_idx[pair_b]
        idx_unpaired = interior_idx[unpaired]

        paired_coalitions = coalitions[idx_a]
        paired_weights = 2.0 * row_weights[idx_a]
        paired_targets = 0.5 * (game_values[idx_a] - game_values[idx_b])

        unpaired_coalitions = coalitions[idx_unpaired]
        unpaired_weights = row_weights[idx_unpaired]
        unpaired_targets = game_values[idx_unpaired] - beta_empty

        coalitions_used = np.concatenate([paired_coalitions, unpaired_coalitions], axis=0)
        weights_used = np.concatenate([paired_weights, unpaired_weights])
        targets_used = np.concatenate([paired_targets, unpaired_targets])

        return coalitions_used, weights_used, targets_used

    def _build_constraint_system(
        self,
        *,
        full_set_value: float,
        empty_set_value: float,
    ) -> tuple[float, float]:
        """Build the exact OddSHAP Fourier constraint scalars.

        Returns:
            beta_empty = (f(full) + f(empty)) / 2
                Exact Fourier coefficient for the empty interaction.
            b = -(f(full) - f(empty)) / 2
                The exact sum constraint on the non-empty odd coefficients:
                ``sum(beta_nonempty) == b``.

        Note:
            The non-empty coefficients live on the hyperplane
            ``sum(beta) == b``, whose normal is the all-ones vector. Projecting
            onto (and off) that hyperplane is therefore just row-mean / global-mean
            subtraction — see ``_solve_constrained_regression`` — so no K x K
            projection matrix (K = number of non-empty active terms) is ever
            built or multiplied.
        """
        n_nonempty_terms = self.n_active_interactions - 1
        if n_nonempty_terms <= 0:
            msg = "OddSHAP support must contain at least one non-empty interaction before building constraint objects."
            raise RuntimeError(msg)

        beta_empty = 0.5 * (full_set_value + empty_set_value)
        b = -0.5 * (full_set_value - empty_set_value)

        return beta_empty, b

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

        The textbook derivation projects onto the ``sum(beta) == b`` hyperplane
        via a dense K x K projection matrix ``I - ones(K,K)/K``. Because that
        matrix's only structure is "subtract the mean", the projection reduces
        algebraically to O(m*K) row-mean subtraction (for ``X_tilde @ projection``)
        and O(K) global-mean subtraction (for reconstructing beta_nonempty),
        which is what is implemented below instead of materializing the matrix.
        """
        beta_empty, b = self._build_constraint_system(
            full_set_value=full_set_value,
            empty_set_value=empty_set_value,
        )
        n_nonempty_terms = self.n_active_interactions - 1

        # Project the constrained problem into an unconstrained least-squares problem:
        # X_tilde @ (I - ones(K,K)/K) == X_tilde - row_mean(X_tilde), and
        # X_tilde @ (b/K * ones(K)) == b * row_mean(X_tilde).
        row_mean = X_tilde.mean(axis=1)
        X_projected = X_tilde - row_mean[:, np.newaxis]
        y_projected = y_tilde - b * row_mean

        # Solve for the free projected coordinates
        z_solution = np.linalg.lstsq(X_projected, y_projected, rcond=None)[0]

        # Reconstruct the constrained non-empty coefficient vector:
        # beta_const + (I - ones(K,K)/K) @ z_solution == b/K + z_solution - mean(z_solution).
        beta_nonempty = z_solution + (b / n_nonempty_terms - z_solution.mean())

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
        interaction_sizes = self.odd_interaction_matrix_float.sum(axis=1)  # (K,)
        # position 0 is () with size 0 — skip it; singletons have size 1
        coeffs = odd_fourier_coefficients[1:]
        sizes = interaction_sizes[1:]
        # float64 (BLAS-backed) matmul: an order of magnitude faster than the
        # equivalent bool matmul for this binary membership matrix.
        masks = self.odd_interaction_matrix_float[1:]  # (K-1, n) float64

        # phi_i = -2 * sum_{T: i in T} beta_T / |T|
        shares = coeffs / sizes  # (K-1,)
        sv_values[1:] = -2.0 * (masks.T @ shares)  # (n,)

        return sv_values
