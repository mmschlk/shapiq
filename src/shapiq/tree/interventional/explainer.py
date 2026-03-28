"""Interventional TreeShap Explainer Implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import binom

from shapiq.interaction_values import InteractionValues
from shapiq.tree.validation import validate_tree_model
from shapiq.utils.modules import safe_isinstance

from .cext import (
    compute_interactions_batched_sparse,  # ty: ignore[unresolved-import]
    compute_interactions_flatten,  # ty: ignore[unresolved-import]
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.tree.base import TreeModel

INDICES_C_IMPLEMENTATION_CAPABLE = [
    "SV",
    "SII",
    "BII",
    "BV",
    "SV",
    "CHII",
    "CV",
    "FBII",
    "FSII",
    "STII",
    "CUSTOM",
]
INDICES_CII_IMPLEMENTATION_CAPABLE = ["WBII", *INDICES_C_IMPLEMENTATION_CAPABLE]


def obtain_E_R_values(tree: TreeModel) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Obtain two arrays E and R indicating for each leaf and each feature whether the leaf is reachable by having the feature equal to 1 (E) or equal to 0 (R)."""
    E = []
    R = []
    leaf_vals = []

    # Iterative DFS — avoids Python recursion overhead and recursion-depth limits.
    # Stack entries: (node_id, e_set, r_set)
    stack = [(0, frozenset(), frozenset())]
    while stack:
        node_id, e_set, r_set = stack.pop()
        if tree.children_left[node_id] == tree.children_right[node_id]:  # leaf
            E.append(np.array(sorted(e_set), dtype=np.int64))
            R.append(np.array(sorted(r_set), dtype=np.int64))
            leaf_vals.append(tree.values[node_id].item())
            continue

        feature = int(tree.features[node_id])

        # Go left: feature must be 0 → add to R (unless already constrained to 1).
        if feature not in e_set:
            stack.append((tree.children_left[node_id], e_set, r_set | {feature}))

        # Go right: feature must be 1 → add to E (unless already constrained to 0).
        if feature not in r_set:
            stack.append((tree.children_right[node_id], e_set | {feature}, r_set))

    return E, R, leaf_vals


def obtain_E_R_values_point(
    tree: TreeModel, point_to_explain: np.ndarray, reference_point: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Obtain two arrays E and R indicating for each leaf and each feature whether the leaf was reached due to features in point_to_explain (E) or due to features in reference_point (R)."""
    E = []
    R = []
    leaf_vals = []

    # Iterative DFS — avoids Python recursion overhead and recursion-depth limits.
    # Stack entries: (node_id, e_set, r_set)
    stack = [(0, frozenset(), frozenset())]
    while stack:
        node_id, e_set, r_set = stack.pop()
        if tree.children_left[node_id] == tree.children_right[node_id]:  # leaf
            E.append(np.array(sorted(e_set), dtype=np.int64))
            R.append(np.array(sorted(r_set), dtype=np.int64))
            leaf_vals.append(tree.values[node_id].item())
            continue

        feature = int(tree.features[node_id])
        explain_goes_left = tree.decision_function(
            point_to_explain[feature],
            tree.thresholds[node_id],
            left_default=tree.children_left_default[node_id],
        )
        child_node_explain = (
            tree.children_left[node_id] if explain_goes_left else tree.children_right[node_id]
        )
        ref_goes_left = tree.decision_function(
            reference_point[feature],
            tree.thresholds[node_id],
            left_default=tree.children_left_default[node_id],
        )
        child_node_ref = (
            tree.children_left[node_id] if ref_goes_left else tree.children_right[node_id]
        )
        if child_node_explain != child_node_ref:
            if feature not in r_set:  # Feature is not fixed by the reference point
                stack.append((child_node_explain, e_set | {feature}, r_set))
            if feature not in e_set:  # Feature is not fixed by the explain point
                stack.append((child_node_ref, e_set, r_set | {feature}))
        else:
            stack.append((child_node_explain, e_set, r_set))
    return E, R, leaf_vals


class InterventionalTreeExplainer:
    """Interventional Tree Shap explainer for a single decision tree.

    This class implements the interventional TreeSHAP algorithm for a single
    sklearn DecisionTreeClassifier or DecisionTreeRegressor.

    Attributes:
        tree: Validated tree structure returned by validate_tree_model for the fitted
            sklearn tree.
        data: Background dataset used to compute the reference point.
        reference_point: The baseline/reference feature values (mean over the background data).
    """

    def __init__(
        self,
        model: object,
        data: np.ndarray,
        class_index: int | None = None,
        *,
        debug: bool = False,
        max_order: int = 2,
        index: str = "SII",
        index_func: Callable | None = None,
        bool_tree: bool = False,
        weight_fn: Callable[[int, int, int], float] | None = None,
    ) -> None:
        """Initialize the InterventionalTreeExplainer.

        Args:
            model: The tree model to explain.
            data: Background dataset used to compute reference points.
            class_index: Class index for classification models. Defaults to ``None``.
            debug: Whether to print debug information. Defaults to ``False``.
            max_order: Maximum order of interactions. Defaults to ``2``.
            index: The interaction index to use. Defaults to ``"SII"``.
            index_func: Custom index function. Defaults to ``None``.
            p: Probability parameter for WBII index. Defaults to ``0.5``.
            bool_tree: Whether to use boolean tree mode. Defaults to ``False``.
            weight_fn: Custom weight callable with signature
                ``weight_fn(coalition_size, interaction_size, n_players) -> float``.
                When provided, overrides ``index`` and uses a precomputed lookup table.
                Defaults to ``None``.
        """
        # If Classification model and class_index is None, set to 1
        if class_index is None and hasattr(model, "predict_proba"):
            class_index = 1
        self.tree = validate_tree_model(model, class_label=class_index)
        self.reference_data: np.ndarray = data.astype(np.float32)
        self.debug = debug
        self.max_order = max_order
        self.index = index
        self.n_players = data.shape[1]
        self.index_func = index_func
        self.bool_tree = bool_tree
        self.look_up_table: np.ndarray | None = None
        self._custom_weight_table: np.ndarray | None = None
        if weight_fn is not None:
            self.weight_fn = weight_fn
            self.index = "CUSTOM"
            self.look_up_table = self._build_custom_weight_table()
        if class_index is not None:
            # If XGBoost model, use DMatrix to obtain logits
            if safe_isinstance(model, "xgboost.sklearn.XGBClassifier"):
                import xgboost as xgb

                dmatrix_data = xgb.DMatrix(self.reference_data)
                booster = model.get_booster()  # ty: ignore[unresolved-attribute]
                logits = booster.predict(dmatrix_data, output_margin=True)
                if logits.ndim == 1:
                    # Binary classification case
                    if class_index == 1:
                        self.baseline_value = np.mean(logits).astype(np.float64)
                    else:
                        self.baseline_value = np.mean(-logits).astype(np.float64)
                else:
                    self.baseline_value = np.mean(logits[:, class_index]).astype(np.float64)
            elif safe_isinstance(model, "lightgbm.LGBMClassifier"):
                raw_scores = model.predict(  # ty: ignore[unresolved-attribute]
                    self.reference_data, raw_score=True
                )
                if raw_scores.ndim == 1:
                    self.baseline_value = (
                        np.mean(raw_scores) if class_index == 1 else np.mean(-raw_scores)
                    ).astype(np.float64)
                else:
                    self.baseline_value = np.mean(raw_scores[:, class_index]).astype(np.float64)
            else:
                proba = model.predict_proba(self.reference_data)  # ty: ignore[unresolved-attribute]
                self.baseline_value = np.mean(proba[:, class_index]).astype(np.float64)
        # Check if we have the case of a leaf_matrix directly provided
        elif isinstance(model, list):
            self.baseline_value = model[0][-1]  # ty: ignore[not-subscriptable]
        else:
            prediction = model.predict(self.reference_data)  # ty: ignore[unresolved-attribute]
            self.baseline_value = np.mean(prediction).astype(np.float64)

        if self.max_order > 2:
            self._preprocess_tree_higher_order()
        if self.bool_tree:
            self._preprocess_boolean_tree()

    def _preprocess_tree_higher_order(self) -> None:
        """Preprocess the tree to gather necessary information for higher-order interactions.

        This method gather the arrays from the tree's as a list to give them to the C implementation for higher-order interactions.
        """
        self.values_list = [tree.values.astype(np.float32).flatten() for tree in self.tree]
        self.threshold_list = [tree.thresholds.astype(np.float32).flatten() for tree in self.tree]
        self.features_list = [tree.features.astype(np.int64).flatten() for tree in self.tree]
        self.children_left_list = [
            tree.children_left.astype(np.int64).flatten() for tree in self.tree
        ]
        self.children_right_list = [
            tree.children_right.astype(np.int64).flatten() for tree in self.tree
        ]
        self.children_left_default_list = [
            tree.children_left_default.astype(bool).flatten() for tree in self.tree
        ]

    def _preprocess_boolean_tree(self) -> None:
        """Gather E and R statistics for boolean tree mode."""
        self.n_features = self.reference_data.shape[1]
        E_list: list[np.ndarray] = []
        R_list: list[np.ndarray] = []
        leaf_vals_list: list[float] = []

        for tree in self.tree:
            E, R, leaf_vals = obtain_E_R_values(tree)
            E_list.extend(E)
            R_list.extend(R)
            leaf_vals_list.extend(leaf_vals)

        self.E_list = E_list
        self.R_list = R_list
        self.leaf_vals = np.array(leaf_vals_list, dtype=np.float32)
        n_leafs = len(E_list)

        # Per-leaf sizes — computed once and reused for all flattened arrays.
        e_sizes = np.array([len(e) for e in E_list], dtype=np.int64)
        r_sizes = np.array([len(r) for r in R_list], dtype=np.int64)
        er_sizes = e_sizes + r_sizes

        self.n_features_e = e_sizes
        self.n_features_r = r_sizes

        # Build flatten numpy arrays
        if n_leafs > 0:
            self.E_R_flatten = np.concatenate(
                [np.concatenate([e, r]) for e, r in zip(E_list, R_list, strict=False)]
            ).astype(np.int64)
            self.feature_in_E = np.concatenate(
                [
                    np.concatenate(
                        [np.ones(int(e), dtype=np.int64), np.zeros(int(r), dtype=np.int64)]
                    )
                    for e, r in zip(e_sizes, r_sizes, strict=False)
                ]
            )
        else:
            self.E_R_flatten = np.array([], dtype=np.int64)
            self.feature_in_E = np.array([], dtype=np.int64)

        self.leaf_id = np.repeat(np.arange(n_leafs, dtype=np.int64), er_sizes)
        self.leaf_vals_flatten = np.repeat(self.leaf_vals, er_sizes)
        self.e_size_flatten = np.repeat(e_sizes, er_sizes)
        self.r_size_flatten = np.repeat(r_sizes, er_sizes)
        self.e_length = len(self.E_R_flatten)
        self.n_leafs = n_leafs

    def _preprocess_tree(self, explain_point: np.ndarray) -> None:
        """Gather E and R statistics for the given explain point.

        Args:
            explain_point: The instance to explain as a 1-dimensional array.
        """
        self.n_features = self.reference_data.shape[1]
        E_list: list[np.ndarray] = []
        R_list: list[np.ndarray] = []
        leaf_vals_list: list[float] = []

        for r in self.reference_data:
            for tree in self.tree:
                E, R, leaf_vals = obtain_E_R_values_point(tree, explain_point, r)
                E_list.extend(E)
                R_list.extend(R)
                leaf_vals_list.extend(leaf_vals)

        self.E_list = E_list
        self.R_list = R_list
        self.leaf_vals = np.array(leaf_vals_list, dtype=np.float32)
        n_leafs = len(E_list)

        # Per-leaf sizes — computed once and reused for all flattened arrays.
        e_sizes = np.array([len(e) for e in E_list], dtype=np.int64)
        r_sizes = np.array([len(r) for r in R_list], dtype=np.int64)
        er_sizes = e_sizes + r_sizes

        self.n_features_e = e_sizes
        self.n_features_r = r_sizes

        # Build flatten numpy arrays
        if n_leafs > 0:
            self.E_R_flatten = np.concatenate(
                [np.concatenate([e, r]) for e, r in zip(E_list, R_list, strict=False)]
            ).astype(np.int64)
            self.feature_in_E = np.concatenate(
                [
                    np.concatenate(
                        [np.ones(int(e), dtype=np.int64), np.zeros(int(r), dtype=np.int64)]
                    )
                    for e, r in zip(e_sizes, r_sizes, strict=False)
                ]
            )
        else:
            self.E_R_flatten = np.array([], dtype=np.int64)
            self.feature_in_E = np.array([], dtype=np.int64)

        self.leaf_id = np.repeat(np.arange(n_leafs, dtype=np.int64), er_sizes)
        self.leaf_vals_flatten = np.repeat(self.leaf_vals, er_sizes)
        self.e_size_flatten = np.repeat(e_sizes, er_sizes)
        self.r_size_flatten = np.repeat(r_sizes, er_sizes)
        self.e_length = len(self.E_R_flatten)
        self.n_leafs = n_leafs

    def _build_custom_weight_table(self) -> np.ndarray:
        """Precompute the flat weight lookup table for the custom weight function.

        Returns:
            A 1-D float64 numpy array of size ``(n+1) * (n+1) * (k+1)^3`` where
            ``n = self.n_features`` and ``k = self.max_order``.
        """
        n = self.n_features
        k = self.max_order
        N = n + 1
        K = k + 1
        table = np.zeros(N * N * K * K * K, dtype=np.float64)
        for e in range(N):
            for r in range(
                N - e
            ):  # r can only go up to n - e since we can't have more than n features in total
                for s in range(
                    1, min(k + 1, e + r)
                ):  # s can only go up to e + r since we can't have an interaction of size s if we don't have at least s features in total in E and R
                    for s_cap_r in range(
                        min(r, s) + 1
                    ):  # s_cap_r can only go up to min(r, s) since we can't have more than r features in R and we can't have more than s features in the interaction
                        if s_cap_r > r:
                            continue
                        idx = e * (N * K * K * K) + r * (K * K) + s_cap_r * K + s
                        table[idx] = self._general_weight(e, r, s_cap_r, s, n)
        return table

    def _discrete_weight_to_moebius(
        self,
        weight_func: Callable[[int, int, int], float],
        coalition_size: int,
        interaction_size: int,
    ) -> float:
        """Convert a discrete weight function to a Möbius weight function.

        Args:
            weight_func: A callable with signature ``weight_func(coalition_size, interaction_size, n_players) -> float`` that returns the weight for a given coalition size and interaction size. This weight represents the "discrete derivative" weight.
            coalition_size: Size of the coalition.
            interaction_size: Size of the interaction.
            n_players: Total number of players.

        Returns:
            The corresponding Möbius weight.
        """
        return weight_func(coalition_size - interaction_size, interaction_size, coalition_size)

    def _general_weight(
        self,
        e: int,
        r: int,
        s_cap_r: int,
        s: int,
        n: int,
    ) -> float:
        r"""Computes a the general weight $\lambda$ for the interventional tree algorithm.

        Args:
            e: Number of features in E.
            r: Number of features in R.
            s_cap_r: Number of features in R that are part of the interaction.
            s: Size of the interaction.
            n: Total number of features.

        Returns:
            The weight $\lambda$ for the given parameters.
        """
        b = n - r
        sign = (-1) ** s_cap_r
        return sign * sum(
            [
                (-1) ** k
                * binom(n - b - s_cap_r, k)
                * self._discrete_weight_to_moebius(
                    weight_func=self.weight_fn, coalition_size=k + s_cap_r + e, interaction_size=s
                )
                for k in range(n - b - s_cap_r + 1)
            ]
        )

    def explain_function(
        self,
        x: np.ndarray,
        **_: dict,
    ) -> InteractionValues:
        """Computes the CII values for a single instance using interventional approach with a pure C++ implementation.

        Args:
            x: The instance to explain as a 1-dimensional array.

        Returns:
            InteractionValues object containing the computed interaction values.
        """
        if not self.bool_tree:
            self._preprocess_tree(x)

        # Build (and cache) the custom weight table once if using a custom weight function.

        interactions = {}
        # For higher order interactions we need to use the sparse implementation as the flatten one is only optimized for main effects and pairwise interactions. For main effects and pairwise interactions we can use the flatten implementation which is faster.
        if self.max_order > 2:
            interactions = compute_interactions_batched_sparse(
                self.values_list,
                self.threshold_list,
                self.features_list,
                self.children_left_list,
                self.children_right_list,
                self.children_left_default_list,
                self.reference_data.astype(np.float32),
                x.astype(np.float32).flatten(),
                self.tree[0].decision_type,
                self.index,
                self.max_order,
                self.debug,  # whether to print debug information
                self.look_up_table,  # optional custom weight table (None → built-in index)
            )
        else:
            interactions = compute_interactions_flatten(
                self.leaf_vals_flatten,
                self.E_R_flatten,
                self.e_size_flatten,
                self.r_size_flatten,
                self.feature_in_E,
                self.leaf_id,
                self.index,
                len(self.leaf_vals_flatten),
                self.n_features,
                self.e_length,
                self.max_order,
                self.debug,  # whether to print debug information
                float(
                    self.reference_data.shape[0]
                ),  # number of reference samples for scaling the results
                self.look_up_table,  # optional custom weight table (None → built-in index)
            )
        interactions[()] = self.baseline_value
        return InteractionValues(
            interactions,
            max_order=self.max_order,
            min_order=1,
            index=self.index,
            n_players=self.n_players,
            baseline_value=self.baseline_value,
        )
