"""Interventional TreeShap Explainer Implementation."""

from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import binom

from shapiq.game_theory.indices import get_computation_index
from shapiq.interaction_values import InteractionValues
from shapiq.tree.validation import validate_tree_model

from .cext import (
    compute_interactions_batched_sparse,  # ty: ignore[unresolved-import]
    compute_interactions_flatten,  # ty: ignore[unresolved-import]
    preprocess_boolean_trees,  # ty: ignore[unresolved-import]
)

# The dense `compute_interactions_flatten` path allocates a result buffer of
# sum(C(n, k) for k in 1..max_order) doubles, plus one such buffer per OpenMP
# thread inside the order-2/3 leaf-parallel kernels. Above this threshold we
# redirect to the sparse `compute_interactions_batched_sparse` path, which only
# materializes interactions actually touched by tree paths. 1_000_000 entries
# = 8 MB per thread — comfortably below typical RAM budgets.
_DENSE_FLATTEN_MAX_RESULT_SIZE = 1_000_000


def _dense_flatten_result_size(n_features: int, max_order: int) -> int:
    return sum(comb(n_features, k) for k in range(1, max_order + 1))


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
    """Any-order interventional Shapley-interaction explainer for tree models.

    Extends interventional TreeSHAP to compute exact Shapley interactions of
    arbitrary order over a single decision tree or a tree ensemble (as
    validated by :func:`shapiq.tree.validation.validate_tree_model`). Each
    coalition's contribution is decomposed against a reference background
    dataset using the ``E``/``R`` partition (features fixed by the explained
    point vs. by the reference), and the recursion is offloaded to one of two
    C++ kernels:

    * **Dense flatten path** — :func:`compute_interactions_flatten` for
      ``max_order <= 3`` when the dense buffer fits within
      :data:`_DENSE_FLATTEN_MAX_RESULT_SIZE`.
    * **Sparse path** — :func:`compute_interactions_batched_sparse` for
      higher orders or wide-feature trees.

    Indices supported via the C path are listed in
    :data:`INDICES_C_IMPLEMENTATION_CAPABLE`. Custom weight functions are
    accepted via ``weight_fn`` and routed through a precomputed lookup table.

    The baseline value is computed from the validated trees by summing per-tree
    predictions over the reference data. ``validate_tree_model`` already scales
    sklearn ensemble trees by ``1/n_estimators`` and extracts class-specific raw
    scores for XGBoost / LightGBM classifiers, so this single path lands on the
    same scale as the kernel for every supported model.

    Attributes:
        tree: Validated tree (or list of trees) from
            :func:`validate_tree_model`.
        reference_data: Background dataset (shape ``(n_ref, n_features)``)
            used to define interventional baselines, cast to ``float32``.
        baseline_value: Mean tree-prediction over ``reference_data`` (scalar);
            written as the order-0 entry of the returned interactions.
        max_order: Maximum interaction order computed.
        index: Interaction index (e.g. ``"SII"``); replaced with ``"CUSTOM"``
            when ``weight_fn`` is supplied.
        n_players: Number of features.
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
        r"""Initialize the InterventionalTreeExplainer.

        Args:
            model: A fitted tree or tree ensemble compatible with
                :func:`shapiq.tree.validation.validate_tree_model` (sklearn,
                XGBoost, LightGBM, or a precomputed leaf-matrix list).
            data: Background dataset of shape ``(n_ref, n_features)`` defining
                the interventional baseline.
            class_index: Class index for classifiers. For binary
                ``predict_proba`` models, defaults to ``1`` if left as
                ``None``. Ignored for regressors.
            debug: If ``True``, the C++ kernel prints debug information.
                Defaults to ``False``.
            max_order: Maximum interaction order to compute. Defaults to ``2``.
            index: Interaction index; one of
                :data:`INDICES_C_IMPLEMENTATION_CAPABLE`. Replaced with
                ``"CUSTOM"`` when ``weight_fn`` is supplied. Defaults to
                ``"SII"``.
            index_func: Reserved for a Python-side custom index function.
                Defaults to ``None``.
            bool_tree: If ``True``, the tree is treated as boolean (coalitions
                in :math:`\\{0, 1\\}^n`) and preprocessed once with the
                BitSet DFS C++ helper instead of per-explanation. Used by
                :class:`~proxyshap.proxyshap.ProxySHAP`. Defaults to ``False``.
            weight_fn: Optional custom weight callable with signature
                ``weight_fn(coalition_size, interaction_size, n_players) -> float``.
                When supplied, overrides ``index`` and triggers building a
                precomputed lookup table. Defaults to ``None``.
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
        # Compute the interventional baseline directly from the validated trees.
        per_sample_predictions = np.array(
            [sum(t.predict_one(ref) for t in self.tree) for ref in self.reference_data],
            dtype=np.float64,
        )
        self.baseline_value = float(per_sample_predictions.mean())

        # The sparse C path needs the per-tree flattened arrays. Populate them
        # whenever we'll route there: max_order > 3 (always sparse) or when the
        # dense flatten path's result buffer would exceed our memory budget.
        n_features_hint = int(self.reference_data.shape[1])
        self._use_sparse_path = (
            self.max_order > 3
            or _dense_flatten_result_size(n_features_hint, self.max_order)
            > _DENSE_FLATTEN_MAX_RESULT_SIZE
        )
        if self._use_sparse_path:
            self._preprocess_tree_sparse_path()
        if self.bool_tree:
            self._preprocess_boolean_tree()

    def _preprocess_tree_sparse_path(self) -> None:
        """Flatten per-tree arrays into the layout expected by the sparse C kernel."""
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
        """Gather E and R statistics for boolean tree mode using C++ BitSet DFS."""
        self.n_features = self.reference_data.shape[1]

        # Prepare tree arrays for C++
        values_list = [tree.values.astype(np.float32).flatten() for tree in self.tree]
        features_list = [tree.features.astype(np.int64).flatten() for tree in self.tree]
        children_left_list = [tree.children_left.astype(np.int64).flatten() for tree in self.tree]
        children_right_list = [tree.children_right.astype(np.int64).flatten() for tree in self.tree]

        (
            self.E_R_flatten,
            self.leaf_vals_flatten,
            self.e_size_flatten,
            self.r_size_flatten,
            self.feature_in_E,
            self.leaf_id,
        ) = preprocess_boolean_trees(
            values_list,
            features_list,
            children_left_list,
            children_right_list,
            self.n_features,
        )

        self.e_length = len(self.E_R_flatten)
        self.n_leafs = int(self.leaf_id[-1]) + 1 if len(self.leaf_id) > 0 else 0

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
        """Convert a discrete-derivative weight to its Möbius counterpart.

        Args:
            weight_func: Callable
                ``weight_func(coalition_size, interaction_size, n_players) -> float``
                returning the discrete-derivative weight.
            coalition_size: Size of the coalition.
            interaction_size: Size of the interaction.

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
        """Compute interaction values for a single instance.

        Routes to the dense flatten C kernel for low-order, narrow-feature
        cases and to the sparse batched C kernel otherwise (see the class
        docstring). The empty interaction ``()`` is always populated with
        ``self.baseline_value`` before constructing the result.

        Args:
            x: The instance to explain, as a 1-D array of length
                ``self.n_players``.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` carrying the
            requested interaction ``index``, ``max_order=self.max_order``, and
            ``min_order=1`` as the declared lower bound on computed orders.
            Note that the underlying interactions dict still contains the
            empty-set entry ``()`` set to ``self.baseline_value``, even though
            ``min_order`` reports ``1``.
        """
        if not self.bool_tree and not self._use_sparse_path:
            self._preprocess_tree(x)
        computation_index = get_computation_index(self.index)
        interactions = {}
        # For higher order interactions we need to use the sparse implementation as the flatten one is only optimized for main effects, pairwise, and triple interactions.
        # For orders up to 3, we can use the flatten implementation which is faster.
        # We also redirect to sparse for orders <= 3 when n_features is large
        # enough that the dense flatten buffer would blow memory (see
        # _DENSE_FLATTEN_MAX_RESULT_SIZE). _use_sparse_path is set in __init__.
        if self._use_sparse_path:
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
                computation_index,
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
                computation_index,
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
            index=computation_index,
            n_players=self.n_players,
            baseline_value=self.baseline_value,
            target_index=self.index,
        )
