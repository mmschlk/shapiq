"""Interventional TreeShap Explainer Implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.special import binom

from shapiq.game_theory.indices import get_computation_index
from shapiq.interaction_values import InteractionValues
from shapiq.tree.base import predict_ensemble
from shapiq.tree.subset_layout import output_size
from shapiq.tree.validation import validate_tree_model


def _ensemble_block(trees: list[TreeModel]) -> tuple[np.ndarray, ...]:
    """The ensemble as one node block, the layout every interventional C kernel takes.

    All trees' node arrays concatenated (node ids stay tree-relative), followed by
    ``tree_offsets`` and ``cat_offsets``: tree ``t`` owns nodes ``[tree_offsets[t],
    tree_offsets[t + 1])`` and categorical split values ``[cat_offsets[t], cat_offsets[t + 1])``.
    Eleven arrays, in the order the kernels expect them.
    """

    def concat(name: str, dtype: type) -> np.ndarray:
        return np.ascontiguousarray(
            np.concatenate([np.asarray(getattr(tree, name)).ravel() for tree in trees]), dtype=dtype
        )

    tree_offsets = np.zeros(len(trees) + 1, dtype=np.int64)
    tree_offsets[1:] = np.cumsum([np.asarray(tree.values).size for tree in trees])
    cat_offsets = np.zeros(len(trees) + 1, dtype=np.int64)
    cat_offsets[1:] = np.cumsum([np.asarray(tree.cat_values).size for tree in trees])
    return (
        concat("values", np.float64),
        concat("thresholds", np.float64),
        concat("features", np.int64),
        concat("children_left", np.int64),
        concat("children_right", np.int64),
        concat("children_left_default", bool),
        concat("cat_values", np.int64),
        concat("cat_start", np.int64),
        concat("cat_size", np.int64),
        tree_offsets,
        cat_offsets,
    )


# Largest structural layout (order-1 block + rows of the subset tables) the cohort kernel runs
# on: each thread owns a full copy of it. Beyond this the sparse per-explanation kernel is used.
_STRUCTURAL_MAX_ROWS = 1_000_000


if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.tree.base import TreeModel

InterventionalTreeSHAPIQIndices = Literal[
    "SV", "SII", "k-SII", "BII", "BV", "CHII", "CV", "FBII", "FSII", "STII", "CUSTOM"
]


class InterventionalTreeSHAPIQ:
    """Any-order interventional Shapley-interaction explainer for tree models.

    Extends interventional TreeSHAP :cite:t:`Zern.2023` to compute exact Shapley
    interactions of arbitrary order over a single decision tree or a tree ensemble (as
    validated by :func:`shapiq.tree.validation.validate_tree_model`). Each
    coalition's contribution is decomposed against a reference background
    dataset using the ``E``/``R`` partition (features fixed by the explained
    point vs. by the reference), and the recursion is offloaded to one of two
    C++ kernels:

    * **Structural path** — :func:`compute_interactions_cohort`: one cohort DFS per tree
      shared across the reference rows, accumulating any order into a dense array over the
      *structural layout*, the feature subsets that co-occur on some root-to-leaf path
      (collected once at construction, see :meth:`_preprocess_trees`). Used while
      that layout has at most :data:`_STRUCTURAL_MAX_ROWS` entries.
    * **Sparse path** — :func:`compute_interactions_batched_sparse`: one DFS per
      (tree, reference row) into a per-explanation hash map, for layouts beyond the budget
      or feature counts whose subset keys overflow an ``int64``.

    Indices supported via the C path are listed in
    :data:`InterventionalTreeSHAPIQIndices`. Custom weight functions are
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
            used to define interventional baselines, rounded the way the source library
            rounds prediction inputs (see :meth:`~shapiq.tree.base.TreeModel.cast_input`).
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
        *,
        class_index: int | None = None,
        max_order: int = 2,
        index: InterventionalTreeSHAPIQIndices = "SII",
        weight_fn: Callable[[int, int, int], float] | None = None,
    ) -> None:
        r"""Initialize the InterventionalTreeSHAPIQ.

        Args:
            model: A fitted tree or tree ensemble compatible with
                :func:`shapiq.tree.validation.validate_tree_model` (sklearn,
                XGBoost, LightGBM, or a precomputed leaf-matrix list).
            data: Background dataset of shape ``(n_ref, n_features)`` defining
                the interventional baseline.
            class_index: Class index for classifiers. For binary
                ``predict_proba`` models, defaults to ``1`` if left as
                ``None``. Ignored for regressors.
            max_order: Maximum interaction order to compute. Defaults to ``2``.
            index: Interaction index; one of
                :data:`InterventionalTreeSHAPIQIndices`. Replaced with
                ``"CUSTOM"`` when ``weight_fn`` is supplied. Defaults to
                ``"SII"``.
            weight_fn: Optional custom weight callable with signature
                ``weight_fn(coalition_size, interaction_size, n_players) -> float``.
                When supplied, overrides ``index`` and triggers building a
                precomputed lookup table. Defaults to ``None``.
        """
        if class_index is None and hasattr(model, "predict_proba"):
            class_index = 1
        self.tree = validate_tree_model(model, class_label=class_index)
        # rounded the way the source library rounds prediction inputs (see cast_input);
        # the kernels themselves compute in float64
        self.reference_data: np.ndarray = self.tree[0].cast_input(
            np.asarray(data, dtype=np.float64)
        )
        self.max_order = max_order
        self.index = index
        self.n_players = data.shape[1]
        self.n_features = self.reference_data.shape[1]
        self.look_up_table: np.ndarray | None = None
        if weight_fn is not None:
            self.weight_fn = weight_fn
            self.index = "CUSTOM"
            self.look_up_table = self._build_custom_weight_table()
        self._baseline_value: float | None = None

        # the ensemble block and its structural subset tables are built once; the tables
        # decide the route: the structural kernel writes a dense array over the co-occurring
        # subsets, the sparse kernel maps per explanation
        self._preprocess_trees()
        self._use_sparse_path = (
            self._subset_index is None or self._n_structural_interactions > _STRUCTURAL_MAX_ROWS
        )
        self._structural_out: np.ndarray | None = None

    @property
    def baseline_value(self) -> float:
        """The interventional baseline value (empty prediction) of the explained ensemble.

        The mean ensemble prediction over the reference data, computed lazily on first access
        and cached.
        """
        if self._baseline_value is None:
            self._baseline_value = self.compute_empty_prediction(self.tree, self.reference_data)
        return self._baseline_value

    @staticmethod
    def compute_empty_prediction(trees: list[TreeModel], reference_data: np.ndarray) -> float:
        """Compute the interventional empty prediction of a tree ensemble.

        The interventional empty prediction (baseline value) is the mean ensemble prediction
        over the reference (background) dataset. Inputs are rounded the way the source library
        rounds prediction inputs (see :meth:`~shapiq.tree.base.TreeModel.cast_input`), so the
        result matches the kernels' split routing. Routing runs in the C++
        ``predict_ensemble_sum`` kernel; trees with a remapped feature space or a non-zero root
        (which the C kernels do not model) fall back to the Python
        :func:`~shapiq.tree.base.predict_ensemble` path.

        Args:
            trees: The validated trees of the ensemble (see
                :func:`shapiq.tree.validation.validate_tree_model`).
            reference_data: Background dataset of shape ``(n_ref, n_features)``.

        Returns:
            The mean ensemble prediction over the reference data.
        """
        reference_data = np.asarray(reference_data, dtype=np.float64)
        plain_trees = all(
            tree.root_node_id == 0
            and all(k == v for k, v in tree.feature_map_internal_original.items())
            for tree in trees
        )
        if plain_trees:
            from .cext import predict_ensemble_sum  # ty: ignore[unresolved-import]

            predictions = predict_ensemble_sum(
                *_ensemble_block(trees), trees[0].cast_input(reference_data), trees[0].decision_type
            )
            return float(predictions.mean())
        # predict_ensemble routes through TreeModel.predict_one, which applies cast_input
        return float(predict_ensemble(trees, reference_data).mean())

    def _preprocess_trees(self) -> None:
        """Build the ensemble block and the structural subset tables, once per explainer.

        The block (:func:`_ensemble_block`) is what every kernel reads the trees from. A
        leaf's ``E`` and ``R`` sets are drawn from its root-to-leaf path, so only subsets of
        features co-occurring on some path can be non-zero -- usually a small fraction of all
        ``C(n_features, order)`` combinations. One structural DFS per tree (C++
        ``preprocess_subset_tables``) collects them into per-order sorted tables together with
        the flat hash index the kernel probes (``cext/subset_tables.hpp``, shared with the
        quadrature explainer). The index is ``None`` when the integer key encoding would
        overflow (``n_features ** max_order`` beyond ``int64``); the sparse kernel is used then.
        """
        from .cext import preprocess_subset_tables  # ty: ignore[unresolved-import]

        self._ensemble: tuple[np.ndarray, ...] = _ensemble_block(self.tree)
        self._decision_type: str = self.tree[0].decision_type
        _, _, features, children_left, children_right, _, _, _, _, tree_offsets, _ = self._ensemble
        self._subset_tables: tuple[np.ndarray, np.ndarray]
        self._subset_index: tuple[np.ndarray, np.ndarray] | None
        self._subset_tables, self._subset_index = preprocess_subset_tables(
            features,
            children_left,
            children_right,
            tree_offsets,
            int(self.n_features),
            int(self.max_order),
        )
        self._n_structural_interactions = output_size(self._subset_tables[1], self.n_features, 1)

    def _explain_structural(self, x: np.ndarray, computation_index: str) -> dict:
        """Cohort kernel over the structural layout; the non-zero interactions as a dict."""
        from .cext import (
            compute_interactions_cohort,  # ty: ignore[unresolved-import]
            layout_to_dict,  # ty: ignore[unresolved-import]
        )

        if self._subset_index is None:
            msg = "the structural layout is unavailable for this explainer (keys overflow)."
            raise RuntimeError(msg)
        keys, counts = self._subset_tables
        if self._structural_out is None:
            self._structural_out = np.zeros(self._n_structural_interactions, dtype=np.float64)
        compute_interactions_cohort(
            *self._ensemble,
            self.reference_data,
            x.flatten(),
            self._decision_type,
            computation_index,
            self.max_order,
            self.look_up_table,
            counts,
            *self._subset_index,
            self._structural_out,
        )
        # the shared C++ readout, skipping exact zeros (as the sparse route reports only the
        # interactions it touched)
        return layout_to_dict(self._structural_out, keys, counts, self.n_features, 1, None, True)  # noqa: FBT003

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

    def explain(self, x: np.ndarray) -> InteractionValues:
        """Compute interaction values for a single instance (alias of ``explain_function``)."""
        return self.explain_function(x)

    def explain_function(
        self,
        x: np.ndarray,
        **_: dict,
    ) -> InteractionValues:
        """Compute interaction values for a single instance.

        Routes to the structural cohort kernel while its layout fits the memory budget and
        to the sparse batched kernel otherwise (see the class docstring). Both routes report
        the non-zero interactions; the empty interaction ``()`` is always populated with
        ``self.baseline_value`` before constructing the result.

        Args:
            x: The instance to explain, as a 1-D array of length
                ``self.n_players``.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` carrying the
            requested interaction ``index``, ``max_order=self.max_order``, and
            ``min_order=1`` as the declared lower bound on computed orders.
        """
        from .cext import compute_interactions_batched_sparse  # ty: ignore[unresolved-import]

        # round the instance the way the source library rounds prediction inputs
        x = self.tree[0].cast_input(np.asarray(x, dtype=np.float64))

        computation_index = get_computation_index(self.index)
        # _use_sparse_path is set in __init__: the structural layout within the memory budget,
        # else the per-explanation sparse kernel; both report the non-zero interactions
        if not self._use_sparse_path:
            interactions = self._explain_structural(x, computation_index)
        else:
            interactions = compute_interactions_batched_sparse(
                *self._ensemble,
                self.reference_data,
                x.flatten(),
                self._decision_type,
                computation_index,
                self.max_order,
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
