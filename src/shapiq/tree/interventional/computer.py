"""Interventional TreeShap Explainer Implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.special import binom

from shapiq.game_theory.indices import get_computation_index
from shapiq.interaction_values import InteractionValues
from shapiq.tree.base import predict_ensemble
from shapiq.tree.validation import validate_tree_model


def _ensemble_block(trees: list[TreeModel]) -> tuple[np.ndarray, ...]:
    """The ensemble as the eleven arrays every interventional C kernel takes.

    All trees' node arrays concatenated (node ids stay tree-relative), followed by
    ``tree_offsets`` and ``cat_offsets``: tree ``t`` owns nodes ``[tree_offsets[t],
    tree_offsets[t + 1])`` and categorical split values ``[cat_offsets[t], cat_offsets[t + 1])``.
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


# A table of interactions of order 2 or 3 holding more than this fraction of all interactions of
# its order is completed to all of them, and the kernel ranks instead of probing. 1.0 disables it.
_STRUCTURAL_COMPLETE_ABOVE = 0.5


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
    validated by :func:`shapiq.tree.validation.validate_tree_model`). Each leaf's
    contribution is decomposed with the ``E``/``R`` partition (features fixed by the
    explained point vs. by the reference) and weighted by the interaction index.

    The C++ extension writes every order into one output array holding only the interactions
    that can be non-zero: those whose features all occur on one root-to-leaf path, collected
    once at construction (:meth:`_preprocess_trees`). Two fills write that array through one
    leaf update: ``fill_cohort`` walks every tree once for all reference rows of a background
    dataset, ``fill_single_reference`` handles a single reference row.

    Indices supported are listed in :data:`InterventionalTreeSHAPIQIndices`. Custom weight
    functions are accepted via ``weight_fn`` and passed as a precomputed lookup table.

    The baseline value is the mean of the validated trees' predictions over the reference
    data, which ``validate_tree_model`` has already put on the scale of the kernel.

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
        # rounded the way the source library rounds prediction inputs (cast_input)
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

        # the ensemble arrays and the tables of interactions are built once per explainer
        self._preprocess_trees()

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

        The mean ensemble prediction over the reference dataset, with the inputs rounded like
        prediction inputs (:meth:`~shapiq.tree.base.TreeModel.cast_input`). Routing runs in the
        C++ ``predict_ensemble_sum``; trees with a remapped feature space or a non-zero root
        fall back to the Python :func:`~shapiq.tree.base.predict_ensemble`.

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
        """Build the ensemble arrays and the tables of interactions, once per explainer.

        The C++ ``preprocess_subset_tables`` walks every tree once and collects, per order, the
        table of interactions whose features all occur on one root-to-leaf path (each entry
        listing its feature ids) plus the hash index from interaction to position. A table of
        order 2 or 3 holding more than :data:`_STRUCTURAL_COMPLETE_ABOVE` of all interactions
        of its order is completed. The index is ``None`` when an interaction's 128-bit key
        would overflow; the fills then search the tables.
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
            _STRUCTURAL_COMPLETE_ABOVE,
        )
        counts = self._subset_tables[1]
        # the output array: the order-1 block, then one entry per table entry of orders 2 and up
        self._n_outputs = int(self.n_features) + int(counts[2:].sum())

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
                # an interaction of size s needs s features in E u R: s <= min(k, e + r)
                for s in range(1, min(k, e + r) + 1):
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
        r"""Computes the general weight $\lambda$ for the interventional tree algorithm.

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

        Runs the single-reference fill when the explainer holds one reference row and the
        cohort fill otherwise, reads the non-zero interactions out of the output array, and
        adds the empty interaction ``()`` with ``self.baseline_value``.

        Args:
            x: The instance to explain, as a 1-D array of length
                ``self.n_players``.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` carrying the
            requested interaction ``index``, ``max_order=self.max_order``, and
            ``min_order=1`` as the declared lower bound on computed orders.
        """
        from .cext import (
            fill_cohort,  # ty: ignore[unresolved-import]
            fill_single_reference,  # ty: ignore[unresolved-import]
            layout_to_dict,  # ty: ignore[unresolved-import]
        )

        # rounded the way the source library rounds prediction inputs
        x = self.tree[0].cast_input(np.asarray(x, dtype=np.float64)).reshape(-1)
        computation_index = get_computation_index(self.index)
        keys, counts = self._subset_tables
        out = np.zeros(self._n_outputs, dtype=np.float64)
        single = self.reference_data.shape[0] == 1
        fill = fill_single_reference if single else fill_cohort
        fill(
            *self._ensemble,
            self.reference_data[0] if single else self.reference_data,
            x,
            self._decision_type,
            computation_index,
            self.max_order,
            self.look_up_table,
            keys,
            counts,
            # without a hash index the fill searches the tables
            *(self._subset_index if self._subset_index is not None else (None, None)),
            out,
        )
        interactions = layout_to_dict(out, keys, counts, self.n_features, 1, None, True)  # noqa: FBT003
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
