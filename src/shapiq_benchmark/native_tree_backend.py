"""Faster traversal of frozen numeric tree oracles; values and query counts are unchanged."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from shapiq.tree.base import predict_ensemble
from shapiq.tree.interventional.cext import predict_ensemble_sum  # ty: ignore[unresolved-import]
from shapiq_games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI

if TYPE_CHECKING:
    from shapiq.tree.base import TreeModel


class NumericTreePredictor:
    """Use the existing row-major kernel when its arithmetic matches NumPy's tree sum."""

    def __init__(self, trees: list[TreeModel]) -> None:
        """Retain frozen trees and prepare compatible numeric kernel arguments."""
        self.trees = trees
        self.supported = len(
            {(tree.input_precision, tree.decision_type) for tree in trees}
        ) == 1 and all(self._supported(tree) for tree in trees)
        self.maximum_feature = max(
            (int(np.max(tree.features[~tree.leaf_mask], initial=-1)) for tree in trees), default=-1
        )
        self.arguments = None
        if self.supported:
            self.arguments = tuple(
                [
                    np.ascontiguousarray(getattr(tree, name), dtype=dtype).reshape(-1)
                    for tree in trees
                ]
                for name, dtype in (
                    ("values", np.float64),
                    ("thresholds", np.float64),
                    ("features", np.int64),
                    ("children_left", np.int64),
                    ("children_right", np.int64),
                    ("children_left_default", np.bool_),
                )
            )

    @staticmethod
    def _supported(tree: TreeModel) -> bool | np.bool_:
        leaves = np.asarray(tree.leaf_mask, dtype=bool)
        branches = ~leaves
        return (
            tree.root_node_id == 0
            and not tree.has_categorical
            and tree.decision_type in ("<", "<=")
            and tree.input_precision in ("float32", "float64")
            and all(k == v for k, v in tree.feature_map_internal_original.items())
            and np.array_equal(leaves, tree.children_left == tree.children_right)
            and np.all(np.isfinite(tree.values[leaves]))
            and np.all(np.isfinite(tree.thresholds[branches]))
            and np.all(tree.features[branches] >= 0)
            and np.all(
                (tree.children_left[branches] >= 0) & (tree.children_left[branches] < tree.n_nodes)
            )
            and np.all(
                (tree.children_right[branches] >= 0)
                & (tree.children_right[branches] < tree.n_nodes)
            )
        )

    def predict(self, rows: np.ndarray) -> np.ndarray:
        """Predict the same scalar outputs, retaining original unsupported-tree behavior."""
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2:
            message = "Frozen prediction requires a two-dimensional row matrix"
            raise ValueError(message)
        # A single row changes NumPy's reduction grouping. Use its original implementation.
        if not self.supported or len(rows) <= 1 or self.maximum_feature >= rows.shape[1]:
            return predict_ensemble(self.trees, rows)
        tree = self.trees[0]
        return predict_ensemble_sum(
            *cast("tuple[list[np.ndarray], ...]", self.arguments),
            tree.cast_input(rows),
            tree.decision_type,
        )


class BatchedPathDependentGame(TreeSHAPIQXAI):
    """Vectorize coalition rows, preserving the shipped recursion's operation order."""

    def __init__(self, original: TreeSHAPIQXAI, chunk_size: int = 1024) -> None:
        """Reuse the original game state with bounded temporary coalition arrays."""
        if chunk_size < 1:
            message = "Positive bounded chunk size required"
            raise ValueError(message)
        self.__dict__ = original.__dict__.copy()
        self.chunk_size = chunk_size
        self.supported = all(self._supported(tree) for tree in self._trees)

    @staticmethod
    def _supported(tree: TreeModel) -> bool | np.bool_:
        branches = ~tree.leaf_mask
        left, right = tree.children_left[branches], tree.children_right[branches]
        weights = tree.node_sample_weight
        return (
            tree.root_node_id == 0
            and tree.nodes[0] == 0
            and not tree.has_categorical
            and all(k == v for k, v in tree.feature_map_internal_original.items())
            and tree.values.dtype == np.float64
            and weights.dtype == np.float64
            and np.all(np.isfinite(tree.values))
            and np.all(np.isfinite(weights))
            and np.all(weights >= 0)
            and np.all(np.isfinite(weights[left] + weights[right]))
            and np.all(weights[left] + weights[right] > 0)
        )

    def _prediction(self, tree: TreeModel, node: int, coalitions: np.ndarray) -> np.ndarray:
        if tree.leaf_mask[node]:
            return np.full(len(coalitions), tree.values[node], dtype=np.float64)
        feature = tree.features[node]
        left, right = tree.children_left[node], tree.children_right[node]
        left_prediction = self._prediction(tree, left, coalitions)
        right_prediction = self._prediction(tree, right, coalitions)
        known = (
            left_prediction if tree.goes_left(node, self.x_explain[feature]) else right_prediction
        )
        left_weight, right_weight = tree.node_sample_weight[left], tree.node_sample_weight[right]
        total = left_weight + right_weight
        unknown = left_prediction * (left_weight / total) + right_prediction * (
            right_weight / total
        )
        return np.where(coalitions[:, feature], known, unknown)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate coalitions using the original recursive arithmetic and baseline."""
        if not self.supported:
            return super().value_function(coalitions)
        values = np.zeros(len(coalitions), dtype=np.float64)
        for start in range(0, len(coalitions), self.chunk_size):
            stop = min(len(coalitions), start + self.chunk_size)
            rows = coalitions[start:stop]
            for tree in self._trees:
                values[start:stop] += self._prediction(tree, tree.nodes[0], rows)
        values[~coalitions.any(axis=1)] = self.empty_value
        return values
