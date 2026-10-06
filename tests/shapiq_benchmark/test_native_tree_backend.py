"""Fast frozen-tree traversal preserves arithmetic, routing and fallback behavior."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from shapiq.tree.base import TreeModel, predict_ensemble
from shapiq.tree.interventional.game import InterventionalGame
from shapiq_benchmark.native_tree_backend import BatchedPathDependentGame, NumericTreePredictor
from shapiq_benchmark.structured import FrozenTreePredictor
from shapiq_games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI


def tree(*, precision="float64", decision="<=", value=1.5):
    return TreeModel(
        children_left=np.array([1, 3, 5, -1, -1, -1, -1]),
        children_right=np.array([2, 4, 6, -1, -1, -1, -1]),
        children_missing=np.array([2, 3, 6, -1, -1, -1, -1]),
        features=np.array([0, 1, 0, -2, -2, -2, -2]),
        thresholds=np.array([1.0, 0.0, 2.0, np.nan, np.nan, np.nan, np.nan]),
        values=np.array([0.0, 0.0, 0.0, value, -0.25, 2.25, -3.125]),
        node_sample_weight=np.array([13.0, 5.0, 8.0, 2.0, 3.0, 1.0, 7.0]),
        input_precision=precision,
        decision_type=decision,
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("decision", ["<", "<="])
def test_numeric_routing_and_singleton_sum(precision, decision):
    trees = [
        tree(precision=precision, decision=decision, value=value)
        for value in [1e16] + [1.0] * 14 + [-1e16]
    ]
    predictor = NumericTreePredictor(trees)
    assert predictor.supported
    rows = np.array(
        [
            [0.0, -1.0],
            [1 - 2e-8, 0.0],
            [1.0, -1.0],
            [1 + 2e-8, 1.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [np.nan, np.nan],
        ]
    )
    for matrix in [rows, rows[::-1], rows[:1], rows[:0]]:
        assert predictor.predict(matrix).tobytes() == predict_ensemble(trees, matrix).tobytes()
    # The singleton case must retain NumPy's distinct pairwise summation grouping.
    assert predictor.predict(rows[:1]).tobytes() == predict_ensemble(trees, rows[:1]).tobytes()


def test_unsupported_trees_use_original_predictor():
    rows = np.array([[0.0, 2.0], [np.nan, 0.0]])
    for change in ("root", "mapping", "category", "threshold", "mixed"):
        first = tree()
        trees = [first]
        if change == "root":
            first.root_node_id = 1
        elif change == "mapping":
            first.feature_map_internal_original = {0: 1, 1: 0}
        elif change == "category":
            first.has_categorical = True
        elif change == "threshold":
            first.thresholds[0] = np.inf
        else:
            trees.append(tree(precision="float32"))
        predictor = NumericTreePredictor(trees)
        assert not predictor.supported
        assert predictor.predict(rows).tobytes() == predict_ensemble(trees, rows).tobytes()


def test_shipped_interventional_game_and_predictor_api():
    trees = [tree(value=value) for value in [1e16] + [1.0] * 14 + [-1e16]]

    class Original:
        def predict(self, rows):
            return predict_ensemble(trees, rows)

    point = np.array([0.0, -1.0])
    coalitions = np.array([[False, False], [True, False], [False, True], [True, True]])
    for size in [1, 2, 3, 16, 17]:
        background = np.random.default_rng(4).normal(size=(size, 2))
        old = InterventionalGame(Original(), background, point)
        predictor = FrozenTreePredictor(trees)
        assert predictor.trees is trees
        new = InterventionalGame(predictor, background, point)
        assert new(coalitions).tobytes() == old(coalitions).tobytes()
        assert new(coalitions[::-1])[::-1].tobytes() == old(coalitions).tobytes()


def test_pathdependent_repeated_features_and_baseline():
    rows = np.array([[False, False], [True, False], [False, True], [True, True]])
    for point in [np.array([1 + 2e-8, -1.0]), np.array([np.nan, 0.0])]:
        old = TreeSHAPIQXAI(
            point, [tree(), tree(precision="float32")], normalize=False, verbose=False
        )
        old.empty_value = 1.23456789
        for chunk in [1, 3, 7]:
            new = BatchedPathDependentGame(old, chunk)
            assert new.supported
            assert new(rows).tobytes() == old(rows).tobytes()
            assert new(rows[::-1])[::-1].tobytes() == old(rows).tobytes()
            assert np.concatenate([new(row[None]) for row in rows]).tobytes() == old(rows).tobytes()
        fallback = copy.deepcopy(old)
        fallback._trees[0].values = fallback._trees[0].values.astype(np.float32)
        new = BatchedPathDependentGame(fallback)
        assert not new.supported
        assert new(rows).tobytes() == fallback(rows).tobytes()
