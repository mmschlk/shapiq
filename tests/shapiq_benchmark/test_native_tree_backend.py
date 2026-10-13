"""Frozen numeric trees retain routing without a second benchmark traversal kernel."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.tree.base import TreeModel, predict_ensemble
from shapiq_benchmark.structured import FrozenTreePredictor, _tree_game


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
def test_frozen_predictor_routing(precision, decision):
    trees = [tree(precision=precision, decision=decision)]
    predictor = FrozenTreePredictor(trees)
    rows = np.array([[0.0, -1.0], [1 - 2e-8, 0.0], [1.0, 1.0], [np.nan, np.nan]])
    for matrix in [rows, rows[::-1], rows[:1], rows[:0]]:
        np.testing.assert_array_equal(predictor.predict(matrix), predict_ensemble(trees, matrix))


@pytest.mark.parametrize("kind", ["tree", "pathdependent_tree"])
def test_structured_oracle_is_batch_independent(kind):
    trees = [tree(), tree(value=-0.25)]
    point = np.array([1 + 2e-8, -1.0])
    background = np.random.default_rng(4).normal(size=(16, 2))
    game = _tree_game(trees, background, point, kind)
    coalitions = np.array([[False, False], [True, False], [False, True], [True, True]])
    expected = game(coalitions)
    np.testing.assert_array_equal(game(coalitions[::-1])[::-1], expected)
    np.testing.assert_array_equal(np.concatenate([game(row[None]) for row in coalitions]), expected)
