"""Parity tests for the experimental multi-output XGBoost tree converter.

These tests fit an ``XGBRegressor(multi_strategy="multi_output_tree")`` on a
synthetic multivariate target and assert that the pure-Python
:class:`MultiOutputTreeModel` reproduces ``model.predict``.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

xgboost = pytest.importorskip("xgboost")

from shapiq.approximator.proxy._multioutput.tree import (  # noqa: E402
    MultiOutputTreeModel,
    convert_multioutput_xgboost,
    convert_multioutput_xgboost_with_base_score,
    predict_multioutput,
)


def _make_multioutput_target(n_features: int, n_classes: int, random_state: int):
    """Build a synthetic ``(n_samples, n_classes)`` regression target.

    The target is the one-hot encoding of a ``make_classification`` label matrix,
    giving a genuinely multivariate value function for the multi-output tree.
    """
    n_samples = 400
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 2),
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), y] = 1.0
    return X.astype(np.float64), one_hot


@pytest.mark.parametrize(
    ("n_features", "n_classes"),
    [(6, 4), (10, 8)],
)
def test_multioutput_tree_predict_parity(n_features: int, n_classes: int) -> None:
    """Converted trees plus base score reproduce ``model.predict``."""
    X, Y = _make_multioutput_target(n_features, n_classes, random_state=0)

    model = xgboost.XGBRegressor(
        multi_strategy="multi_output_tree",
        n_estimators=8,
        max_depth=4,
        random_state=0,
    )
    model.fit(X, Y)

    trees, base_score = convert_multioutput_xgboost_with_base_score(model)

    # structural sanity checks on the container layout.
    assert len(trees) == 8
    assert base_score.shape == (n_classes,)
    for tree in trees:
        assert isinstance(tree, MultiOutputTreeModel)
        assert tree.n_outputs == n_classes
        assert tree.values.shape == (tree.n_nodes, n_classes)
        # internal-node value rows are exactly zero.
        assert np.all(tree.values[~tree.leaf_mask] == 0.0)
        # leaf rows have -1 children and -2 feature sentinels.
        assert np.all(tree.children_left[tree.leaf_mask] == -1)
        assert np.all(tree.children_right[tree.leaf_mask] == -1)
        assert np.all(tree.features[tree.leaf_mask] == -2)
        assert np.all(np.isnan(tree.thresholds[tree.leaf_mask]))

    reference = model.predict(X)
    predicted = predict_multioutput(model, X)
    assert predicted.shape == reference.shape == (X.shape[0], n_classes)
    np.testing.assert_allclose(predicted, reference, atol=1e-4)


def test_convert_multioutput_xgboost_returns_trees() -> None:
    """The plain ``convert_multioutput_xgboost`` entry point returns the tree list."""
    X, Y = _make_multioutput_target(6, 4, random_state=1)
    model = xgboost.XGBRegressor(
        multi_strategy="multi_output_tree",
        n_estimators=5,
        max_depth=3,
        random_state=1,
    )
    model.fit(X, Y)

    trees = convert_multioutput_xgboost(model)
    assert len(trees) == 5
    assert all(isinstance(t, MultiOutputTreeModel) for t in trees)

    # predict_one of a single tree matches predict on a length-1 batch.
    tree = trees[0]
    single = tree.predict_one(X[0])
    batched = tree.predict(X[:1])
    np.testing.assert_allclose(single, batched[0])
