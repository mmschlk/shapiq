"""This module contains all tests for the TreeExplainer class of the shapiq package."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.explainer.tree import TreeExplainer, TreeModel, TreeSHAPIQ


def test_init(dt_clf_model, background_clf_data):
    """Test the initialization of the TreeExplainer class."""
    explainer = TreeSHAPIQ(model=dt_clf_model, max_order=1, index="SII", verbose=True)

    x_explain = background_clf_data[0]
    _ = explainer.explain(x_explain)

    explainer = TreeSHAPIQ(model=dt_clf_model, max_order=1, index="k-SII")
    x_explain = background_clf_data[0]
    _ = explainer.explain(x_explain)

    # test with dict input as tree
    tree_model = {
        "children_left": np.asarray([1, 2, 3, -1, -1, -1, 7, -1, -1]),
        "children_right": np.asarray([6, 5, 4, -1, -1, -1, 8, -1, -1]),
        "features": np.asarray([0, 1, 0, -2, -2, -2, 2, -2, -2]),
        "thresholds": np.asarray([0, 0, -0.5, -2, -2, -2, 0, -2, -2]),
        "node_sample_weight": np.asarray([100, 50, 38, 15, 23, 12, 50, 20, 30]),
        "values": np.asarray([110, 105, 95, 20, 50, 100, 75, 10, 40]),
    }
    explainer = TreeSHAPIQ(model=tree_model, max_order=1, index="SII")
    x_explain = np.asarray([-1, -0.5, 1, 0])
    _ = explainer.explain(x_explain)

    assert True


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (
            "SII",
            {
                (0,): -10.18947368,
                (1,): -13.31052632,
                (2,): 3.0,
                (0, 1): -11.77894737,
                (0, 2): -6.0,
                (1, 2): 0,
            },
        ),
        (
            "BII",
            {
                (0,): -10.18947368,
                (1,): -13.31052632,
                (2,): 3.0,
                (0, 1): -11.77894737,
                (0, 2): -6.0,
                (1, 2): 0,
            },
        ),
        (
            "FSII",
            {
                (0,): -39.45789474,
                (1,): -45.82105263,
                (2,): 6.0,
                (0, 1): -11.77894737,
                (0, 2): -6.0,
                (1, 2): 0,
            },
        ),
        (
            "STII",
            {
                (0,): -20.37894737,
                (1,): -26.62105263,
                (2,): 6.0,
                (0, 1): -11.77894737,
                (0, 2): -6.0,
                (1, 2): 0,
            },
        ),
    ],
)
def test_against_old_treeshapiq_implementation(index: str, expected: dict):
    """Test the tree explainer against the old TreeSHAP-IQ implementation's results."""
    # manual values for a tree to test against the original treeshapiq implementation
    children_left = np.asarray([1, 2, 3, -1, -1, -1, 7, -1, -1])
    children_right = np.asarray([6, 5, 4, -1, -1, -1, 8, -1, -1])
    features = np.asarray([0, 1, 0, -2, -2, -2, 2, -2, -2])
    thresholds = np.asarray([0, 0, -0.5, -2, -2, -2, 0, -2, -2])
    node_sample_weight = np.asarray([100, 50, 38, 15, 23, 12, 50, 20, 30])
    values = np.asarray([110, 105, 95, 20, 50, 100, 75, 10, 40])

    x_explain = np.asarray([-1, -0.5, 1, 0])

    tree_model = TreeModel(
        children_left=children_left,
        children_right=children_right,
        features=features,
        thresholds=thresholds,
        node_sample_weight=node_sample_weight,
        values=values,
    )

    explainer = TreeSHAPIQ(model=tree_model, max_order=2, index=index)

    explanation = explainer.explain(x_explain)

    for key, value in expected.items():
        assert np.isclose(explanation[key], value, atol=1e-5)


def test_edge_case_params():
    """Test the TreeSHAPIQ class with edge case parameters."""
    children_left = np.asarray([1, 2, 3, -1, -1, -1, 7, -1, -1])
    children_right = np.asarray([6, 5, 4, -1, -1, -1, 8, -1, -1])
    features = np.asarray([0, 1, 0, -2, -2, -2, 2, -2, -2])
    thresholds = np.asarray([0, 0, -0.5, -2, -2, -2, 0, -2, -2])
    node_sample_weight = np.asarray([100, 50, 38, 15, 23, 12, 50, 20, 30])
    values = np.asarray([110, 105, 95, 20, 50, 100, 75, 10, 40])

    tree_model = TreeModel(
        children_left=children_left,
        children_right=children_right,
        features=features,
        thresholds=thresholds,
        node_sample_weight=node_sample_weight,
        values=values,
    )

    # test with max_order = 0
    with pytest.raises(ValueError):
        _ = TreeSHAPIQ(model=tree_model, max_order=0)


def test_no_bug_with_one_feature_tree():
    """Test that the TreeExplainer does not raise an error with a tree that has only one feature."""
    # create the dataset
    X = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 2],
            [2, 1, 1, 1],
            [3, 2, 1, 1],
        ],
    )

    # Define simple one feature tree
    tree = {
        "children_left": np.array([1, -1, -1]),
        "children_right": np.array([2, -1, -1]),
        "features": np.array([0, -2, -2]),
        "thresholds": np.array([2.5, -2, -2]),
        "values": np.array([0.5, 0.0, 1]),
        "node_sample_weight": np.array([14, 5, 9]),
    }
    tree = TreeModel(**tree)
    explainer = TreeExplainer(model=tree, index="SV", max_order=1)
    explainer.explain(X[2])
