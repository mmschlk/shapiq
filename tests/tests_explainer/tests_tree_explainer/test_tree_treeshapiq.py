"""This module contains all tests for the TreeExplainer class of the shapiq package."""
import numpy as np
import pytest

from shapiq.explainer.tree import TreeModel, TreeSHAPIQ


def test_init(dt_clf_model, background_clf_data):
    """Test the initialization of the TreeExplainer class."""
    explainer = TreeSHAPIQ(model=dt_clf_model, max_order=1, interaction_type="SII", verbose=True)

    x_explain = background_clf_data[0]
    _ = explainer.explain(x_explain)

    explainer = TreeSHAPIQ(model=dt_clf_model, max_order=1, interaction_type="k-SII")
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
    explainer = TreeSHAPIQ(model=tree_model, max_order=1, interaction_type="SII")
    x_explain = np.asarray([-1, -0.5, 1, 0])
    _ = explainer.explain(x_explain)

    assert True


@pytest.mark.parametrize(
    "index, expected",
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
            "BZF",
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
            "FSI",
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
            "STI",
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
def test_manual_tree(index: str, expected: dict):
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

    explainer = TreeSHAPIQ(model=tree_model, max_order=2, interaction_type=index)

    explanation = explainer.explain(x_explain)
    print(explanation)

    for key, value in expected.items():
        assert np.isclose(explanation[key], value, atol=1e-5)
