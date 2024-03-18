"""This test module collects all tests for the conversions of the supported tree models for the
TreeExplainer class."""

import numpy as np

from shapiq import safe_isinstance
from shapiq.explainer.tree.base import TreeModel
from explainer.tree.conversion.sklearn import convert_sklearn_tree


def test_tree_model_init():
    """Test the initialization of the TreeModel class."""
    left_children = np.array([1, 2, -1, -1, -1])
    right_children = np.array([4, 3, -1, -1, -1])
    features = np.array([0, 1, -2, -3, -2])  # intentionally wrong value at index 3
    thresholds = np.array([0.5, 0.5, 0, 0, 0])
    values = np.array([100, 200, 300, 400, 500])
    sample_weights = np.array([1.0, 0.5, 0.25, 0.25, 0.5])
    tree_model = TreeModel(
        children_left=left_children,
        children_right=right_children,
        features=features,
        thresholds=thresholds,
        values=values,
        node_sample_weight=sample_weights,
    )
    assert np.all(tree_model.children_left == np.array([1, 2, -1, -1, -1]))
    assert np.all(tree_model.children_right == np.array([4, 3, -1, -1, -1]))
    assert np.all(tree_model.features == np.array([0, 1, -2, -2, -2]))
    assert np.isnan(tree_model.thresholds[2])
    assert np.isnan(tree_model.thresholds[3])
    # check if is_leaf is correctly computed
    assert np.all(tree_model.leaf_mask == np.array([False, False, True, True, True]))
    # check if empty prediction is correctly computed
    assert tree_model.empty_prediction == 425.0
    # test __getitem__
    assert np.all(tree_model["children_left"] == np.array([1, 2, -1, -1, -1]))


def test_sklean_conversion(dt_reg_model, dt_clf_model):
    """Test the conversion of a scikit-learn decision tree model."""
    # test regression model
    class_path_str = ["explainer.tree.base.TreeModel"]
    tree_model = convert_sklearn_tree(dt_reg_model)
    assert safe_isinstance(tree_model, class_path_str)
    assert tree_model.empty_prediction is not None

    # test scaling
    tree_model = convert_sklearn_tree(dt_reg_model, scaling=0.5)
    assert safe_isinstance(tree_model, class_path_str)
    assert tree_model.empty_prediction is not None

    # test classification model with class label
    tree_model = convert_sklearn_tree(dt_clf_model, class_label=0)
    assert safe_isinstance(tree_model, class_path_str)
    assert tree_model.empty_prediction is not None

    # test classification model without class label
    tree_model = convert_sklearn_tree(dt_clf_model)
    assert safe_isinstance(tree_model, class_path_str)
    assert tree_model.empty_prediction is not None
