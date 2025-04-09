"""This test module collects all tests for the conversions of the supported tree models for the
TreeExplainer class.
"""

import numpy as np
import pytest

from shapiq import TreeExplainer
from shapiq.explainer.tree.base import TreeModel
from shapiq.explainer.tree.conversion.edges import create_edge_tree
from shapiq.explainer.tree.conversion.sklearn import (
    convert_sklearn_forest,
    convert_sklearn_isolation_forest,
    convert_sklearn_tree,
)
from shapiq.explainer.tree.validation import SUPPORTED_MODELS
from shapiq.explainer.utils import get_predict_function_and_model_type
from shapiq.utils import safe_isinstance
from tests.conftest import TREE_MODEL_FIXTURES


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


def test_edge_tree_init():
    """Tests the initialization of the EdgeTree class."""
    # setup test data (same as in test_manual_tree of test_tree_treeshapiq.py)
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

    max_feature_id = tree_model.max_feature_id
    n_nodes = tree_model.n_nodes

    interaction_update_positions = {
        1: {0: np.array([0]), 1: np.array([1]), 2: np.array([2])},
        2: {0: np.array([0, 1]), 1: np.array([0, 2]), 2: np.array([1, 2])},
    }

    edge_tree = create_edge_tree(
        children_left=tree_model.children_left,
        children_right=tree_model.children_right,
        features=tree_model.features,
        node_sample_weight=tree_model.node_sample_weight,
        values=tree_model.values,
        max_interaction=1,
        n_features=max_feature_id + 1,
        n_nodes=n_nodes,
        subset_updates_pos_store=interaction_update_positions,
    )

    assert safe_isinstance(edge_tree, ["shapiq.explainer.tree.base.EdgeTree"])


def test_sklean_dt_conversion(dt_reg_model, dt_clf_model):
    """Test the conversion of a scikit-learn decision tree model."""
    # test regression model
    tree_model_class_path_str = ["shapiq.explainer.tree.base.TreeModel"]
    tree_model = convert_sklearn_tree(dt_reg_model)
    assert safe_isinstance(tree_model, tree_model_class_path_str)
    assert tree_model.empty_prediction is not None

    # test scaling
    tree_model = convert_sklearn_tree(dt_reg_model, scaling=0.5)
    assert safe_isinstance(tree_model, tree_model_class_path_str)
    assert tree_model.empty_prediction is not None

    # test classification model with class label
    tree_model = convert_sklearn_tree(dt_clf_model, class_label=0)
    assert safe_isinstance(tree_model, tree_model_class_path_str)
    assert tree_model.empty_prediction is not None

    # test classification model without class label
    tree_model = convert_sklearn_tree(dt_clf_model)
    assert safe_isinstance(tree_model, tree_model_class_path_str)
    assert tree_model.empty_prediction is not None


def test_skleanr_rf_conversion(rf_clf_model, rf_reg_model):
    """Test the conversion of a scikit-learn random forest model."""
    tree_model_class_path_str = ["shapiq.explainer.tree.base.TreeModel"]

    # test the regression model
    tree_model = convert_sklearn_forest(rf_reg_model)
    assert isinstance(tree_model, list)
    assert safe_isinstance(tree_model[0], tree_model_class_path_str)
    assert tree_model[0].empty_prediction is not None

    # test the classification model
    tree_model = convert_sklearn_forest(rf_clf_model)
    assert isinstance(tree_model, list)
    assert safe_isinstance(tree_model[0], tree_model_class_path_str)
    assert tree_model[0].empty_prediction is not None


def test_sklearn_if_conversion(if_clf_model):
    """Test the conversion of a scikit-learn isolation forest model."""
    tree_model_class_path_str = ["shapiq.explainer.tree.base.TreeModel"]

    # test the isolation forest model
    tree_model = convert_sklearn_isolation_forest(if_clf_model)
    assert isinstance(tree_model, list)
    assert safe_isinstance(tree_model[0], tree_model_class_path_str)
    assert tree_model[0].empty_prediction is not None


@pytest.mark.external_libraries
@pytest.mark.parametrize("model_fixture, model_class", TREE_MODEL_FIXTURES)
def test_conversion_predict_identity(model_fixture, model_class, background_reg_data, request):
    if model_class not in SUPPORTED_MODELS:
        pytest.skip(
            f"skipped test, {model_class} not in the supported models for the tree explainer.",
        )
    else:
        model = request.getfixturevalue(model_fixture)
        predict_function, _ = get_predict_function_and_model_type(model, model_class)
        original_pred = predict_function(model, background_reg_data)
        tree_explainer = TreeExplainer(model=model, max_order=1, min_order=1, index="SV")
        for index in range(len(background_reg_data)):
            sv = tree_explainer.explain(background_reg_data[index])
            prediction = sum(sv.values)
            if sv[()] == 0:
                prediction += sv.baseline_value
            original_pred_value = original_pred[index]
            if pytest.approx(prediction, abs=1e-4) == original_pred_value:
                assert True
            else:
                if "xgb" or "lightgbm" in model_fixture:
                    # xgboost sometimes predicts a different value
                    # see .test_tree_bugfix.test_xgb_predicts_with_wrong_leaf_node
                    # TODO: take a look at this in more detail, why is it hard to get efficiency
                    continue
                raise AssertionError("Prediction does not match the original prediction.")


def test_tree_model_predict(
    background_reg_dataset,
    dt_reg_model,
    background_clf_dataset,
    dt_clf_model,
):
    """Tests weather the tree model predict_one is correct."""
    X_reg, _ = background_reg_dataset
    X_clf, _ = background_clf_dataset
    predictions_reg = dt_reg_model.predict(X_reg)
    predictions_clf = dt_clf_model.predict_proba(X_clf)[:, 0]

    # convert
    tree_model_reg = convert_sklearn_tree(dt_reg_model)
    tree_model_clf = convert_sklearn_tree(dt_clf_model, class_label=0)

    # test prediction
    for i in range(len(predictions_reg)):
        assert tree_model_reg.predict_one(X_reg[i]) == predictions_reg[i]

    # test prediction
    for i in range(len(predictions_clf)):
        assert tree_model_clf.predict_one(X_clf[i]) == predictions_clf[i]
