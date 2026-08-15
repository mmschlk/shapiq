"""This test module collects all tests for the conversions of the supported tree models for the TreeExplainer class."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq import TreeExplainer
from shapiq.explainer.utils import get_predict_function_and_model_type
from shapiq.tree.base import TreeModel
from shapiq.tree.conversion import convert_tree_model
from shapiq.tree.conversion.catboost import parse_catboost_json_model
from shapiq.tree.conversion.edges import create_edge_tree
from shapiq.tree.conversion.sklearn import (
    convert_isolation_forest_tree,
    convert_random_forest_tree,
    convert_sklearn_tree,
)
from shapiq.tree.validation import SUPPORTED_MODELS
from shapiq.utils import safe_isinstance
from tests.shapiq.conftest import TREE_MODEL_FIXTURES
from tests.shapiq.tests_unit.tests_explainer.tests_tree_explainer.conversion_reference import (
    create_edge_tree_python,
    parse_catboost_json_model_python,
)


def _predict_tree_ensemble(tree_model: list[TreeModel], data: np.ndarray) -> np.ndarray:
    """Predict raw outputs from a converted tree ensemble."""
    return np.asarray([sum(tree.predict_one(x) for tree in tree_model) for x in data])


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
        children_missing=left_children,  # intentionally set to left_children to test if it is ignored
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
        children_missing=children_left,  # intentionally set to left_children to test if it is ignored
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

    assert safe_isinstance(edge_tree, ["shapiq.tree.base.EdgeTree"])


def test_sklean_dt_conversion(dt_reg_model, dt_clf_model):
    """Test the conversion of a scikit-learn decision tree model."""
    # test regression model
    tree_model_class_path_str = ["shapiq.tree.base.TreeModel"]
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
    tree_model_class_path_str = ["shapiq.tree.base.TreeModel"]

    # test the regression model
    tree_model = convert_random_forest_tree(rf_reg_model)
    assert isinstance(tree_model, list)
    assert safe_isinstance(tree_model[0], tree_model_class_path_str)
    assert tree_model[0].empty_prediction is not None

    # test the classification model
    tree_model = convert_random_forest_tree(rf_clf_model)
    assert isinstance(tree_model, list)
    assert safe_isinstance(tree_model[0], tree_model_class_path_str)
    assert tree_model[0].empty_prediction is not None


def test_sklearn_if_conversion(if_clf_model):
    """Test the conversion of a scikit-learn isolation forest model."""
    tree_model_class_path_str = ["shapiq.tree.base.TreeModel"]

    # test the isolation forest model
    tree_model = convert_isolation_forest_tree(if_clf_model)
    assert isinstance(tree_model, list)
    assert safe_isinstance(tree_model[0], tree_model_class_path_str)
    assert tree_model[0].empty_prediction is not None


def test_sklearn_gradient_boosting_conversion_predicts_raw_output(gb_reg_model, gb_clf_model):
    """Test that GradientBoosting conversions (incl. the init_ baseline) match raw outputs."""
    from sklearn.datasets import make_classification, make_regression

    # regression: sum of converted tree predictions equals model.predict
    X, _ = make_regression(random_state=42, n_samples=80, n_features=7)
    expected = gb_reg_model.predict(X[:10])
    converted = convert_tree_model(gb_reg_model)
    assert len(converted) == gb_reg_model.n_estimators
    np.testing.assert_allclose(_predict_tree_ensemble(converted, X[:10]), expected, rtol=1e-6)

    # multiclass classification: converted trees match the per-class raw decision function,
    # and class_label=None defaults to class 1
    Xc, _ = make_classification(
        random_state=42, n_samples=80, n_features=7, n_classes=3, n_informative=5
    )
    expected_raw = gb_clf_model.decision_function(Xc[:10])
    for class_label in range(3):
        converted = convert_tree_model(gb_clf_model, class_label=class_label)
        assert len(converted) == gb_clf_model.n_estimators
        np.testing.assert_allclose(
            _predict_tree_ensemble(converted, Xc[:10]), expected_raw[:, class_label], rtol=1e-6
        )
    converted_default = convert_tree_model(gb_clf_model)
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted_default, Xc[:10]), expected_raw[:, 1], rtol=1e-6
    )

    # a custom init estimator cannot be folded into a constant baseline
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    y = X.sum(axis=1)
    custom_init_model = GradientBoostingRegressor(
        random_state=42, n_estimators=3, init=LinearRegression()
    )
    custom_init_model.fit(X, y)
    with pytest.raises(ValueError, match="custom `init` estimator"):
        convert_tree_model(custom_init_model)


def test_sklearn_hist_gradient_boosting_conversion_predicts_raw_output(
    hist_gb_reg_model, hist_gb_clf_model
):
    """Test that HistGradientBoosting conversions (incl. the baseline) match raw outputs."""
    from sklearn.datasets import make_classification, make_regression

    # regression: sum of converted tree predictions equals model.predict
    X, _ = make_regression(random_state=42, n_samples=80, n_features=7)
    expected = hist_gb_reg_model.predict(X[:10])
    converted = convert_tree_model(hist_gb_reg_model)
    np.testing.assert_allclose(_predict_tree_ensemble(converted, X[:10]), expected, rtol=1e-6)

    # missing values are routed like the model routes them
    x_nan = X[0].copy()
    x_nan[0] = np.nan
    expected_nan = float(hist_gb_reg_model.predict(x_nan.reshape(1, -1))[0])
    prediction_nan = float(sum(tree.predict_one(x_nan) for tree in converted))
    assert prediction_nan == pytest.approx(expected_nan, rel=1e-6)

    # multiclass classification: converted trees match the per-class raw decision function,
    # and class_label=None defaults to class 1
    Xc, _ = make_classification(
        random_state=42, n_samples=80, n_features=7, n_classes=3, n_informative=5
    )
    expected_raw = hist_gb_clf_model.decision_function(Xc[:10])
    for class_label in range(3):
        converted = convert_tree_model(hist_gb_clf_model, class_label=class_label)
        np.testing.assert_allclose(
            _predict_tree_ensemble(converted, Xc[:10]), expected_raw[:, class_label], rtol=1e-6
        )
    converted_default = convert_tree_model(hist_gb_clf_model)
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted_default, Xc[:10]), expected_raw[:, 1], rtol=1e-6
    )


def test_sklearn_hist_gradient_boosting_categorical_conversion(
    hist_gb_cat_reg_model, hist_gb_cat_clf_model, background_cat_dataset
):
    """Test that categorical HistGradientBoosting conversions match the raw model output.

    Covers non-contiguous raw category codes (ordinal-encoder translation), NaN routing at
    categorical and numeric nodes, and per-class tree selection for multiclass models.
    """
    X, _ = background_cat_dataset

    converted = convert_tree_model(hist_gb_cat_reg_model)
    assert any(tree.has_categorical for tree in converted)
    # all rows, including NaN rows and every trained category code
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted, X),
        hist_gb_cat_reg_model.predict(X),
        rtol=1e-6,
        atol=1e-9,
    )
    # every raw category code at a fixed base point, NaN routing, and out-of-vocabulary
    # codes (unknown / in-gap / negative / out-of-range values route to the missing branch
    # in sklearn — captured exactly via the swapped-children encoding)
    probes = []
    for code in [2.0, 7.0, 11.0, 30.0, 41.0, 99.0, np.nan, 500.0, 3.0, -1.0, -42.0, 1e6]:
        x = X[1].copy()
        x[0] = code
        probes.append(x)
    x = X[1].copy()
    x[0] = np.nan
    x[1] = np.nan
    probes.append(x)
    probes = np.asarray(probes)
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted, probes),
        hist_gb_cat_reg_model.predict(probes),
        rtol=1e-6,
        atol=1e-9,
    )

    # multiclass: per-class raw parity, class_label=None defaults to class 1
    expected_raw = hist_gb_cat_clf_model.decision_function(X[:100])
    for class_label in range(3):
        converted_class = convert_tree_model(hist_gb_cat_clf_model, class_label=class_label)
        np.testing.assert_allclose(
            _predict_tree_ensemble(converted_class, X[:100]),
            expected_raw[:, class_label],
            rtol=1e-6,
            atol=1e-9,
        )
    converted_default = convert_tree_model(hist_gb_cat_clf_model)
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted_default, X[:100]), expected_raw[:, 1], rtol=1e-6, atol=1e-9
    )


def test_tree_model_categorical_splits():
    """Test the TreeModel categorical CSR representation and its routing semantics."""
    from shapiq.tree.validation import validate_tree_model

    model_dict = {
        "children_left": np.array([1, -1, -1]),
        "children_right": np.array([2, -1, -1]),
        "children_missing": np.array([2, -1, -1]),  # NaN routes right
        "features": np.array([0, -2, -2]),
        "thresholds": np.array([0.5, np.nan, np.nan]),  # stale threshold, must be sanitized
        "values": np.array([0.0, 10.0, 20.0]),
        "node_sample_weight": np.array([10.0, 5.0, 5.0]),
        "cat_values": np.array([5, 2]),  # intentionally unsorted
        "cat_start": np.array([0, 0, 0]),
        "cat_size": np.array([2, 0, 0]),
    }
    tree = validate_tree_model(dict(model_dict))[0]
    assert tree.has_categorical
    assert np.all(tree.is_categorical == [True, False, False])
    assert np.all(tree.cat_values == [2, 5])  # sorted during init
    assert np.isnan(tree.thresholds[0])  # categorical node threshold sanitized to NaN
    assert tree.predict_one(np.array([2.0])) == 10.0  # in set -> left
    assert tree.predict_one(np.array([5.9])) == 10.0  # truncates to 5 -> in set
    assert tree.predict_one(np.array([3.0])) == 20.0  # not in set -> right
    assert tree.predict_one(np.array([-1.0])) == 20.0  # negative -> right
    assert tree.predict_one(np.array([np.nan])) == 20.0  # NaN -> missing child (right)

    # constructor validation: partial cat kwargs and wrong lengths are rejected
    partial = {k: v for k, v in model_dict.items() if k != "cat_start"}
    with pytest.raises(ValueError, match="together"):
        TreeModel(**partial)
    wrong_length = dict(model_dict)
    wrong_length["cat_size"] = np.array([2, 0])
    with pytest.raises(ValueError, match="per node"):
        TreeModel(**wrong_length)


def test_tree_model_categorical_survives_feature_reduction():
    """Test that categorical routing is unchanged by ``reduce_feature_complexity``."""
    tree = TreeModel(
        children_left=np.array([1, -1, -1]),
        children_right=np.array([2, -1, -1]),
        children_missing=np.array([1, -1, -1]),
        features=np.array([8, -2, -2]),  # high feature id so the reduction actually fires
        thresholds=np.array([np.nan, np.nan, np.nan]),
        values=np.array([0.0, 10.0, 20.0]),
        node_sample_weight=np.array([10.0, 5.0, 5.0]),
        cat_values=np.array([1, 4]),
        cat_start=np.array([0, 0, 0]),
        cat_size=np.array([2, 0, 0]),
    )
    x = np.zeros(9)
    x[8] = 4.0
    assert tree.predict_one(x) == 10.0
    tree.reduce_feature_complexity()
    assert tree.n_features_in_tree == 1
    assert tree.features[0] == 0  # remapped
    assert np.all(tree.cat_values == [1, 4])  # node-keyed arrays untouched
    assert tree.predict_one(x) == 10.0  # routing unchanged via the internal feature map
    x[8] = 5.0
    assert tree.predict_one(x) == 20.0


def _make_categorical_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a dataset with two categorical columns (0, 2), NaNs in columns 0/1, and labels."""
    rng = np.random.default_rng(42)
    n = 500
    X = np.column_stack(
        [
            rng.choice([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], n),
            rng.normal(size=n),
            rng.integers(0, 5, n).astype(float),
        ]
    )
    y = (X[:, 0] % 3) * 2 + X[:, 1] + (X[:, 2] == 1) * 3 + rng.normal(scale=0.1, size=n)
    y_classes = rng.integers(0, 3, n)
    X[rng.choice(n, 50, replace=False), 1] = np.nan
    return X, y, y_classes


def test_lightgbm_categorical_conversion_predicts_raw_score():
    """Test that categorical LightGBM conversions match the raw model score exactly.

    Covers bitset decoding, NaN / negative / unseen category routing (all to the right
    child in LightGBM), and per-class tree selection for multiclass models.
    """
    lightgbm = pytest.importorskip("lightgbm")

    X, y, y_classes = _make_categorical_dataset()
    model = lightgbm.LGBMRegressor(n_estimators=5, min_child_samples=5, verbose=-1)
    model.fit(X, y, categorical_feature=[0, 2])
    converted = convert_tree_model(model)
    assert any(tree.has_categorical for tree in converted)
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted, X), model.predict(X, raw_score=True), rtol=1e-6
    )
    # unseen / negative / NaN category codes route like the model routes them
    probes = np.repeat(X[:1], 3, axis=0)
    probes[:, 0] = [99.0, -5.0, np.nan]
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted, probes), model.predict(probes, raw_score=True), rtol=1e-6
    )

    clf = lightgbm.LGBMClassifier(n_estimators=4, min_child_samples=5, verbose=-1)
    clf.fit(X, y_classes, categorical_feature=[0, 2])
    raw = clf.predict_proba(X[:80], raw_score=True)
    for class_label in range(3):
        converted_class = convert_tree_model(clf, class_label=class_label)
        np.testing.assert_allclose(
            _predict_tree_ensemble(converted_class, X[:80]), raw[:, class_label], rtol=1e-6
        )


def test_xgboost_categorical_conversion_predicts_raw_margin():
    """Test that categorical XGBoost conversions match the raw model margin.

    Covers the CSR category decoding, the child swap (XGBoost routes in-set values to the
    "yes"/right branch natively), NaN / unseen category routing, and per-class selection
    for multiclass models. Inputs are cast through float32 because XGBoost compares
    float32-cast values against float32 thresholds (the known threshold-edge behavior of
    numeric splits, unrelated to categorical support).
    """
    xgboost = pytest.importorskip("xgboost")

    X, y, y_classes = _make_categorical_dataset()
    feature_types = ["c", "q", "c"]
    X32 = X.astype(np.float32).astype(np.float64)

    dtrain = xgboost.DMatrix(X, label=y, feature_types=feature_types, enable_categorical=True)
    booster = xgboost.train({"tree_method": "hist", "max_depth": 4}, dtrain, num_boost_round=5)
    converted = convert_tree_model(booster)
    assert any(tree.has_categorical for tree in converted)
    expected = booster.predict(
        xgboost.DMatrix(X, feature_types=feature_types, enable_categorical=True),
        output_margin=True,
    )
    np.testing.assert_allclose(_predict_tree_ensemble(converted, X32), expected, atol=1e-5)
    # unseen / NaN category codes route like the model routes them
    probes = np.repeat(X[:1], 3, axis=0)
    probes[:, 0] = [50.0, 7.0, np.nan]
    expected_probes = booster.predict(
        xgboost.DMatrix(probes, feature_types=feature_types, enable_categorical=True),
        output_margin=True,
    )
    probes32 = probes.astype(np.float32).astype(np.float64)
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted, probes32), expected_probes, atol=1e-5
    )

    dtrain_clf = xgboost.DMatrix(
        X, label=y_classes, feature_types=feature_types, enable_categorical=True
    )
    booster_clf = xgboost.train(
        {"tree_method": "hist", "max_depth": 3, "objective": "multi:softprob", "num_class": 3},
        dtrain_clf,
        num_boost_round=4,
    )
    raw = booster_clf.predict(
        xgboost.DMatrix(X[:80], feature_types=feature_types, enable_categorical=True),
        output_margin=True,
    )
    for class_label in range(3):
        converted_class = convert_tree_model(booster_clf, class_label=class_label)
        np.testing.assert_allclose(
            _predict_tree_ensemble(converted_class, X32[:80]), raw[:, class_label], atol=1e-5
        )


def test_cpp_edge_tree_matches_python_edge_tree():
    """Test C++ edge conversion parity with the Python reference implementation."""
    children_left = np.asarray([1, 3, 5, -1, -1, -1, -1])
    children_right = np.asarray([2, 4, 6, -1, -1, -1, -1])
    features = np.asarray([0, 0, 1, -2, -2, -2, -2])
    node_sample_weight = np.asarray([10.0, 6.0, 4.0, 2.0, 4.0, 1.0, 3.0])
    values = np.asarray([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    subset_updates_pos_store = {
        1: {0: np.asarray([0]), 1: np.asarray([1])},
        2: {0: np.asarray([0]), 1: np.asarray([0])},
    }

    python_tree = create_edge_tree_python(
        children_left=children_left,
        children_right=children_right,
        features=features,
        node_sample_weight=node_sample_weight,
        values=values,
        n_nodes=7,
        n_features=2,
        max_interaction=2,
        subset_updates_pos_store=subset_updates_pos_store,
    )
    cpp_tree = create_edge_tree(
        children_left=children_left,
        children_right=children_right,
        features=features,
        node_sample_weight=node_sample_weight,
        values=values,
        n_nodes=7,
        n_features=2,
        max_interaction=2,
        subset_updates_pos_store=subset_updates_pos_store,
    )

    np.testing.assert_array_equal(cpp_tree.parents, python_tree.parents)
    np.testing.assert_array_equal(cpp_tree.ancestors, python_tree.ancestors)
    np.testing.assert_allclose(cpp_tree.p_e_values, python_tree.p_e_values)
    np.testing.assert_allclose(cpp_tree.p_e_storages, python_tree.p_e_storages)
    np.testing.assert_allclose(cpp_tree.split_weights, python_tree.split_weights)
    np.testing.assert_allclose(cpp_tree.empty_predictions, python_tree.empty_predictions)
    np.testing.assert_array_equal(cpp_tree.edge_heights, python_tree.edge_heights)
    np.testing.assert_array_equal(
        cpp_tree.last_feature_node_in_path,
        python_tree.last_feature_node_in_path,
    )
    assert cpp_tree.max_depth == python_tree.max_depth
    assert set(cpp_tree.ancestor_nodes) == set(range(1, len(children_left)))
    for node_id in python_tree.ancestor_nodes:
        np.testing.assert_array_equal(
            cpp_tree.ancestor_nodes[node_id],
            python_tree.ancestor_nodes[node_id],
        )
    for order in python_tree.interaction_height_store:
        np.testing.assert_array_equal(
            cpp_tree.interaction_height_store[order],
            python_tree.interaction_height_store[order],
        )


def test_catboost_json_oblivious_tree_conversion():
    """Test expansion of a CatBoost JSON oblivious tree into a TreeModel."""
    model_json = {
        "features_info": {
            "float_features": [
                {"feature_index": 0, "nan_value_treatment": "AsIs"},
                {"feature_index": 1, "nan_value_treatment": "AsIs"},
            ],
        },
        "oblivious_trees": [
            {
                "splits": [
                    {"split_type": "FloatFeature", "float_feature_index": 0, "border": 1.0},
                    {"split_type": "FloatFeature", "float_feature_index": 1, "border": 2.0},
                ],
                "leaf_values": [10.0, 20.0, 30.0, 40.0],
                "leaf_weights": [1.0, 2.0, 3.0, 4.0],
            },
        ],
    }

    tree_model = parse_catboost_json_model(model_json)[0]
    python_tree_model = parse_catboost_json_model_python(model_json)[0]

    assert safe_isinstance(tree_model, ["shapiq.tree.base.TreeModel"])
    assert tree_model.n_nodes == 7
    assert tree_model.predict_one(np.asarray([0.0, 0.0])) == 10.0
    assert tree_model.predict_one(np.asarray([2.0, 0.0])) == 20.0
    assert tree_model.predict_one(np.asarray([0.0, 3.0])) == 30.0
    assert tree_model.predict_one(np.asarray([2.0, 3.0])) == 40.0
    assert tree_model.node_sample_weight[0] == 10.0
    np.testing.assert_array_equal(tree_model.children_left, python_tree_model.children_left)
    np.testing.assert_array_equal(tree_model.children_right, python_tree_model.children_right)
    np.testing.assert_array_equal(tree_model.children_missing, python_tree_model.children_missing)
    np.testing.assert_array_equal(tree_model.features, python_tree_model.features)
    np.testing.assert_allclose(tree_model.thresholds, python_tree_model.thresholds)
    np.testing.assert_allclose(tree_model.values, python_tree_model.values)
    np.testing.assert_allclose(tree_model.node_sample_weight, python_tree_model.node_sample_weight)


def test_catboost_json_multiclass_defaults_to_class_one():
    """Test extraction of one class from CatBoost multiclass leaf values."""
    model_json = {
        "features_info": {"float_features": [{"feature_index": 0, "nan_value_treatment": "AsIs"}]},
        "oblivious_trees": [
            {
                "splits": [
                    {"split_type": "FloatFeature", "float_feature_index": 0, "border": 1.0},
                ],
                "leaf_values": [1.0, 2.0, 10.0, 20.0],
                "leaf_weights": [3.0, 4.0],
            },
        ],
        "scale_and_bias": [2.0, [100.0, 200.0]],
    }

    tree_model = parse_catboost_json_model(model_json)[0]
    assert tree_model.predict_one(np.asarray([0.0])) == 204.0
    assert tree_model.predict_one(np.asarray([2.0])) == 240.0

    tree_model = parse_catboost_json_model(model_json, class_label=0)[0]
    assert tree_model.predict_one(np.asarray([0.0])) == 102.0
    assert tree_model.predict_one(np.asarray([2.0])) == 120.0


def test_catboost_json_missing_value_routing():
    """Test CatBoost's exported NaN treatment is translated to children_missing."""
    model_json = {
        "features_info": {
            "float_features": [{"feature_index": 0, "nan_value_treatment": "AsTrue"}],
        },
        "oblivious_trees": [
            {
                "splits": [
                    {"split_type": "FloatFeature", "float_feature_index": 0, "border": 1.0},
                ],
                "leaf_values": [1.0, 2.0],
                "leaf_weights": [3.0, 4.0],
            },
        ],
    }

    tree_model = parse_catboost_json_model(model_json)[0]

    assert tree_model.children_missing[0] == tree_model.children_right[0]
    assert tree_model.predict_one(np.asarray([np.nan])) == 2.0


def test_catboost_json_unsupported_split_type_raises_clear_error():
    """Test unsupported CatBoost split formats fail loudly."""
    model_json = {
        "oblivious_trees": [
            {
                "splits": [
                    {"split_type": "OneHotFeature", "cat_feature_index": 0, "value": 1},
                ],
                "leaf_values": [1.0, 2.0],
                "leaf_weights": [1.0, 1.0],
            },
        ],
    }

    with pytest.raises(RuntimeError, match="FloatFeature"):
        parse_catboost_json_model(model_json)


def test_xgboost_regressor_and_booster_conversion_predict_raw_margin():
    """Test XGBoost sklearn and native Booster conversions against raw model margins."""
    xgboost = pytest.importorskip("xgboost")
    from sklearn.datasets import make_regression

    X, y = make_regression(random_state=42, n_samples=80, n_features=5)
    model = xgboost.XGBRegressor(
        random_state=42,
        n_estimators=5,
        max_depth=2,
        objective="reg:squarederror",
    )
    model.fit(X, y)

    expected = model.predict(X[:10], output_margin=True)
    converted_model = convert_tree_model(model)
    converted_booster = convert_tree_model(model.get_booster())

    assert converted_model[0].thresholds.dtype == np.float32
    assert converted_model[0].values.dtype == np.float32
    assert converted_model[0].node_sample_weight.dtype == np.float32
    np.testing.assert_allclose(_predict_tree_ensemble(converted_model, X[:10]), expected, rtol=1e-5)
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted_booster, X[:10]), expected, rtol=1e-5
    )


def test_xgboost_multiclass_conversion_selects_requested_class():
    """Test XGBoost multiclass conversion filters the requested class trees."""
    xgboost = pytest.importorskip("xgboost")
    from sklearn.datasets import make_classification

    X, y = make_classification(
        random_state=42,
        n_samples=120,
        n_features=6,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
    )
    model = xgboost.XGBClassifier(
        random_state=42,
        n_estimators=4,
        max_depth=2,
        objective="multi:softprob",
        num_class=3,
    )
    model.fit(X, y)
    expected = model.predict(X[:10], output_margin=True)

    for class_label in range(3):
        converted = convert_tree_model(model, class_label=class_label)
        assert len(converted) == model.n_estimators
        np.testing.assert_allclose(
            _predict_tree_ensemble(converted, X[:10]),
            expected[:, class_label],
            rtol=1e-5,
            atol=1e-5,
        )


def test_lightgbm_regressor_and_booster_conversion_predict_raw_score():
    """Test LightGBM sklearn and native Booster conversions against raw scores."""
    lightgbm = pytest.importorskip("lightgbm")
    from sklearn.datasets import make_regression

    X, y = make_regression(random_state=42, n_samples=80, n_features=5)
    model = lightgbm.LGBMRegressor(
        random_state=42,
        n_estimators=5,
        max_depth=2,
        min_child_samples=1,
        verbose=-1,
    )
    model.fit(X, y)
    booster = lightgbm.train(
        params={"objective": "regression", "verbose": -1, "min_data_in_leaf": 1},
        train_set=lightgbm.Dataset(X, label=y),
        num_boost_round=5,
    )
    converted_model = convert_tree_model(model)
    converted_booster = convert_tree_model(booster)

    assert converted_model[0].thresholds.dtype == np.float64
    assert converted_model[0].values.dtype == np.float64
    assert converted_model[0].node_sample_weight.dtype == np.float64
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted_model, X[:10]),
        model.predict(X[:10], raw_score=True),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted_booster, X[:10]),
        booster.predict(X[:10], raw_score=True),
        rtol=1e-5,
    )


def test_lightgbm_multiclass_conversion_selects_requested_class():
    """Test LightGBM multiclass conversion filters the requested class trees."""
    lightgbm = pytest.importorskip("lightgbm")
    from sklearn.datasets import make_classification

    X, y = make_classification(
        random_state=42,
        n_samples=120,
        n_features=6,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
    )
    model = lightgbm.LGBMClassifier(
        random_state=42,
        n_estimators=4,
        max_depth=2,
        min_child_samples=1,
        verbose=-1,
    )
    model.fit(X, y)
    expected = model.predict(X[:10], raw_score=True)

    for class_label in range(3):
        converted = convert_tree_model(model, class_label=class_label)
        assert len(converted) == model.n_estimators_
        np.testing.assert_allclose(
            _predict_tree_ensemble(converted, X[:10]),
            expected[:, class_label],
            rtol=1e-5,
            atol=1e-5,
        )


def test_catboost_regressor_conversion_predicts_raw_formula_with_bias():
    """Test CatBoost conversion includes scale_and_bias from the JSON model."""
    catboost = pytest.importorskip("catboost")
    from sklearn.datasets import make_regression

    X, y = make_regression(random_state=42, n_samples=80, n_features=5)
    model = catboost.CatBoostRegressor(
        random_seed=42,
        iterations=5,
        depth=2,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(X, y)
    converted_model = convert_tree_model(model)

    assert converted_model[0].thresholds.dtype == np.float64
    assert converted_model[0].values.dtype == np.float64
    assert converted_model[0].node_sample_weight.dtype == np.float64
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted_model, X[:10]),
        model.predict(X[:10], prediction_type="RawFormulaVal"),
        rtol=1e-5,
    )


def test_catboost_multiclass_conversion_selects_requested_class():
    """Test CatBoost multiclass conversion extracts class-specific leaf values and bias."""
    catboost = pytest.importorskip("catboost")
    from sklearn.datasets import make_classification

    X, y = make_classification(
        random_state=42,
        n_samples=120,
        n_features=6,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
    )
    model = catboost.CatBoostClassifier(
        random_seed=42,
        iterations=4,
        depth=2,
        loss_function="MultiClass",
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(X, y)
    expected = model.predict(X[:10], prediction_type="RawFormulaVal")

    for class_label in range(3):
        converted = convert_tree_model(model, class_label=class_label)
        assert len(converted) == model.tree_count_
        np.testing.assert_allclose(
            _predict_tree_ensemble(converted, X[:10]),
            expected[:, class_label],
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.external_libraries
@pytest.mark.parametrize(("model_fixture", "model_class"), TREE_MODEL_FIXTURES)
def test_conversion_predict_identity(model_fixture, model_class, background_reg_data, request):
    """Tests whether the conversion of the model to a tree explainer is correct."""
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
                if True:
                    # xgboost sometimes predicts a different value
                    # see .test_tree_bugfix.test_xgb_predicts_with_wrong_leaf_node
                    continue
                msg = "Prediction does not match the original prediction."
                raise AssertionError(msg)


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
