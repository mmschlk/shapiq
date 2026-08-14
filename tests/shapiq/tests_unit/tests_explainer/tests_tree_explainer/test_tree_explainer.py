"""This test module contains all tests for the tree explainer module of the shapiq package."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from shapiq.tree import TreeExplainer, TreeModel
from tests.shapiq.markers import skip_if_no_lightgbm


def test_decision_tree_classifier(rf_clf_model, background_clf_data):
    """Test TreeExplainer with a simple decision tree classifier."""
    explainer = TreeExplainer(model=rf_clf_model, max_order=2, min_order=1, index="k-SII")

    x_explain = background_clf_data[0]
    explanation = explainer.explain(x_explain)
    prediction = rf_clf_model.predict_proba(x_explain.reshape(1, -1))[0]

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type

    # check init with class label
    _ = TreeExplainer(model=rf_clf_model, max_order=2, min_order=0, index="k-SII", class_index=0)

    assert True

    explainer = _ = TreeExplainer(model=rf_clf_model, max_order=1, min_order=0, class_index=1)
    explanation = explainer.explain(x_explain)

    # compare baseline_value with the per-tree empty predictions; max_order=1 SV routes
    # through the LinearTreeSHAP path so we read from `_trees` rather than the now-empty
    # `_treeshapiq_explainers` list.
    assert explainer.baseline_value == sum(tree.empty_prediction for tree in explainer._trees)
    assert explanation.baseline_value == explainer.baseline_value

    # test efficiency
    sum_of_values = sum(explanation.values)
    assert prediction[1] == pytest.approx(sum_of_values)


def test_decision_tree_regression(dt_reg_model, background_reg_data):
    """Test TreeExplainer with a simple decision tree regressor."""
    explainer = TreeExplainer(model=dt_reg_model, max_order=2, min_order=1, index="k-SII")

    x_explain = background_reg_data[0]
    explanation = explainer.explain(x_explain)
    prediction = dt_reg_model.predict(x_explain.reshape(1, -1))

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type

    # compare baseline_value with empty_predictions
    assert explainer.baseline_value == sum(
        [treeshapiq.empty_prediction for treeshapiq in explainer._treeshapiq_explainers],
    )
    assert explanation.baseline_value == explainer.baseline_value

    # test efficiency: with min_order=1 the empty interaction is excluded,
    # so the baseline must be added back when checking the k-SII efficiency property
    sum_of_values = sum(explanation.values) + explanation.baseline_value
    assert prediction == pytest.approx(sum_of_values)


def test_random_forest_regression(rf_reg_model, background_reg_data):
    """Test TreeExplainer with a simple decision tree regressor."""
    explainer = TreeExplainer(model=rf_reg_model, max_order=2, min_order=1, index="k-SII")
    explainer._init_explainers()

    x_explain = background_reg_data[0]
    explanation = explainer.explain(x_explain)

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type

    # compare baseline_value with empty_predictions
    assert explainer.baseline_value == sum(
        [treeshapiq.empty_prediction for treeshapiq in explainer._treeshapiq_explainers],
    )
    assert explanation.baseline_value == explainer.baseline_value

    # assert efficiency: with min_order=1 the empty interaction is excluded,
    # so the baseline must be added back when checking the k-SII efficiency property
    prediction = rf_reg_model.predict(x_explain.reshape(1, -1))[0]
    sum_of_values = sum(explanation.values) + explanation.baseline_value
    assert prediction == pytest.approx(sum_of_values)


def test_random_forest_classification(rf_clf_model, background_clf_data):
    """Test TreeExplainer with a simple decision tree regressor."""
    class_label = 0
    explainer = TreeExplainer(
        model=rf_clf_model,
        max_order=1,
        min_order=0,
        index="SV",
        class_index=class_label,
    )

    x_explain = background_clf_data[0]
    explanation = explainer.explain(x_explain)

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type

    # compare baseline_value with the per-tree empty predictions; max_order=1 SV routes
    # through the LinearTreeSHAP path so we read from `_trees` rather than the now-empty
    # `_treeshapiq_explainers` list.
    assert explainer.baseline_value == sum(tree.empty_prediction for tree in explainer._trees)
    assert explanation.baseline_value == explainer.baseline_value

    # assert efficieny
    prediction = rf_clf_model.predict_proba(x_explain.reshape(1, -1))[0, class_label]
    sum_of_values = sum(explanation.values)
    assert prediction == pytest.approx(sum_of_values)


def test_against_shap_implementation():
    """Test the tree explainer against the shap implementation's tree explainer results."""
    # manual values for a tree to test against the shap implementation
    children_left = np.asarray([1, 2, 3, -1, -1, -1, 7, -1, -1])
    children_right = np.asarray([6, 5, 4, -1, -1, -1, 8, -1, -1])
    features = np.asarray([0, 1, 0, -2, -2, -2, 2, -2, -2])
    thresholds = np.asarray([0, 0, -0.5, -2, -2, -2, 0, -2, -2])
    node_sample_weight = np.asarray([100, 50, 38, 15, 23, 12, 50, 20, 30])

    # create a classification tree model
    values = [110, 105, 95, 20, 50, 100, 75, 10, 40]
    values = [values[i] / max(values) for i in range(len(values))]
    values = np.asarray(values)

    x_explain = np.asarray([-1, -0.5, 1, 0])

    tree_model = TreeModel(
        children_left=children_left,
        children_right=children_right,
        children_missing=children_left,  # no missing values, so we can set this to anything
        features=features,
        thresholds=thresholds,
        node_sample_weight=node_sample_weight,
        values=values,
    )

    explainer = TreeExplainer(model=tree_model, max_order=1, min_order=1, index="SV")
    explanation = explainer.explain(x_explain)

    assert explanation[(0,)] == pytest.approx(-0.09263158, abs=1e-4)
    assert explanation[(1,)] == pytest.approx(-0.12100478, abs=1e-4)
    assert explanation[(2,)] == pytest.approx(0.02727273, abs=1e-4)
    assert explanation[(3,)] == pytest.approx(0.0, abs=1e-4)

    explainer = TreeExplainer(model=tree_model, max_order=1, min_order=1, index="SII")
    explanation = explainer.explain(x_explain)

    assert explanation[(0,)] == pytest.approx(-0.09263158, abs=1e-4)
    assert explanation[(1,)] == pytest.approx(-0.12100478, abs=1e-4)
    assert explanation[(2,)] == pytest.approx(0.02727273, abs=1e-4)
    assert explanation[(3,)] == pytest.approx(0.0, abs=1e-4)

    with pytest.warns(UserWarning):
        _ = TreeExplainer(model=tree_model, max_order=2, min_order=1, index="SV")


def test_min_order_filters_interactions(rf_reg_model, background_reg_data):
    """``min_order`` must actually restrict the returned interactions (issue #325)."""
    x_explain = background_reg_data[0]

    # min_order >= 2 drops order-1 interactions but keeps the higher orders
    explainer = TreeExplainer(model=rf_reg_model, max_order=3, min_order=2, index="k-SII")
    explanation = explainer.explain(x_explain)
    assert explanation.min_order == 2
    assert explanation.max_order == 3
    assert len(explanation.interaction_lookup) > 0
    assert all(len(interaction) >= 2 for interaction in explanation.interaction_lookup)

    # min_order == 0 still injects the empty interaction at the baseline value
    explainer_zero = TreeExplainer(model=rf_reg_model, max_order=2, min_order=0, index="k-SII")
    explanation_zero = explainer_zero.explain(x_explain)
    assert () in explanation_zero.interaction_lookup
    assert explanation_zero[()] == pytest.approx(explanation_zero.baseline_value)
    assert any(len(interaction) == 1 for interaction in explanation_zero.interaction_lookup)

    # min_order == 1 matches the unfiltered default-min_order=0 explainer with the
    # empty interaction removed (regression guard for the default code path)
    explainer_one = TreeExplainer(model=rf_reg_model, max_order=2, min_order=1, index="k-SII")
    explanation_one = explainer_one.explain(x_explain)
    assert explanation_one.min_order == 1
    assert () not in explanation_one.interaction_lookup
    for interaction in explanation_one.interaction_lookup:
        assert explanation_one[interaction] == pytest.approx(explanation_zero[interaction])


def test_min_order_validation(rf_reg_model):
    """Invalid ``min_order`` configurations must be rejected eagerly."""
    with pytest.raises(ValueError, match="min_order"):
        TreeExplainer(model=rf_reg_model, max_order=2, min_order=3, index="k-SII")
    with pytest.raises(ValueError, match="min_order"):
        TreeExplainer(model=rf_reg_model, max_order=2, min_order=-1, index="k-SII")


def test_xgboost_reg(xgb_reg_model, background_reg_data):
    """Tests the shapiq implementation of TreeSHAP agains SHAP's implementation for XGBoost."""
    explanation_instance = 0

    # the following code is used to get the shap values from the SHAP implementation
    """
    # import shap
    # explainer_shap = shap.TreeExplainer(model=xgb_reg_model)
    # x_explain_shap = background_reg_data[explanation_instance].reshape(1, -1)
    # sv_shap = explainer_shap.shap_values(x_explain_shap)[0]
    """
    sv_shap = [-2.555832, 28.50987, 1.7708225, -7.8653603, 10.7955885, -0.1877861, 4.549199]
    sv_shap = np.asarray(sv_shap)

    # compute with shapiq
    explainer_shapiq = TreeExplainer(model=xgb_reg_model, max_order=1, index="SV")
    x_explain_shapiq = background_reg_data[explanation_instance]
    sv_shapiq = explainer_shapiq.explain(x=x_explain_shapiq)
    sv_shapiq_values = sv_shapiq.get_n_order_values(1)
    baseline_shapiq = sv_shapiq.baseline_value

    assert np.allclose(sv_shap, sv_shapiq_values, rtol=1e-5)

    # get prediction of the model
    prediction = xgb_reg_model.predict(x_explain_shapiq.reshape(1, -1))
    assert prediction == pytest.approx(baseline_shapiq + np.sum(sv_shapiq_values), rel=1e-5)


def test_xgboost_clf(xgb_clf_model, background_clf_data):
    """Tests the shapiq implementation of TreeSHAP against SHAP's implementation for XGBoost."""
    explanation_instance = 1
    class_label = 1

    # the following code is used to get the shap values from the SHAP implementation
    """
    # import shap
    # model_copy = copy.deepcopy(xgb_clf_model)
    # explainer_shap = shap.TreeExplainer(model=model_copy)
    # baseline_shap = float(explainer_shap.expected_value[class_label])
    # print(baseline_shap)
    # x_explain_shap = copy.deepcopy(background_clf_data[explanation_instance].reshape(1, -1))
    # sv_shap_all_classes = explainer_shap.shap_values(x_explain_shap)
    # sv_shap = sv_shap_all_classes[0][:, class_label]
    # print(sv_shap)
    """
    sv = [-0.00543903, -0.15696308, -0.17532629, -0.24037467, 0.00245022, 0.00986468, -0.01556843]
    sv_shap = np.array(sv)

    # compute with shapiq
    explainer_shapiq = TreeExplainer(
        model=xgb_clf_model,
        max_order=1,
        index="SV",
        class_index=class_label,
    )
    x_explain_shapiq = copy.deepcopy(background_clf_data[explanation_instance])
    sv_shapiq = explainer_shapiq.explain(x=x_explain_shapiq)
    sv_shapiq_values = sv_shapiq.get_n_order_values(1)
    baseline_shapiq = sv_shapiq.baseline_value

    assert np.allclose(sv_shap, sv_shapiq_values, rtol=1e-5)

    # get prediction of the model (as the log odds)
    prediction = xgb_clf_model.predict(x_explain_shapiq.reshape(1, -1), output_margin=True)[0][
        class_label
    ]
    assert prediction == pytest.approx(baseline_shapiq + np.sum(sv_shapiq_values), rel=1e-5)


def test_random_forest_reg(rf_reg_model, background_reg_data):
    """Tests the shapiq implementation of TreeSHAP vs. SHAP's implementation for Random Forest."""
    explanation_instance = 1

    # the following code is used to get the shap values from the SHAP implementation
    """
    # import shap
    # model_copy = copy.deepcopy(rf_reg_model)
    # explainer_shap = shap.TreeExplainer(model=model_copy)
    # baseline_shap = float(explainer_shap.expected_value)
    # x_explain_shap = copy.deepcopy(background_reg_data[explanation_instance].reshape(1, -1))
    # sv_shap_all_classes = explainer_shap.shap_values(x_explain_shap)
    # sv_shap = sv_shap_all_classes[0]
    # print(sv_shap_all_classes, baseline_shap)
    """
    sv_shap = [25.8278293, -77.40235947, 0.0, 21.7067263, -4.85542565, 0.0, 4.91330141]
    sv_shap = np.asarray(sv_shap)
    baseline_shap = -0.713665621534487

    # compute with shapiq
    explainer_shapiq = TreeExplainer(model=rf_reg_model, max_order=1, index="SV")
    x_explain_shapiq = copy.deepcopy(background_reg_data[explanation_instance])
    sv_shapiq = explainer_shapiq.explain(x=x_explain_shapiq)
    sv_shapiq_values = sv_shapiq.get_n_order_values(1)
    baseline_shapiq = sv_shapiq.baseline_value

    assert baseline_shap == pytest.approx(baseline_shapiq, rel=1e-4)
    assert np.allclose(sv_shap, sv_shapiq_values, rtol=1e-5)


def test_random_forest_shap(rf_clf_model, background_clf_data):
    """Tests the shapiq implementation of TreeSHAP vs. SHAP's implementation for Random Forest."""
    explanation_instance = 1
    class_label = 1

    # the following code is used to get the shap values from the SHAP implementation
    """
    # import shap
    # model_copy = copy.deepcopy(rf_clf_model)
    # explainer_shap = shap.TreeExplainer(model=model_copy)
    # baseline_shap = float(explainer_shap.expected_value[class_label])
    # x_explain_shap = copy.deepcopy(background_clf_data[explanation_instance].reshape(1, -1))
    # sv_shap_all_classes = explainer_shap.shap_values(x_explain_shap)
    # sv_shap = sv_shap_all_classes[0][:, class_label]
    # print(sv_shap_all_classes, baseline_shap)
    """
    sv_shap = [-0.00537992, 0.0, -0.08206514, -0.03122057, 0.0025626, 0.03182904, 0.03782473]
    sv_shap = np.asarray(sv_shap)
    baseline_shap = 0.32000000000000006

    # compute with shapiq
    explainer_shapiq = TreeExplainer(
        model=rf_clf_model,
        max_order=1,
        index="SV",
        class_index=class_label,
    )
    x_explain_shapiq = copy.deepcopy(background_clf_data[explanation_instance])
    sv_shapiq = explainer_shapiq.explain(x=x_explain_shapiq)
    sv_shapiq_values = sv_shapiq.get_n_order_values(1)
    baseline_shapiq = sv_shapiq.baseline_value

    assert baseline_shap == pytest.approx(baseline_shapiq, rel=1e-4)
    assert np.allclose(sv_shap, sv_shapiq_values, rtol=1e-5)


def test_lightgbm_clf_shap(lightgbm_clf_model, background_clf_data):
    """Tests the shapiq implementation of TreeSHAP vs. SHAP's implementation for LightGBM."""
    explanation_instance = 1
    class_label = 1

    # the following code is used to get the shap values from the SHAP implementation
    # note that you need to uncomment these lines in the shap library you have locally installed:
    # https://github.com/shap/shap/blob/6c4a71ce59ea579be58917d824fa0ba5cd97e787/shap/explainers/_tree.py#L543C1-L547C26

    """
    # import shap
    # model_copy = copy.deepcopy(lightgbm_clf_model)
    # explainer_shap = shap.TreeExplainer(model=model_copy)
    # baseline_shap = float(explainer_shap.expected_value[class_label])
    # x_explain_shap = copy.deepcopy(background_clf_data[explanation_instance].reshape(1, -1))
    # sv_shap_all_classes = explainer_shap.shap_values(x_explain_shap)
    # sv_shap = sv_shap_all_classes[0][:, class_label]
    # print(sv_shap_all_classes, baseline_shap)
    """
    sv_shap = [0.0, 0.0, -0.05747963, -0.20128496, 0.0, 0.0, 0.01560273]
    sv_shap = np.asarray(sv_shap)
    baseline_shap = -1.0862557008895362

    # compute with shapiq
    explainer_shapiq = TreeExplainer(
        model=lightgbm_clf_model,
        max_order=1,
        index="SV",
        class_index=class_label,
    )
    x_explain_shapiq = copy.deepcopy(background_clf_data[explanation_instance])
    sv_shapiq = explainer_shapiq.explain(x=x_explain_shapiq)
    sv_shapiq_values = sv_shapiq.get_n_order_values(1)
    baseline_shapiq = sv_shapiq.baseline_value

    assert baseline_shap == pytest.approx(baseline_shapiq, rel=1e-4)
    assert np.allclose(sv_shap, sv_shapiq_values, rtol=1e-5)


def test_xgboost_shap_error(xgb_clf_model, background_clf_data):
    """Tests for the strange behavior of SHAP's XGBoost implementation.

    The test is used to show that the shapiq implementation is correct and the SHAP implementation
    is doing something weird. For some instances (e.g. the one used in this test) the SHAP values
    are different from the shapiq values. However, when we round the `thresholds` of the xgboost
    trees in shapiq, then the computed explanations match. This is a strange behavior as rounding
    the thresholds makes the model less true to the original model but only then the explanations
    match.
    """
    explanation_instance = 0
    class_label = 1

    # get the shap explanations (the following code is used to get SVs from SHAP)
    # import shap  # noqa: ERA001
    # model_copy = copy.deepcopy(xgb_clf_model) # noqa: ERA001
    # explainer_shap = shap.TreeExplainer(model=model_copy)  # noqa: ERA001
    # baseline_shap = float(explainer_shap.expected_value[class_label])  # noqa: ERA001
    # x_explain_shap = copy.deepcopy(background_clf_data[explanation_instance].reshape(1, -1))  # noqa: ERA001
    # sv_shap_all_classes = explainer_shap.shap_values(x_explain_shap)  # noqa: ERA001
    # sv_shap = sv_shap_all_classes[0][:, class_label]  # noqa: ERA001
    # print(sv_shap)  # noqa: ERA001
    # print(baseline_shap)  # noqa: ERA001
    sv = [-0.00163171, 0.05075389, -0.13064955, -0.4421068, 0.00424677, -0.04832656, -0.01364264]
    sv_shap = np.array(sv)

    # setup shapiq TreeSHAP
    explainer_shapiq = TreeExplainer(
        model=xgb_clf_model,
        max_order=1,
        index="SV",
        class_index=class_label,
    )
    x_explain_shapiq = copy.deepcopy(background_clf_data[explanation_instance])
    sv_shapiq = explainer_shapiq.explain(x=x_explain_shapiq)
    sv_shapiq_values = sv_shapiq.get_n_order_values(1)

    # the SHAP sv values should be different from the shapiq values
    assert not np.allclose(sv_shap, sv_shapiq_values, rtol=1e-5)

    # when we round the model thresholds of the xgb model (thresholds decide weather a feature is
    # used or not) -> then suddenly the shap and shapiq values are the same, which points to the
    # fact that the shapiq implementation is correct
    explainer_shapiq_rounded = TreeExplainer(
        model=xgb_clf_model,
        max_order=1,
        index="SV",
        class_index=class_label,
    )
    explainer_shapiq_rounded._init_explainers()
    # max_order=1 SV routes through the LinearTreeSHAP path; round thresholds on whichever
    # per-tree explainer list was populated so the mutation actually takes effect at explain time.
    per_tree_explainers = (
        explainer_shapiq_rounded._lineartreeshap_explainers
        or explainer_shapiq_rounded._treeshapiq_explainers
    )
    for tree_explainer in per_tree_explainers:
        tree_explainer._tree.thresholds = np.round(tree_explainer._tree.thresholds, 4)
    x_explain_shapiq_rounded = copy.deepcopy(background_clf_data[explanation_instance])
    sv_shapiq_rounded = explainer_shapiq_rounded.explain(x=x_explain_shapiq_rounded)
    sv_shapiq_rounded_values = sv_shapiq_rounded.get_n_order_values(1)

    # now the values surprisingly are the same
    assert np.allclose(sv_shap, sv_shapiq_rounded_values, rtol=1e-5)


def test_iso_forest_shap(if_clf_model):
    """Tests the shapiq implementation of TreeSHAP vs. SHAP's implementation for Isolation Forest."""
    x_explain = np.array([0.125, 0.05])

    # the following code is used to get the shap values from the SHAP implementation
    # import shap  # noqa: ERA001
    # model_copy = copy.deepcopy(if_clf_model)  # noqa: ERA001
    # explainer_shap = shap.TreeExplainer(model=model_copy)  # noqa: ERA001
    # baseline_shap = float(explainer_shap.expected_value)  # noqa: ERA001
    # sv_shap = explainer_shap.shap_values(x_explain)  # noqa: ERA001
    # print(sv_shap)  # noqa: ERA001
    # print(baseline_shap)  # noqa: ERA001
    sv_shap = np.array([-1.40624839, -3.21377854])
    baseline_shap = 11.953360265689595

    # compute with shapiq
    explainer_shapiq = TreeExplainer(model=if_clf_model, max_order=1, index="SV")
    sv_shapiq = explainer_shapiq.explain(x=x_explain)
    sv_shapiq_values = sv_shapiq.get_n_order_values(1)
    baseline_shapiq = sv_shapiq.baseline_value

    assert np.allclose(sv_shap, sv_shapiq_values, rtol=1e-5)
    assert baseline_shap == pytest.approx(baseline_shapiq, rel=1e-6)


@skip_if_no_lightgbm
def test_decision_stumps(background_reg_dataset, background_clf_dataset):
    """Tests weather you can explain a decision stumps with the shapiq implementation."""
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from xgboost import XGBClassifier, XGBRegressor

    models_reg = [
        LGBMRegressor(random_state=42, n_estimators=20, max_depth=1),
        XGBRegressor(random_state=42, n_estimators=20, max_depth=1),
        DecisionTreeRegressor(random_state=42, max_depth=1),
        RandomForestRegressor(random_state=42, n_estimators=20, max_depth=1),
    ]

    models_clf = [
        LGBMClassifier(random_state=42, n_estimators=20, max_depth=1),
        XGBClassifier(random_state=42, n_estimators=20, max_depth=1),
        DecisionTreeClassifier(random_state=42, max_depth=1),
        RandomForestClassifier(random_state=42, n_estimators=20, max_depth=1),
    ]

    for model in models_reg:
        X, y = background_reg_dataset
        model.fit(X, y)

        explainer = TreeExplainer(model=model, max_order=3, index="k-SII")
        x_explain = X[0]
        explanation = explainer.explain(x_explain)

        efficiency = sum(explanation.values)

        pred = model.predict(x_explain.reshape(1, -1))
        assert pred == pytest.approx(efficiency, rel=1e-5)

    for model in models_clf:
        X, y = background_clf_dataset
        model.fit(X, y)

        explainer = TreeExplainer(model=model, max_order=3, index="k-SII", class_index=0)
        x_explain = X[1]
        explanation = explainer.explain(x_explain)

        efficiency = sum(explanation.values)

        if isinstance(model, RandomForestClassifier | DecisionTreeClassifier):
            pred = model.predict_proba(x_explain.reshape(1, -1))[0, 0]
        elif isinstance(model, XGBClassifier):
            pred = model.predict(x_explain.reshape(1, -1), output_margin=True)[0, 0]
        else:  # skip lightgbm
            continue

        assert pred == pytest.approx(efficiency, rel=1e-5)


def test_extra_trees_clf(et_clf_model, background_clf_data):
    """Test the shapiq implementation of TreeSHAP vs. SHAP's implementation for Extra Trees."""
    explanation_instance = 1
    class_label = 1

    # the following code is used to get the shap values from the SHAP implementation
    """
    #import shap
    # model_copy = copy.deepcopy(et_clf_model)
    # explainer_shap = shap.TreeExplainer(model=model_copy)
    # baseline_shap = float(explainer_shap.expected_value[class_label])
    # x_explain_shap = copy.deepcopy(background_clf_data[explanation_instance].reshape(1, -1))
    # sv_shap_all_classes = explainer_shap.shap_values(x_explain_shap)
    # sv_shap = sv_shap_all_classes[0][:, class_label]
    # print(sv_shap_all_classes, format(baseline_shap, '.20f'))
    """
    sv_shap = [0.00207427, 0.00949552, -0.00108266, -0.03825587, -0.02694092, 0.0170296, 0.02046364]
    sv_shap = np.asarray(sv_shap)
    baseline_shap = 0.34000000000000002

    # compute with shapiq
    explainer_shapiq = TreeExplainer(
        model=et_clf_model, max_order=1, index="SV", class_index=class_label
    )
    x_explain_shapiq = copy.deepcopy(background_clf_data[explanation_instance])
    sv_shapiq = explainer_shapiq.explain(x=x_explain_shapiq)
    sv_shapiq_values = sv_shapiq.get_n_order_values(1)
    baseline_shapiq = sv_shapiq.baseline_value

    assert baseline_shap == pytest.approx(baseline_shapiq, rel=1e-4)
    assert np.allclose(sv_shap, sv_shapiq_values, rtol=1e-5)


def test_interventional_missing_reference_dataset_raises(rf_reg_model):
    """``mode='interventional'`` requires a ``reference_dataset`` already at construction."""
    with pytest.raises(ValueError, match="reference_dataset"):
        TreeExplainer(model=rf_reg_model, mode="interventional", max_order=2, index="SII")


def test_interventional_dt_regression(dt_reg_model, background_reg_data):
    """The interventional route runs end-to-end on a single decision-tree regressor."""
    reference = background_reg_data[:20]
    x_explain = background_reg_data[0]

    explainer = TreeExplainer(
        model=dt_reg_model,
        mode="interventional",
        reference_dataset=reference,
        max_order=2,
        min_order=1,
        index="SII",
    )

    # the interventional path must be the one that's wired up
    explainer._init_explainers()
    assert explainer._interventional_explainer is not None
    assert explainer._treeshapiq_explainers == []
    assert explainer._lineartreeshap_explainers == []

    explanation = explainer.explain(x_explain)
    assert type(explanation).__name__ == "InteractionValues"
    assert explanation.max_order == 2
    assert explanation.min_order == 1

    # interventional baseline: mean tree-prediction over the reference data.
    expected_baseline = float(dt_reg_model.predict(reference).mean())
    assert explanation.baseline_value == pytest.approx(expected_baseline, rel=1e-5)

    # interventional SV efficiency holds against the stored baseline directly.
    explainer_sv = TreeExplainer(
        model=dt_reg_model,
        mode="interventional",
        reference_dataset=reference,
        max_order=1,
        min_order=1,
        index="SV",
    )
    sv_explanation = explainer_sv.explain(x_explain)
    order_one_values = sv_explanation.get_n_order_values(1)
    prediction = float(dt_reg_model.predict(x_explain.reshape(1, -1))[0])
    assert sum(order_one_values) + sv_explanation.baseline_value == pytest.approx(
        prediction, rel=1e-4
    )


def test_interventional_rf_regression(rf_reg_model, background_reg_data):
    """The interventional route handles tree ensembles (RandomForestRegressor)."""
    reference = background_reg_data[:20]
    x_explain = background_reg_data[0]

    explainer = TreeExplainer(
        model=rf_reg_model,
        mode="interventional",
        reference_dataset=reference,
        max_order=1,
        min_order=1,
        index="SV",
    )

    explanation = explainer.explain(x_explain)
    assert type(explanation).__name__ == "InteractionValues"

    expected_baseline = float(rf_reg_model.predict(reference).mean())
    assert explanation.baseline_value == pytest.approx(expected_baseline, rel=1e-5)

    order_one_values = explanation.get_n_order_values(1)
    prediction = float(rf_reg_model.predict(x_explain.reshape(1, -1))[0])
    assert sum(order_one_values) + explanation.baseline_value == pytest.approx(prediction, rel=1e-4)


def test_interventional_rf_classification(rf_clf_model, background_clf_data):
    """The interventional route handles classifiers when ``class_index`` is supplied."""
    reference = background_clf_data[:20]
    x_explain = background_clf_data[0]
    class_label = 1

    explainer = TreeExplainer(
        model=rf_clf_model,
        mode="interventional",
        reference_dataset=reference,
        max_order=2,
        min_order=1,
        index="SII",
        class_index=class_label,
    )

    explanation = explainer.explain(x_explain)
    assert type(explanation).__name__ == "InteractionValues"
    assert explanation.max_order == 2
    assert explanation.min_order == 1

    # baseline is the mean predicted probability for the selected class on the reference data.
    expected_baseline = float(rf_clf_model.predict_proba(reference)[:, class_label].mean())
    assert explanation.baseline_value == pytest.approx(expected_baseline, rel=1e-5)


def test_interventional_k_sii_higher_order(dt_reg_model, background_reg_data):
    """k-SII is accepted (resolved to SII internally) and yields all orders up to max_order."""
    reference = background_reg_data[:20]
    x_explain = background_reg_data[0]

    explainer = TreeExplainer(
        model=dt_reg_model,
        mode="interventional",
        reference_dataset=reference,
        max_order=3,
        min_order=1,
        index="k-SII",
    )

    explanation = explainer.explain(x_explain)
    assert explanation.max_order == 3
    orders = {len(interaction) for interaction in explanation.interaction_lookup}
    # the underlying interventional kernel always populates the empty interaction
    assert orders <= {0, 1, 2, 3}
    assert {1, 2, 3} <= orders


def test_interventional_matches_direct_explainer(dt_reg_model, background_reg_data):
    """TreeExplainer's interventional route must produce the same per-feature values as the raw
    :class:`InterventionalTreeExplainer`.

    The actual Shapley contributions come from the same kernel and must agree, and both
    explainers must report the same reference-data-mean baseline.
    """
    from shapiq.tree import InterventionalTreeExplainer

    reference = background_reg_data[:20]
    x_explain = background_reg_data[0]

    wrapper = TreeExplainer(
        model=dt_reg_model,
        mode="interventional",
        reference_dataset=reference,
        max_order=1,
        min_order=1,
        index="SV",
    )
    wrapper_values = wrapper.explain(x_explain).get_n_order_values(1)

    direct = InterventionalTreeExplainer(
        model=dt_reg_model, data=reference, max_order=1, index="SV"
    )
    direct_iv = direct.explain_function(x_explain)
    n_features = background_reg_data.shape[1]
    direct_values = np.array([direct_iv[(i,)] for i in range(n_features)])

    assert np.allclose(wrapper_values, direct_values, rtol=1e-5, atol=1e-6)
    assert wrapper.baseline_value == pytest.approx(direct.baseline_value)


def test_baseline_value_per_mode(rf_reg_model, background_reg_data):
    """``baseline_value`` is the empty prediction of the mode the explainer runs in.

    Path-dependent: the sum of the per-tree (coverage-weighted) empty predictions.
    Interventional: the mean ensemble prediction over the reference dataset — not the
    path-dependent value, which is what a woodelf-routed explanation used to report.
    """
    from shapiq.tree import InterventionalTreeExplainer

    reference = background_reg_data  # 100 rows: 100 * 1 >= 100 routes even explain() to Woodelf
    x_explain = background_reg_data[0]

    pathdependent = TreeExplainer(model=rf_reg_model, max_order=1, index="SV")
    assert pathdependent.baseline_value == pytest.approx(
        sum(tree.empty_prediction for tree in pathdependent._trees)
    )

    interventional = TreeExplainer(
        model=rf_reg_model,
        mode="interventional",
        reference_dataset=reference,
        max_order=1,
        min_order=0,
        index="SV",
    )
    expected_baseline = InterventionalTreeExplainer(
        model=rf_reg_model, data=reference, max_order=1, index="SV"
    ).baseline_value
    assert interventional.baseline_value == pytest.approx(expected_baseline)
    assert expected_baseline == pytest.approx(float(rf_reg_model.predict(reference).mean()))

    # the woodelf-routed single-instance explanation must carry the interventional baseline
    pytest.importorskip("woodelf")
    assert interventional._should_use_woodelf(1)  # guard: Woodelf, not the fallback
    explanation = interventional.explain(x_explain)
    assert explanation.baseline_value == pytest.approx(expected_baseline)
    assert explanation[()] == pytest.approx(expected_baseline)


def test_woodelf_interventional_matches_direct_explainer(rf_reg_model, background_reg_data):
    """Background SHAP over 20 rows x 10 background rows routes through Woodelf.

    ``10 * 20 >= 100`` crosses the interventional Woodelf cut-off, so ``explain_X`` no longer
    runs the shapiq kernel. The per-feature values must still match the direct
    :class:`InterventionalTreeExplainer`.
    """
    from shapiq.tree import InterventionalTreeExplainer

    pytest.importorskip("woodelf")
    reference = background_reg_data[:10]
    x_explain = background_reg_data[10:30]
    n_features = background_reg_data.shape[1]

    explainer = TreeExplainer(
        model=rf_reg_model,
        mode="interventional",
        reference_dataset=reference,
        max_order=1,
        min_order=1,
        index="SV",
    )
    assert explainer._should_use_woodelf(len(x_explain))  # guard: Woodelf, not the fallback
    woodelf_values = explainer.explain_X(x_explain)

    direct = InterventionalTreeExplainer(
        model=rf_reg_model, data=reference, max_order=1, index="SV"
    )
    # Woodelf omits a feature entirely when it is zero for every row, so default to zeros.
    zeros = np.zeros(len(x_explain))
    for row, x in enumerate(x_explain):
        direct_iv = direct.explain_function(x)
        for feature in range(n_features):
            woodelf_value = woodelf_values.get((feature,), zeros)[row]
            assert woodelf_value == pytest.approx(direct_iv[(feature,)], abs=1e-5)


def test_woodelf_pathdependent_matches_treeshapiq(dt_reg_model, background_reg_data):
    """Path-dependent SII with ``min_order=0``, ``max_order=3`` routes through Woodelf.

    ``max_order > 1`` crosses the path-dependent Woodelf cut-off. Orders 1-3 must match
    :class:`TreeSHAPIQ` run directly per tree, and ``min_order=0`` must still put the
    baseline at the empty interaction.
    """
    from shapiq.tree import TreeSHAPIQ
    from shapiq.tree.validation import validate_tree_model

    pytest.importorskip("woodelf")
    x_explain = background_reg_data[0]

    explainer = TreeExplainer(model=dt_reg_model, max_order=3, min_order=0, index="SII")
    assert explainer._should_use_woodelf(1)  # guard: Woodelf, not the fallback
    explanation = explainer.explain(x_explain)

    per_tree = [
        TreeSHAPIQ(model=tree, max_order=3, index="SII").explain(x_explain)
        for tree in validate_tree_model(dt_reg_model)
    ]
    reference = sum(per_tree[1:], start=per_tree[0])

    assert explanation.min_order == 0
    assert explanation.max_order == 3
    assert explanation[()] == pytest.approx(explainer.baseline_value)

    # a subset missing on one side is an exact zero there, so compare over the union
    interactions = set(explanation.interactions) | set(reference.interactions)
    assert {len(interaction) for interaction in interactions} == {0, 1, 2, 3}
    for interaction in interactions:
        if interaction:  # skip the empty interaction, checked against the baseline above
            assert explanation[interaction] == pytest.approx(reference[interaction], abs=1e-5)


@pytest.mark.parametrize(
    ("model_fixture", "data_fixture", "n_classes"),
    [
        ("rf_clf_model", "background_clf_dataset", 3),
        ("rf_clf_binary_model", "background_clf_dataset_binary", 2),
        ("xgb_clf_model", "background_clf_dataset", 3),
    ],
    ids=["rf-multiclass", "rf-binary", "xgb-multiclass"],
)
def test_woodelf_pathdependent_honors_class_index(
    model_fixture,
    data_fixture,
    n_classes,
    request,
):
    """Woodelf must explain the class ``class_index`` asks for.

    ``max_order=2`` crosses the path-dependent Woodelf cut-off, so every class here is
    computed by Woodelf rather than the shapiq kernel. Each one must match
    :class:`~shapiq.tree.TreeSHAPIQ` run directly on the class-selected trees. Sklearn
    ensembles select a class by slicing leaf values and boosters by filtering trees, so
    both mechanisms are covered.
    """
    from shapiq.tree import TreeSHAPIQ
    from shapiq.tree.validation import validate_tree_model

    pytest.importorskip("woodelf")
    model = request.getfixturevalue(model_fixture)
    X, _ = request.getfixturevalue(data_fixture)
    x_explain = np.asarray(X)[2]

    for class_index in range(n_classes):
        explainer = TreeExplainer(
            model=model,
            mode="pathdependent",
            max_order=2,
            min_order=1,
            index="SII",
            class_index=class_index,
        )
        assert explainer._should_use_woodelf(1)  # guard: Woodelf, not the fallback
        explanation = explainer.explain(x_explain)

        per_tree = [
            TreeSHAPIQ(model=tree, max_order=2, index="SII").explain(x_explain)
            for tree in validate_tree_model(model, class_label=class_index)
        ]
        reference = sum(per_tree[1:], start=per_tree[0])

        interactions = set(explanation.interactions) | set(reference.interactions)
        for interaction in interactions:
            if (
                len(interaction) >= 1
            ):  # the empty interaction carries the baseline, not a class value
                assert explanation[interaction] == pytest.approx(reference[interaction], abs=1e-5)


def test_extra_trees_reg(et_reg_model, background_reg_data):
    """Test the shapiq implementation of TreeSHAP vs. SHAP's implementation for Extra Trees."""
    explanation_instance = 1

    # the following code is used to get the shap values from the SHAP implementation
    """
    # import shap
    # model_copy = copy.deepcopy(et_reg_model)
    # explainer_shap = shap.TreeExplainer(model=model_copy)
    # baseline_shap = float(explainer_shap.expected_value)
    # x_explain_shap = copy.deepcopy(background_reg_data[explanation_instance].reshape(1, -1))
    # sv_shap_all_classes = explainer_shap.shap_values(x_explain_shap)
    # sv_shap = sv_shap_all_classes[0]
    # print(sv_shap_all_classes, format(baseline_shap, '.20f'))
    """
    sv_shap = [19.28673017, -19.87182634, 0.0, 10.89201698, -9.62498263, 0.35992212, 42.31290091]
    sv_shap = np.asarray(sv_shap)
    baseline_shap = -2.56682283435175007

    # compute with shapiq
    explainer_shapiq = TreeExplainer(model=et_reg_model, max_order=1, index="SV")
    x_explain_shapiq = copy.deepcopy(background_reg_data[explanation_instance])
    sv_shapiq = explainer_shapiq.explain(x=x_explain_shapiq)
    sv_shapiq_values = sv_shapiq.get_n_order_values(1)
    baseline_shapiq = sv_shapiq.baseline_value

    assert baseline_shap == pytest.approx(baseline_shapiq, rel=1e-4)
    assert np.allclose(sv_shap, sv_shapiq_values, rtol=1e-5)
