"""This test module contains all tests for the tree explainer module of the shapiq package."""

import copy

import numpy as np
import pytest

from shapiq.explainer.tree import TreeExplainer, TreeModel


def test_decision_tree_classifier(dt_clf_model, background_clf_data):
    """Test TreeExplainer with a simple decision tree classifier."""
    explainer = TreeExplainer(model=dt_clf_model, max_order=2, min_order=1)

    x_explain = background_clf_data[0]
    explanation = explainer.explain(x_explain)
    prediction = dt_clf_model.predict_proba(x_explain.reshape(1, -1))[0]

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type

    # check init with class label
    _ = TreeExplainer(model=dt_clf_model, max_order=2, min_order=1, class_index=0)

    assert True

    explainer = _ = TreeExplainer(model=dt_clf_model, max_order=1, min_order=1, class_index=1)
    explanation = explainer.explain(x_explain)

    # compare baseline_value with empty_predictions
    assert explainer.baseline_value == sum(
        [treeshapiq.empty_prediction for treeshapiq in explainer._treeshapiq_explainers]
    )
    assert explanation.baseline_value == explainer.baseline_value

    # test efficiency
    sum_of_values = sum(explanation.values)
    assert prediction[1] == pytest.approx(sum_of_values)


def test_decision_tree_regression(dt_reg_model, background_reg_data):
    """Test TreeExplainer with a simple decision tree regressor."""
    explainer = TreeExplainer(model=dt_reg_model, max_order=2, min_order=1)

    x_explain = background_reg_data[0]
    explanation = explainer.explain(x_explain)
    prediction = dt_reg_model.predict(x_explain.reshape(1, -1))

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type

    # compare baseline_value with empty_predictions
    assert explainer.baseline_value == sum(
        [treeshapiq.empty_prediction for treeshapiq in explainer._treeshapiq_explainers]
    )
    assert explanation.baseline_value == explainer.baseline_value

    # test efficiency
    sum_of_values = sum(explanation.values)
    assert prediction == pytest.approx(sum_of_values)


def test_random_forest_regression(rf_reg_model, background_reg_data):
    """Test TreeExplainer with a simple decision tree regressor."""
    explainer = TreeExplainer(model=rf_reg_model, max_order=2, min_order=1)

    x_explain = background_reg_data[0]
    explanation = explainer.explain(x_explain)

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type

    # compare baseline_value with empty_predictions
    assert explainer.baseline_value == sum(
        [treeshapiq.empty_prediction for treeshapiq in explainer._treeshapiq_explainers]
    )
    assert explanation.baseline_value == explainer.baseline_value

    # assert efficieny
    prediction = rf_reg_model.predict(x_explain.reshape(1, -1))[0]
    sum_of_values = sum(explanation.values)
    assert prediction == pytest.approx(sum_of_values)


def test_random_forest_classification(rf_clf_model, background_clf_data):
    """Test TreeExplainer with a simple decision tree regressor."""
    class_label = 0
    explainer = TreeExplainer(
        model=rf_clf_model, max_order=1, min_order=1, index="SV", class_index=class_label
    )

    x_explain = background_clf_data[0]
    explanation = explainer.explain(x_explain)

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type

    # compare baseline_value with empty_predictions
    assert explainer.baseline_value == sum(
        [treeshapiq.empty_prediction for treeshapiq in explainer._treeshapiq_explainers]
    )
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


def test_xgboost_reg(xgb_reg_model, background_reg_data):
    """Tests the shapiq implementation of TreeSHAP agains SHAP's implementation for XGBoost."""

    explanation_instance = 0

    # the following code is used to get the shap values from the SHAP implementation
    # import shap
    # explainer_shap = shap.TreeExplainer(model=xgb_reg_model)
    # x_explain_shap = background_reg_data[explanation_instance].reshape(1, -1)
    # sv_shap = explainer_shap.shap_values(x_explain_shap)[0]
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
    """Tests the shapiq implementation of TreeSHAP agains SHAP's implementation for XGBoost."""

    explanation_instance = 1
    class_label = 1

    # the following code is used to get the shap values from the SHAP implementation
    # import shap
    # model_copy = copy.deepcopy(xgb_clf_model)
    # explainer_shap = shap.TreeExplainer(model=model_copy)
    # baseline_shap = float(explainer_shap.expected_value[class_label])
    # print(baseline_shap)
    # x_explain_shap = copy.deepcopy(background_clf_data[explanation_instance].reshape(1, -1))
    # sv_shap_all_classes = explainer_shap.shap_values(x_explain_shap)
    # sv_shap = sv_shap_all_classes[0][:, class_label]
    # print(sv_shap)
    sv = [-0.00545454, -0.15837783, -0.17675081, -0.24213657, 0.00247543, 0.00988865, -0.01564346]
    sv_shap = np.array(sv)

    # compute with shapiq
    explainer_shapiq = TreeExplainer(
        model=xgb_clf_model, max_order=1, index="SV", class_index=class_label
    )
    x_explain_shapiq = copy.deepcopy(background_clf_data[explanation_instance])
    sv_shapiq = explainer_shapiq.explain(x=x_explain_shapiq)
    sv_shapiq_values = sv_shapiq.get_n_order_values(1)
    baseline_shapiq = sv_shapiq.baseline_value

    # assert baseline_shap == pytest.approx(baseline_shapiq, rel=1e-4)
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
    # import shap
    # model_copy = copy.deepcopy(rf_reg_model)
    # explainer_shap = shap.TreeExplainer(model=model_copy)
    # baseline_shap = float(explainer_shap.expected_value)
    # x_explain_shap = copy.deepcopy(background_reg_data[explanation_instance].reshape(1, -1))
    # sv_shap_all_classes = explainer_shap.shap_values(x_explain_shap)
    # sv_shap = sv_shap_all_classes[0]
    # print(sv_shap_all_classes, baseline_shap)
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
    # import shap
    # model_copy = copy.deepcopy(rf_clf_model)
    # explainer_shap = shap.TreeExplainer(model=model_copy)
    # baseline_shap = float(explainer_shap.expected_value[class_label])
    # x_explain_shap = copy.deepcopy(background_clf_data[explanation_instance].reshape(1, -1))
    # sv_shap_all_classes = explainer_shap.shap_values(x_explain_shap)
    # sv_shap = sv_shap_all_classes[0][:, class_label]
    # print(sv_shap_all_classes, baseline_shap)
    sv_shap = [-0.00537992, 0.0, -0.08206514, -0.03122057, 0.0025626, 0.03182904, 0.03782473]
    sv_shap = np.asarray(sv_shap)
    baseline_shap = 0.32000000000000006

    # compute with shapiq
    explainer_shapiq = TreeExplainer(
        model=rf_clf_model, max_order=1, index="SV", class_index=class_label
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

    # import shap
    # model_copy = copy.deepcopy(lightgbm_clf_model)
    # explainer_shap = shap.TreeExplainer(model=model_copy)
    # baseline_shap = float(explainer_shap.expected_value[class_label])
    # x_explain_shap = copy.deepcopy(background_clf_data[explanation_instance].reshape(1, -1))
    # sv_shap_all_classes = explainer_shap.shap_values(x_explain_shap)
    # sv_shap = sv_shap_all_classes[0][:, class_label]
    # print(sv_shap_all_classes, baseline_shap)
    sv_shap = [0.0, 0.0, -0.05747963, -0.20128496, 0.0, 0.0, 0.01560273]
    sv_shap = np.asarray(sv_shap)
    baseline_shap = -1.0862557008895362

    # compute with shapiq
    explainer_shapiq = TreeExplainer(
        model=lightgbm_clf_model, max_order=1, index="SV", class_index=class_label
    )
    x_explain_shapiq = copy.deepcopy(background_clf_data[explanation_instance])
    sv_shapiq = explainer_shapiq.explain(x=x_explain_shapiq)
    sv_shapiq_values = sv_shapiq.get_n_order_values(1)
    baseline_shapiq = sv_shapiq.baseline_value

    print(sv_shapiq_values, baseline_shapiq)

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
    # import shap
    # model_copy = copy.deepcopy(xgb_clf_model)
    # explainer_shap = shap.TreeExplainer(model=model_copy)
    # baseline_shap = float(explainer_shap.expected_value[class_label])
    # x_explain_shap = copy.deepcopy(background_clf_data[explanation_instance].reshape(1, -1))
    # sv_shap_all_classes = explainer_shap.shap_values(x_explain_shap)
    # sv_shap = sv_shap_all_classes[0][:, class_label]
    # print(sv_shap)
    # print(baseline_shap)
    sv = [-0.00163636, 0.05099502, -0.13182959, -0.44538185, 0.00428653, -0.04872373, -0.01370917]
    sv_shap = np.array(sv)

    # setup shapiq TreeSHAP
    explainer_shapiq = TreeExplainer(
        model=xgb_clf_model, max_order=1, index="SV", class_index=class_label
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
        model=xgb_clf_model, max_order=1, index="SV", class_index=class_label
    )
    for tree_explainer in explainer_shapiq_rounded._treeshapiq_explainers:
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
    # import shap
    # model_copy = copy.deepcopy(if_clf_model)
    # explainer_shap = shap.TreeExplainer(model=model_copy)
    # baseline_shap = float(explainer_shap.expected_value)
    # sv_shap = explainer_shap.shap_values(x_explain)
    # print(sv_shap)
    # print(baseline_shap)
    sv_shap = np.array([-2.34951688, -4.55545493])
    baseline_shap = 12.238305148044713

    # compute with shapiq
    explainer_shapiq = TreeExplainer(model=if_clf_model, max_order=1, index="SV")
    sv_shapiq = explainer_shapiq.explain(x=x_explain)
    sv_shapiq_values = sv_shapiq.get_n_order_values(1)
    baseline_shapiq = sv_shapiq.baseline_value

    assert baseline_shap == pytest.approx(baseline_shapiq, rel=1e-6)
    assert np.allclose(sv_shap, sv_shapiq_values, rtol=1e-5)
