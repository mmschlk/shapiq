"""This test module contains all tests for bugfixes regarding TreeSHAP-IQ."""

import numpy as np
import pytest

from shapiq.explainer.tree import TreeExplainer, TreeModel, TreeSHAPIQ


def test_bike_bug():
    """A test for the bug denoted in GH #118. Should be fixed."""

    children_left = [
        1,
        2,
        3,
        4,
        -1,
        -1,
        7,
        -1,
        -1,
        10,
        11,
        -1,
        -1,
        14,
        -1,
        -1,
        17,
        18,
        19,
        -1,
        -1,
        22,
        -1,
        -1,
        25,
        26,
        -1,
        -1,
        29,
        -1,
        -1,
    ]
    chidren_right = [
        16,
        9,
        6,
        5,
        -1,
        -1,
        8,
        -1,
        -1,
        13,
        12,
        -1,
        -1,
        15,
        -1,
        -1,
        24,
        21,
        20,
        -1,
        -1,
        23,
        -1,
        -1,
        28,
        27,
        -1,
        -1,
        30,
        -1,
        -1,
    ]
    features = [
        0,
        0,
        0,
        10,
        -2,
        -2,
        10,
        -2,
        -2,
        10,
        2,
        -2,
        -2,
        2,
        -2,
        -2,
        1,
        6,
        5,
        -2,
        -2,
        0,
        -2,
        -2,
        6,
        0,
        -2,
        -2,
        0,
        -2,
        -2,
    ]
    thresholds = [
        -0.45833333,
        -0.54166666,
        -0.875,
        0.5,
        np.nan,
        np.nan,
        0.5,
        np.nan,
        np.nan,
        0.5,
        -0.55244875,
        np.nan,
        np.nan,
        -0.23671414,
        np.nan,
        np.nan,
        -0.03125,
        0.5,
        2.5,
        np.nan,
        np.nan,
        0.625,
        np.nan,
        np.nan,
        0.5,
        0.70833334,
        np.nan,
        np.nan,
        0.70833334,
        np.nan,
        np.nan,
    ]
    node_sample_weight = [
        13903.0,
        3996.0,
        3424.0,
        1156.0,
        375.0,
        781.0,
        2268.0,
        731.0,
        1537.0,
        572.0,
        188.0,
        72.0,
        116.0,
        384.0,
        172.0,
        212.0,
        9907.0,
        4451.0,
        2297.0,
        1540.0,
        757.0,
        2154.0,
        1636.0,
        518.0,
        5456.0,
        2640.0,
        2221.0,
        419.0,
        2816.0,
        2347.0,
        469.0,
    ]
    values = [
        190.5770697,
        31.9014014,
        24.79964953,
        43.35034602,
        79.39733333,
        26.04225352,
        15.34435626,
        23.87824897,
        11.28562134,
        74.41258741,
        18.87765957,
        9.19444444,
        24.88793103,
        101.6015625,
        71.88372093,
        125.71226415,
        254.5790855,
        177.66344642,
        127.30605137,
        101.08311688,
        180.65257596,
        231.363974,
        264.46821516,
        126.81081081,
        317.32679619,
        247.60530303,
        267.30616839,
        143.17661098,
        382.69069602,
        417.93694078,
        206.30916844,
    ]

    buggy_tree_model = {
        "children_left": np.asarray(children_left),
        "children_right": np.asarray(chidren_right),
        "empty_prediction": 190.5770,
        "features": np.asarray(features),
        "thresholds": np.asarray(thresholds),
        "node_sample_weight": np.asarray(node_sample_weight),
        "values": np.asarray(values),
    }
    tree_model: TreeModel = TreeModel(**buggy_tree_model)

    x_explain = np.asarray(
        [
            0.58333333,
            0.9375,
            0.73706148,
            -1.2,
            0.0,
            0.0,
            1.0,
            5.0,
            0.0,
            6.0,
            0.0,
            0.0,
        ]
    )

    tree_explainer = TreeSHAPIQ(model=tree_model, index="SII", max_order=2, min_order=1)
    tree_explainer.explain(x_explain)  # bug appears for node 22

    # if this test runs without an error, the bug is fixed
    assert True
    return


def test_xgboost_bug():
    """Test that xgboost works when not all features are used in the tree."""
    import xgboost as xgb
    from sklearn.datasets import make_regression

    n_features_data = 7

    # fit the tree on data that does not use all features
    X, y = make_regression(random_state=42, n_samples=100, n_features=n_features_data)
    model = xgb.XGBRegressor(random_state=42, n_estimators=10, max_depth=2)
    model.fit(X, y)

    # make sure not all features are used in the tree
    booster = model.get_booster()
    data_frame = booster.trees_to_dataframe()
    feature_names = np.setdiff1d(data_frame["Feature"], "Leaf")
    n_features_in_tree = len(feature_names)
    assert booster.num_features() == n_features_data
    assert booster.num_features != n_features_in_tree

    # test the shapiq implementation
    explainer = TreeExplainer(model=model, max_order=1, index="SV")
    x_explain = X[0]
    explanation = explainer.explain(x_explain)

    for value in explanation.values:
        assert not np.isnan(value)


@pytest.mark.skip("Seems to be resolved")
def test_xgb_predicts_with_wrong_leaf_node():
    """This test illustrates that the predictions of the xgboost model do not perfectly align
    with the xgboost models internal representation.

    Sometimes the model goes in a wrong direction in the path. This test illustrates this where
    one datapoint should be predicted with a left leave node but is predicted with the right leave
    node with the xgboost model. We are parsing the xgboost model and creating our tree model
    representation, where we correctly predict with the left leave node.
    """

    from sklearn.datasets import make_regression
    from xgboost import XGBRegressor

    X, y = make_regression(n_samples=100, n_features=7, random_state=42)
    model = XGBRegressor(random_state=42, n_estimators=1, max_depth=1)
    model.fit(X, y)
    booster = model.get_booster()

    # get the data point
    x_explain = X[14]
    x_explain_left = x_explain.copy()
    x_explain_left[1] = 0.0  # set the feature value to 0.0 which is smaller than even the og. val
    assert x_explain_left[1] < x_explain[1]  # make sure the value is smaller

    # parse the xgboost model
    df = booster.trees_to_dataframe()
    threshold = df[df["Node"] == 0]["Split"].values[0]
    feature_id = df[df["Node"] == 0]["Feature"].values[0]
    intercept = model.intercept_[0]
    prediction_left_df = df[df["Node"] == 1]["Gain"].values[0] + intercept
    prediction_right_df = df[df["Node"] == 2]["Gain"].values[0] + intercept

    # make sure the xgboost model is using the features we are playing around with
    assert feature_id == "f1"  # feature 1 is used
    assert x_explain[1] < threshold  # feature value is < threshold (this instance should go left)
    assert len(df) == 3  # only 3 nodes in the tree (one decision node and two leaf nodes)

    # get the predictions of the xgboost model
    prediction_xgb = model.predict(x_explain.reshape(1, -1))
    prediction_xgb_left = model.predict(x_explain_left.reshape(1, -1))
    assert not np.allclose(prediction_xgb, prediction_xgb_left)  # predictions are different
    # the original prediction is going right not left as it should
    assert np.allclose(prediction_xgb, prediction_right_df)
    assert np.allclose(prediction_xgb, prediction_left_df)

    # get our tree model representation
    tree_explainer = TreeExplainer(model=model, index="SV")
    tree_model = tree_explainer._treeshapiq_explainers[0]._tree
    prediction_tree_model = tree_model.predict_one(x_explain)
    prediction_tree_model_left = tree_model.predict_one(x_explain_left)
    # predictions of og xgb is different from our tree model
    # where both instances are correctly predicted to be left
    assert prediction_xgb != prediction_tree_model
    assert prediction_tree_model == prediction_left_df
    assert prediction_tree_model_left == prediction_left_df
    assert prediction_tree_model != prediction_right_df

    # get the explanation of the tree model
    sv = tree_explainer.explain(x_explain)
    efficiency = sum(sv.values)
    if sv[()] == 0:
        efficiency += sv.baseline_value
    # efficiency is correct as the prediction with the tree model and not like the xgb model
    assert pytest.approx(efficiency) == prediction_tree_model
    assert pytest.approx(efficiency, rel=0.0001) == prediction_left_df
    assert not pytest.approx(efficiency, rel=0.0001) == prediction_right_df
