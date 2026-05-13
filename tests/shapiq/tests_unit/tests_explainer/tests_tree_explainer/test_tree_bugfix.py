"""This test module contains all tests for bugfixes regarding TreeSHAP-IQ."""

from __future__ import annotations

import locale
import struct
from pathlib import Path

import numpy as np
import pytest

from shapiq.tree import TreeExplainer, TreeModel, TreeSHAPIQ

_LIGHTGBM_UBUNTU_BYTES_FILE = (
    Path(__file__).parent.parent.parent.parent / "data" / "models" / "lightgbm_ubuntu.bytes"
)


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
        "children_missing": np.asarray(
            children_left
        ),  # intentionally set to left_children to test if it is ignored
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
        ],
    )

    tree_explainer = TreeSHAPIQ(model=tree_model, index="SII", max_order=2, min_order=1)
    tree_explainer.explain(x_explain)  # bug appears for node 22

    # if this test runs without an error, the bug is fixed
    assert True


def test_xgboost_multiclass_base_score():
    """Test that XGBoost 3 multi-class base_score array is correctly parsed per class.

    In XGBoost 3, base_score for multi-class models is serialized as a per-class typed array
    instead of a single scalar. Previously readBaseScoreOrZero() would fall back to 0.0,
    breaking the efficiency property of Shapley values.
    """
    import xgboost as xgb
    from sklearn.datasets import make_classification

    X, y = make_classification(
        random_state=42, n_samples=200, n_features=5, n_classes=3, n_informative=5, n_redundant=0
    )
    model = xgb.XGBClassifier(
        random_state=42, n_estimators=10, max_depth=2, objective="multi:softprob", num_class=3
    )
    model.fit(X, y)

    x_explain = X[0]
    booster = model.get_booster()

    raw_scores = booster.predict(xgb.DMatrix(x_explain.reshape(1, -1)), output_margin=True)[0]

    for class_idx in range(3):
        explainer = TreeExplainer(model=model, max_order=1, index="SV", class_index=class_idx)
        sv = explainer.explain(x_explain)
        # TreeExplainer defaults to min_order=0, so sv[()] = baseline_value is included in
        # sv.values. The efficiency property is therefore: sv.values.sum() == raw_score.
        # (Adding sv.baseline_value again would double-count it.)
        assert sv.baseline_value != 0.0, (
            f"baseline_value is 0.0 for class {class_idx}, indicating base_score was not read correctly"
        )
        efficiency = sv.values.sum()
        assert pytest.approx(efficiency, rel=1e-4) == raw_scores[class_idx], (
            f"Efficiency failed for class {class_idx}: {efficiency} != {raw_scores[class_idx]}. "
            f"baseline_value={sv.baseline_value}"
        )


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
    """Test that the xgboost model does not predict with the correct leaf node.

    This test illustrates that the predictions of the xgboost model do not perfectly align
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
    data_df = booster.trees_to_dataframe()
    threshold = data_df[data_df["Node"] == 0]["Split"].values[0]
    feature_id = data_df[data_df["Node"] == 0]["Feature"].values[0]
    intercept = model.intercept_[0]
    prediction_left_df = data_df[data_df["Node"] == 1]["Gain"].values[0] + intercept
    prediction_right_df = data_df[data_df["Node"] == 2]["Gain"].values[0] + intercept

    # make sure the xgboost model is using the features we are playing around with
    assert feature_id == "f1"  # feature 1 is used
    assert x_explain[1] < threshold  # feature value is < threshold (this instance should go left)
    assert len(data_df) == 3  # only 3 nodes in the tree (one decision node and two leaf nodes)

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
    assert pytest.approx(efficiency, rel=0.0001) != prediction_right_df


def test_lightgbm_locale_bug():
    """Regression test: LightGBM conversion must work regardless of LC_NUMERIC locale.

    On Ubuntu (and any system) with a non-C LC_NUMERIC locale (e.g. de_DE where
    ',' is the decimal separator), the C extension's strtod calls would fail to
    parse the '.' decimal separator used in LightGBM's model_to_string() output.
    This test reproduces that failure by temporarily switching to the German
    locale and verifies that the fix (locale-independent strtod_l) makes the
    conversion succeed with correct predictions.

    Bytes file: tests/shapiq/data/models/lightgbm_ubuntu.bytes
    (generated from a real LightGBM regression model on a bike-sharing dataset)
    """
    from shapiq.tree.conversion.cext import parse_lightgbm_string_treemodels

    if not _LIGHTGBM_UBUNTU_BYTES_FILE.exists():
        pytest.skip("lightgbm_ubuntu.bytes test data file not found")

    byte_data = _LIGHTGBM_UBUNTU_BYTES_FILE.read_bytes()

    # Try to switch to German locale to reproduce the original Ubuntu bug.
    # If the locale is not available on this platform, still run the basic test.
    original_lc_numeric = locale.getlocale(locale.LC_NUMERIC)
    locale_switched = False
    try:
        locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
        locale_switched = True
    except locale.Error:
        pass  # locale not available; test will still validate basic correctness

    try:
        trees = parse_lightgbm_string_treemodels(byte_data, -1)
    finally:
        # Always restore the original locale
        try:
            locale.setlocale(locale.LC_NUMERIC, original_lc_numeric[0] or "C")
        except locale.Error:
            locale.setlocale(locale.LC_NUMERIC, "C")

    assert len(trees) == 100, f"Expected 100 trees, got {len(trees)}"

    # Verify leaf values are non-zero (sanity check that parsing succeeded)
    tree0 = trees[0]
    leaf_values = tree0.values[tree0.leaf_mask]
    assert len(leaf_values) == 31
    assert not np.all(leaf_values == 0), "All leaf values are zero — parsing likely failed"

    # Verify the expected leaf values from the model (first three leaves of tree 0)
    expected_first_leaves = np.array([174.53361, 181.49871, 182.43762], dtype=np.float32)
    np.testing.assert_allclose(leaf_values[:3], expected_first_leaves, rtol=1e-4)

    # Shapley-value efficiency: sum of SV (with min_order=0) must equal the prediction.
    # Use a deterministic test point and check the efficiency property for all 100 trees.
    rng = np.random.default_rng(0)
    x = rng.random(12).astype(np.float32)

    from shapiq import TreeExplainer

    # Reconstruct a Booster from the bytes to drive TreeExplainer
    try:
        import lightgbm as lgb
    except ImportError:
        pytest.skip("lightgbm not installed")

    booster = lgb.Booster(model_str=byte_data.decode("utf-8"))
    lgb_pred = booster.predict(x.reshape(1, -1))[0]

    explainer = TreeExplainer(model=booster, max_order=1, min_order=0, index="SV")
    sv = explainer.explain(x)
    shapiq_pred = sv.values.sum()

    assert pytest.approx(shapiq_pred, rel=1e-4) == lgb_pred, (
        f"TreeExplainer prediction {shapiq_pred} does not match LightGBM {lgb_pred}"
    )
    if locale_switched:
        assert True, "Test passed with German locale — locale bug is fixed"


@pytest.mark.external_libraries
def test_xgboost_locale_bug():
    """Regression test: XGBoost binary (UBJSON) conversion must work regardless of LC_NUMERIC.

    ByteStream::readDoubleLike() uses strtod for 'S' and 'H' string-encoded floats.
    On systems with a non-C LC_NUMERIC locale those calls would silently produce 0.0
    or raise a parse error.  This test reproduces that scenario by temporarily switching
    to the German locale and verifies the fix (strtod_c) makes conversion succeed with
    correct Shapley-value efficiency.
    """
    import xgboost as xgb
    from sklearn.datasets import make_regression

    X, y = make_regression(random_state=42, n_samples=100, n_features=5)
    model = xgb.XGBRegressor(random_state=42, n_estimators=10, max_depth=3)
    model.fit(X, y)

    x_explain = X[0]
    raw_score = model.get_booster().predict(
        xgb.DMatrix(x_explain.reshape(1, -1)), output_margin=True
    )[0]

    original_lc_numeric = locale.getlocale(locale.LC_NUMERIC)
    locale_switched = False
    try:
        locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
        locale_switched = True
    except locale.Error:
        pass  # locale not available; test still validates correctness

    try:
        explainer = TreeExplainer(model=model, max_order=1, min_order=0, index="SV")
        sv = explainer.explain(x_explain)
    finally:
        try:
            locale.setlocale(locale.LC_NUMERIC, original_lc_numeric[0] or "C")
        except locale.Error:
            locale.setlocale(locale.LC_NUMERIC, "C")

    efficiency = sv.values.sum()
    assert pytest.approx(efficiency, rel=1e-4) == raw_score, (
        f"Efficiency failed: {efficiency} != {raw_score}"
    )
    if locale_switched:
        assert True, "Test passed with German locale — XGBoost locale bug is fixed"


@pytest.mark.external_libraries
def test_xgboost_multiclass_locale_bug():
    """Regression test: XGBoost multiclass base_score parsing must ignore LC_NUMERIC.

    In XGBoost 3, base_score for multi-class models is a CSV string array.
    ByteStream::readBaseScoreOrZero() calls strtod on each token, which is locale-sensitive.
    On de_DE.UTF-8 this produces 0.0, breaking the efficiency property.
    This test reproduces that scenario and verifies the strtod_c fix.
    """
    import xgboost as xgb
    from sklearn.datasets import make_classification

    X, y = make_classification(
        random_state=42, n_samples=200, n_features=5, n_classes=3, n_informative=5, n_redundant=0
    )
    model = xgb.XGBClassifier(
        random_state=42, n_estimators=10, max_depth=2, objective="multi:softprob", num_class=3
    )
    model.fit(X, y)

    x_explain = X[0]
    raw_scores = model.get_booster().predict(
        xgb.DMatrix(x_explain.reshape(1, -1)), output_margin=True
    )[0]

    original_lc_numeric = locale.getlocale(locale.LC_NUMERIC)
    locale_switched = False
    try:
        locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
        locale_switched = True
    except locale.Error:
        pass  # locale not available; test still validates correctness

    try:
        for class_idx in range(3):
            explainer = TreeExplainer(
                model=model, max_order=1, min_order=0, index="SV", class_index=class_idx
            )
            sv = explainer.explain(x_explain)
            efficiency = sv.values.sum()
            assert pytest.approx(efficiency, rel=1e-4) == raw_scores[class_idx], (
                f"Efficiency failed for class {class_idx}: {efficiency} != {raw_scores[class_idx]}"
            )
    finally:
        try:
            locale.setlocale(locale.LC_NUMERIC, original_lc_numeric[0] or "C")
        except locale.Error:
            locale.setlocale(locale.LC_NUMERIC, "C")

    if locale_switched:
        assert True, "Test passed with German locale — XGBoost multiclass locale bug is fixed"


@pytest.mark.external_libraries
def test_lightgbm_multiclass_locale_bug():
    """Regression test: LightGBM multiclass conversion must work regardless of LC_NUMERIC.

    The existing test_lightgbm_locale_bug covers a regression model.  This test extends
    locale coverage to a multiclass LightGBM model, exercising class_label selection
    in the string-stream parser under de_DE.UTF-8 locale.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        pytest.skip("lightgbm not installed")

    from sklearn.datasets import make_classification

    X, y = make_classification(
        random_state=42, n_samples=300, n_features=5, n_classes=3, n_informative=5, n_redundant=0
    )
    X = X.astype(np.float32)
    model = lgb.LGBMClassifier(
        random_state=42, n_estimators=10, max_depth=3, num_class=3, verbose=-1
    )
    model.fit(X, y)

    x_explain = X[0]
    # Use raw margin (pre-softmax) scores — Shapley values are additive on the raw output space
    raw_scores = model.booster_.predict(x_explain.reshape(1, -1), raw_score=True)[0]

    original_lc_numeric = locale.getlocale(locale.LC_NUMERIC)
    locale_switched = False
    try:
        locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
        locale_switched = True
    except locale.Error:
        pass  # locale not available; test still validates correctness

    try:
        explainer = TreeExplainer(model=model, max_order=1, min_order=0, index="SV", class_index=0)
        sv = explainer.explain(x_explain)
        # Efficiency: sv.values.sum() == model prediction for class 0
        efficiency = sv.values.sum()
    finally:
        try:
            locale.setlocale(locale.LC_NUMERIC, original_lc_numeric[0] or "C")
        except locale.Error:
            locale.setlocale(locale.LC_NUMERIC, "C")

    assert pytest.approx(efficiency, rel=1e-3) == raw_scores[0], (
        f"Efficiency failed for class 0: {efficiency} != {raw_scores[0]}"
    )
    if locale_switched:
        assert True, "Test passed with German locale — LightGBM multiclass locale bug is fixed"


# ── bounds-check tests (converter.cc buffer-overread fixes) ──────────────────


def _build_xgb_ubjson(n_nodes: int = 3) -> bytes:
    """Build a minimal XGBoost-like UBJSON byte sequence with one tree of *n_nodes* nodes.

    Uses the same key ordering that ``ByteStream.extractTreeStructure()`` expects,
    so the normal parser code path is exercised end-to-end.  All leaf/node values
    are 1.0; children are -1 (every node is treated as a leaf sentinel).
    """

    def _key(name: str) -> bytes:
        """Encode a UBJSON string key: int8-marker + length byte + name bytes."""
        return bytes([0x69, len(name)]) + name.encode()

    def _i8(v: int) -> bytes:
        """Encode a UBJSON int8 scalar value."""
        return bytes([0x69, v & 0xFF])

    def _f64_arr(vals: list) -> bytes:
        """UBJSON typed float64 array: [$D#i<n> <big-endian doubles>."""
        return b"[$D#" + bytes([0x69, len(vals)]) + b"".join(struct.pack(">d", v) for v in vals)

    def _i32_arr(vals: list) -> bytes:
        """UBJSON typed int32 array: [$l#i<n> <big-endian int32>."""
        return b"[$l#" + bytes([0x69, len(vals)]) + b"".join(struct.pack(">i", v) for v in vals)

    _EMPTY = b"[$i#U\x00"  # typed int8, 0 elements

    n = n_nodes
    buf = bytearray()
    # Top-level scanner keys
    buf += _key("num_trees") + _i8(1)
    buf += _key("trees") + b"["  # plain array — no count prefix
    # Tree object (must start with '{' so require_object_start guard passes)
    buf += b"{"
    buf += _key("base_weights") + _f64_arr([1.0] * n)
    buf += _key("categories") + _EMPTY
    buf += _key("categories_nodes") + _EMPTY
    buf += _key("categories_segments") + _EMPTY
    buf += _key("categories_sizes") + _EMPTY
    buf += _key("default_left") + _i32_arr([0] * n)
    buf += _key("id") + _i8(0)
    buf += _key("left_children") + _i32_arr([-1] * n)
    buf += _key("loss_changes") + _EMPTY
    buf += _key("parents") + _EMPTY
    buf += _key("right_children") + _i32_arr([-1] * n)
    buf += _key("split_conditions") + _f64_arr([0.5] * n)
    buf += _key("split_indices") + _i32_arr([0] * n)
    buf += _key("split_type") + _EMPTY
    buf += _key("sum_hessian") + _f64_arr([1.0] * n)
    buf += b"}]"
    return bytes(buf)


def test_xgb_ubjson_valid_parse():
    """Happy-path: minimal hand-crafted UBJSON parses without error and returns correct data."""
    from shapiq.tree.conversion.cext import parse_xgboost_ubjson

    buf = _build_xgb_ubjson(3)
    result = parse_xgboost_ubjson(buf, -1)
    assert len(result) == 8, "Expected 8-tuple of arrays"
    node_ids, feature_ids, thresholds, values, left, right, default_, weights = result
    assert len(node_ids) == 1, "Expected 1 tree"
    assert node_ids[0].shape == (3,), "Expected 3 nodes"
    np.testing.assert_array_equal(node_ids[0], [0, 1, 2])
    # base_score = 0 → each value = 1.0 (stored as float32)
    np.testing.assert_allclose(values[0], [1.0, 1.0, 1.0], rtol=1e-5)
    np.testing.assert_allclose(thresholds[0], [0.5, 0.5, 0.5], rtol=1e-5)
    # All children are -1 (leaf sentinel)
    np.testing.assert_array_equal(left[0], [-1, -1, -1])
    np.testing.assert_array_equal(right[0], [-1, -1, -1])


def test_xgb_ubjson_truncated_double_array_raises():
    """Bug #1: truncated typed float64 array in fillArray(double*) must raise RuntimeError.

    Before the fix, ``ByteStream::fillArray(double*)`` called ``std::memcpy`` without
    first checking that ``pos + num_nodes * 8 <= size``, causing a buffer over-read on
    truncated or malformed input.  The fix adds a bounds check that throws
    "End of stream" before the memcpy.
    """
    from shapiq.tree.conversion.cext import parse_xgboost_ubjson

    buf = _build_xgb_ubjson(3)
    # Keep only 1 data byte of the base_weights float64 array (need 3*8 = 24).
    hdr_idx = buf.index(b"[$D#")  # position of base_weights typed-array header
    truncated = buf[: hdr_idx + 6 + 1]  # 6-byte header + 1 data byte
    with pytest.raises(RuntimeError, match="End of stream"):
        parse_xgboost_ubjson(truncated, -1)


def test_xgb_ubjson_truncated_int32_array_raises():
    """Bug #2: truncated typed int32 array in fillArray(int64_t*) must raise RuntimeError.

    Before the fix, the fast-path bulk loop in ``ByteStream::fillArray(int64_t*)``
    called ``std::memcpy(&raw, data + pos + i*4, 4)`` without checking
    ``pos + num_nodes * 4 <= size``, causing a buffer over-read.
    """
    from shapiq.tree.conversion.cext import parse_xgboost_ubjson

    buf = _build_xgb_ubjson(3)
    # The first [$l# header in the buffer is default_left (first int32 typed array).
    hdr_idx = buf.index(b"[$l#")
    truncated = buf[: hdr_idx + 6 + 1]  # 6-byte header + 1 data byte instead of 12
    with pytest.raises(RuntimeError, match="End of stream"):
        parse_xgboost_ubjson(truncated, -1)


def test_xgb_ubjson_skip_typed_array_truncated_raises():
    """Bug #4: truncated typed array in skipArrayValue must raise RuntimeError.

    Before the fix, ``skipArrayValue`` advanced ``pos`` by ``count * elementSize``
    without checking that the buffer contained that many bytes, silently walking
    pos past the buffer end.  The fix adds an overflow guard and a bounds check
    that throw before the advance.
    """
    from shapiq.tree.conversion.cext import parse_xgboost_ubjson

    buf = _build_xgb_ubjson(3)
    # Replace the first empty array (categories [$i#U\x00) with a typed int8 array
    # that declares 5 elements but supplies zero data bytes.
    empty_idx = buf.index(b"[$i#U\x00")
    truncated = buf[:empty_idx] + b"[$i#i\x05"  # 5 elements declared, 0 bytes supplied
    with pytest.raises(RuntimeError, match="End of stream"):
        parse_xgboost_ubjson(truncated, -1)


def test_xgb_ubjson_skip_single_value_truncated_raises():
    """Bug #3: truncated element payload in skipSingleValue must raise RuntimeError.

    Before the fix, ``skipSingleValue`` advanced ``pos += N`` for fixed-width
    markers (e.g. 'D' = 8 bytes) without calling ``readByte()``, so no bounds
    check was performed and ``pos`` silently walked past the buffer end.
    The fix replaces each bare ``pos += N`` with ``N`` calls to ``readByte()``,
    each of which checks bounds.
    """
    from shapiq.tree.conversion.cext import parse_xgboost_ubjson

    buf = _build_xgb_ubjson(3)
    # Replace the first empty array (categories) with a count-prefixed array
    # ([#i\x01) containing one float64 element whose 8 data bytes are absent.
    empty_idx = buf.index(b"[$i#U\x00")
    truncated = buf[:empty_idx] + b"[#i\x01D"  # count=1, type='D', no payload bytes
    with pytest.raises(RuntimeError, match="End of stream"):
        parse_xgboost_ubjson(truncated, -1)
