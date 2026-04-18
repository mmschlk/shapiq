"""Protocol, SHAP comparison, and edge-case tests for the tree module."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.tree import TreeExplainer, TreeModel

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------
skip_if_no_xgboost = pytest.mark.skipif(
    not importlib.util.find_spec("xgboost"), reason="xgboost not installed"
)
skip_if_no_lightgbm = pytest.mark.skipif(
    not importlib.util.find_spec("lightgbm"), reason="lightgbm not installed"
)

# ---------------------------------------------------------------------------
# Shared data (module-level, generated once)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(42)
_BG_REG_X = _RNG.normal(size=(100, 7))
_BG_REG_Y = _BG_REG_X[:, 0] + 0.5 * _BG_REG_X[:, 1] + _RNG.normal(0, 0.1, size=100)
_BG_CLF_Y = (np.median(_BG_REG_Y) < _BG_REG_Y).astype(int)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dt_reg():
    from sklearn.tree import DecisionTreeRegressor

    m = DecisionTreeRegressor(max_depth=3, random_state=42)
    m.fit(_BG_REG_X, _BG_REG_Y)
    return m


@pytest.fixture(scope="module")
def rf_reg():
    from sklearn.ensemble import RandomForestRegressor

    m = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    m.fit(_BG_REG_X, _BG_REG_Y)
    return m


@pytest.fixture(scope="module")
def rf_clf():
    from sklearn.ensemble import RandomForestClassifier

    m = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    m.fit(_BG_REG_X, _BG_CLF_Y)
    return m


@pytest.fixture(scope="module")
def et_reg():
    from sklearn.ensemble import ExtraTreesRegressor

    m = ExtraTreesRegressor(n_estimators=5, max_depth=3, random_state=42)
    m.fit(_BG_REG_X, _BG_REG_Y)
    return m


@pytest.fixture(scope="module")
def xgb_reg():
    pytest.importorskip("xgboost")
    from xgboost import XGBRegressor

    m = XGBRegressor(n_estimators=3, max_depth=3, random_state=42)
    m.fit(_BG_REG_X, _BG_REG_Y)
    return m


@pytest.fixture(scope="module")
def xgb_clf():
    pytest.importorskip("xgboost")
    from xgboost import XGBClassifier

    m = XGBClassifier(n_estimators=3, max_depth=3, random_state=42)
    m.fit(_BG_REG_X, _BG_CLF_Y)
    return m


@pytest.fixture(scope="module")
def lgbm_clf():
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMClassifier

    m = LGBMClassifier(
        n_estimators=3, max_depth=3, random_state=42, verbose=-1, min_child_samples=5
    )
    m.fit(_BG_REG_X, _BG_CLF_Y)
    return m


@pytest.fixture(scope="module")
def lgbm_reg():
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMRegressor

    m = LGBMRegressor(n_estimators=3, max_depth=3, random_state=42, verbose=-1, min_child_samples=5)
    m.fit(_BG_REG_X, _BG_REG_Y)
    return m


@pytest.fixture(scope="module")
def lgbm_booster():
    pytest.importorskip("lightgbm")
    import lightgbm as lgb

    train = lgb.Dataset(_BG_REG_X, label=_BG_REG_Y)
    return lgb.train(
        {"verbose": -1, "min_data_in_leaf": 5, "max_depth": 3},
        train,
        num_boost_round=3,
    )


# ===================================================================
# Protocol: every tree model must satisfy these
# ===================================================================


# task: "regression" (predict), "proba_clf" (predict_proba[0, class_index]),
#       "basic" (only check IV shape, no efficiency)
SKLEARN_TREE_MODELS = [
    ("dt_reg", "regression", None),
    ("rf_reg", "regression", None),
    ("rf_clf", "proba_clf", 0),
    ("et_reg", "regression", None),
]

OPTIONAL_TREE_MODELS = [
    pytest.param("xgb_reg", "regression", None, marks=skip_if_no_xgboost),
    pytest.param("xgb_clf", "basic", 0, marks=skip_if_no_xgboost),
    pytest.param("lgbm_reg", "regression", None, marks=skip_if_no_lightgbm),
    pytest.param("lgbm_clf", "basic", 0, marks=skip_if_no_lightgbm),
    pytest.param("lgbm_booster", "regression", None, marks=skip_if_no_lightgbm),
]

ALL_TREE_MODELS = SKLEARN_TREE_MODELS + OPTIONAL_TREE_MODELS


def _tree_id(item) -> str:
    if hasattr(item, "values"):
        return item.values[0]
    return item[0]


@pytest.mark.parametrize(
    ("model_fixture", "task", "class_index"),
    ALL_TREE_MODELS,
    ids=[_tree_id(t) for t in ALL_TREE_MODELS],
)
class TestTreeProtocol:
    """Universal contract checks for tree explainers across model types."""

    def test_explain_returns_interaction_values(self, model_fixture, task, class_index, request):
        model = request.getfixturevalue(model_fixture)
        explainer = TreeExplainer(model=model, max_order=2, min_order=1, class_index=class_index)
        x = _BG_REG_X[0]
        result = explainer.explain(x)

        assert isinstance(result, InteractionValues)
        assert result.max_order == 2
        assert result.n_players == _BG_REG_X.shape[1]

    def test_efficiency(self, model_fixture, task, class_index, request):
        """sum(values) == prediction for SV (regression / sklearn classifier)."""
        if task == "basic":
            pytest.skip("Efficiency check not applicable for booster classifiers")
        model = request.getfixturevalue(model_fixture)
        explainer = TreeExplainer(
            model=model, max_order=1, min_order=0, index="SV", class_index=class_index
        )
        x = _BG_REG_X[0]
        result = explainer.explain(x)

        if task == "regression":
            prediction = float(model.predict(x.reshape(1, -1))[0])
        else:  # proba_clf
            prediction = float(model.predict_proba(x.reshape(1, -1))[0, class_index])

        assert float(np.sum(result.values)) == pytest.approx(prediction, rel=1e-4, abs=1e-6)

    def test_baseline_matches_empty_prediction(self, model_fixture, task, class_index, request):
        model = request.getfixturevalue(model_fixture)
        explainer = TreeExplainer(
            model=model, max_order=1, min_order=0, index="SV", class_index=class_index
        )
        expected_baseline = sum(te.empty_prediction for te in explainer._treeshapiq_explainers)
        assert explainer.baseline_value == pytest.approx(expected_baseline)


# ===================================================================
# Manual TreeModel test (no sklearn dependency)
# ===================================================================


class TestManualTreeModel:
    """Test TreeExplainer with a hand-crafted TreeModel."""

    def test_against_known_values(self):
        """Verify SV computation against known SHAP library values."""
        children_left = np.asarray([1, 2, 3, -1, -1, -1, 7, -1, -1])
        children_right = np.asarray([6, 5, 4, -1, -1, -1, 8, -1, -1])
        features = np.asarray([0, 1, 0, -2, -2, -2, 2, -2, -2])
        thresholds = np.asarray([0, 0, -0.5, -2, -2, -2, 0, -2, -2])
        node_sample_weight = np.asarray([100, 50, 38, 15, 23, 12, 50, 20, 30])
        values = [110, 105, 95, 20, 50, 100, 75, 10, 40]
        values = np.asarray([v / max(values) for v in values])

        tree_model = TreeModel(
            children_left=children_left,
            children_right=children_right,
            children_missing=children_left,
            features=features,
            thresholds=thresholds,
            node_sample_weight=node_sample_weight,
            values=values,
        )

        x = np.asarray([-1, -0.5, 1, 0])
        explainer = TreeExplainer(model=tree_model, max_order=1, min_order=1, index="SV")
        result = explainer.explain(x)

        assert result[(0,)] == pytest.approx(-0.09263158, abs=1e-4)
        assert result[(1,)] == pytest.approx(-0.12100478, abs=1e-4)
        assert result[(2,)] == pytest.approx(0.02727273, abs=1e-4)
        assert result[(3,)] == pytest.approx(0.0, abs=1e-4)

    def test_sv_warning_for_order_2(self):
        """SV with max_order > 1 should warn."""
        children_left = np.asarray([1, -1, -1])
        children_right = np.asarray([2, -1, -1])
        features = np.asarray([0, -2, -2])
        thresholds = np.asarray([0.0, -2.0, -2.0])
        node_sample_weight = np.asarray([10, 5, 5])
        values = np.asarray([0.5, 0.3, 0.7])

        tree_model = TreeModel(
            children_left=children_left,
            children_right=children_right,
            children_missing=children_left,
            features=features,
            thresholds=thresholds,
            node_sample_weight=node_sample_weight,
            values=values,
        )
        with pytest.warns(UserWarning):
            TreeExplainer(model=tree_model, max_order=2, min_order=1, index="SV")


# ===================================================================
# Edge cases (regression tests for past bugs)
# ===================================================================


class TestTreeEdgeCases:
    """Regression tests for specific bugs fixed in the tree module."""

    def test_high_dimensional_indices_do_not_overflow(self):
        """Regression: int64 indices with >127 features (was overflowing to int8)."""
        from shapiq.approximator.proxy.proxyshap import MSRBiased
        from shapiq.tree.interventional.cext import compute_interactions_sparse

        n_features = 170
        approximator = MSRBiased(n=n_features, max_order=1, index="SV")
        coalition_matrix = np.zeros((3, n_features), dtype=np.int64)
        coalition_matrix[0, :5] = 1
        coalition_matrix[1, 100:110] = 1
        coalition_matrix[2, 160:] = 1

        e_matrix, r_matrix, e_counts, r_counts = approximator._coalitions_to_tree_paths(
            coalition_matrix
        )

        assert e_matrix.dtype == np.int64
        assert r_matrix.dtype == np.int64

        coalition_values = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        interactions = compute_interactions_sparse(
            coalition_values, e_matrix, r_matrix, e_counts, r_counts, "SV", n_features, 1
        )
        assert isinstance(interactions, dict)
        assert all(0 <= f < n_features for key in interactions for f in key)

    def test_repeated_flatten_calls_no_segfault(self):
        """Regression: refcount corruption in C-extension flatten output."""
        from shapiq.tree.interventional.cext import compute_interactions_flatten

        n_features = 200
        n_iterations = n_features
        leaf_predictions = np.ones(n_iterations, dtype=np.float32)
        features = np.arange(n_features, dtype=np.int64)
        e_sizes = np.ones(n_iterations, dtype=np.int64)
        r_sizes = np.zeros(n_iterations, dtype=np.int64)
        feature_in_e = np.ones(n_iterations, dtype=np.int64)
        leaf_id = np.zeros(n_iterations, dtype=np.int64)

        for _ in range(5):
            out = compute_interactions_flatten(
                leaf_predictions,
                features,
                e_sizes,
                r_sizes,
                feature_in_e,
                leaf_id,
                "SV",
                n_iterations,
                n_features,
                n_iterations,
                1,
                0,
                1.0,
            )
            assert len(out) == n_features


# ===================================================================
# Known gap: native xgboost.Booster conversion is unimplemented
# ===================================================================


class TestXGBoostBoosterUnsupported:
    """Pin the current behavior for native ``xgboost.Booster`` inputs.

    The ``delayed_register`` hook at ``src/shapiq/tree/conversion/__init__.py``
    fires on first use, but ``convert_xgboost_model`` only handles the
    sklearn wrappers (``XGBRegressor`` / ``XGBClassifier``) — passing a raw
    ``Booster`` raises ``TypeError``. This test pins that behavior so we
    reverse-alarm when native ``Booster`` support is added.
    """

    @skip_if_no_xgboost
    def test_raises_type_error(self):
        import xgboost as xgb

        dtrain = xgb.DMatrix(_BG_REG_X, label=_BG_REG_Y)
        booster = xgb.train({"max_depth": 3}, dtrain, num_boost_round=3)

        with pytest.raises(TypeError, match="not supported"):
            TreeExplainer(model=booster, max_order=1, index="SV").explain(_BG_REG_X[0])
