"""Tests for the treelite-backed conversion of scikit-learn's gradient boosting models.

The four model families covered here (``GradientBoostingRegressor``,
``GradientBoostingClassifier``, ``HistGradientBoostingRegressor``, and
``HistGradientBoostingClassifier``) are converted through
:mod:`shapiq.tree.conversion.treelite`. The converted ensembles model the *raw* model output --
``predict`` for the regressors and ``decision_function`` for the classifiers -- so every test in
this module compares against that, in line with shapiq's other gradient boosting converters.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)

import shapiq
from shapiq.explainer.utils import get_predict_function_and_model_type
from shapiq.tree.conversion import convert_tree_model
from shapiq.tree.conversion.treelite import convert_treelite_model
from shapiq.tree.validation import SUPPORTED_MODELS, validate_tree_model
from shapiq.utils import safe_isinstance
from tests.shapiq.markers import skip_if_no_treelite

pytestmark = skip_if_no_treelite

RANDOM_STATE = 42
N_ESTIMATORS = 4
MAX_DEPTH = 3

REGRESSORS = [GradientBoostingRegressor, HistGradientBoostingRegressor]
CLASSIFIERS = [GradientBoostingClassifier, HistGradientBoostingClassifier]

MODEL_CLASS_PATHS = [
    "sklearn.ensemble.GradientBoostingRegressor",
    "sklearn.ensemble._gb.GradientBoostingRegressor",
    "sklearn.ensemble.GradientBoostingClassifier",
    "sklearn.ensemble._gb.GradientBoostingClassifier",
    "sklearn.ensemble.HistGradientBoostingRegressor",
    "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor",
    "sklearn.ensemble.HistGradientBoostingClassifier",
    "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier",
]

TREE_MODEL_PATH = ["shapiq.tree.base.TreeModel"]


def _fit(model_class, X, y):
    """Fit one of the four gradient boosting classes with a comparable configuration."""
    kwargs = (
        {"max_iter": N_ESTIMATORS, "early_stopping": False}
        if "Hist" in model_class.__name__
        else {"n_estimators": N_ESTIMATORS}
    )
    return model_class(random_state=RANDOM_STATE, max_depth=MAX_DEPTH, **kwargs).fit(X, y)


def _predict_tree_ensemble(trees, data: np.ndarray) -> np.ndarray:
    """Predict raw outputs from a converted tree ensemble."""
    return np.asarray([sum(tree.predict_one(x) for tree in trees) for x in data])


def _raw_output(model, data: np.ndarray, class_index: int | None = None) -> np.ndarray:
    """Return the raw (link-space) output the converted ensemble is supposed to reproduce."""
    if not hasattr(model, "decision_function"):
        return np.asarray(model.predict(data))
    raw = np.asarray(model.decision_function(data))
    if raw.ndim == 1:  # binary classification: a single log-odds column
        return raw
    return raw[:, 1 if class_index is None else class_index]


@pytest.fixture
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a small regression dataset."""
    return make_regression(n_samples=150, n_features=6, random_state=RANDOM_STATE)


@pytest.fixture
def binary_clf_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a small binary classification dataset."""
    return make_classification(
        n_samples=150,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=RANDOM_STATE,
    )


@pytest.fixture
def multiclass_clf_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a small three-class classification dataset."""
    return make_classification(
        n_samples=150,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_class", REGRESSORS)
def test_regressor_conversion_reproduces_predict(model_class, reg_data):
    """Test the converted regressors sum to the model's own prediction."""
    X, y = reg_data
    model = _fit(model_class, X, y)
    converted = convert_tree_model(model)

    assert len(converted) == N_ESTIMATORS
    assert all(safe_isinstance(tree, TREE_MODEL_PATH) for tree in converted)
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted, X[:20]),
        model.predict(X[:20]),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("model_class", CLASSIFIERS)
def test_binary_classifier_conversion_reproduces_decision_function(model_class, binary_clf_data):
    """Test the converted binary classifiers sum to the model's log-odds."""
    X, y = binary_clf_data
    model = _fit(model_class, X, y)
    converted = convert_tree_model(model)

    assert len(converted) == N_ESTIMATORS
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted, X[:20]),
        model.decision_function(X[:20]),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("model_class", CLASSIFIERS)
def test_multiclass_conversion_selects_requested_class(model_class, multiclass_clf_data):
    """Test multiclass conversion keeps only the trees of the requested class."""
    X, y = multiclass_clf_data
    model = _fit(model_class, X, y)
    expected = model.decision_function(X[:20])

    for class_label in range(3):
        converted = convert_tree_model(model, class_label=class_label)
        # one tree per boosting round and class; only the requested class is returned
        assert len(converted) == N_ESTIMATORS
        np.testing.assert_allclose(
            _predict_tree_ensemble(converted, X[:20]),
            expected[:, class_label],
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.parametrize("model_class", CLASSIFIERS)
def test_multiclass_conversion_defaults_to_class_one(model_class, multiclass_clf_data):
    """Test the default class of a multiclass conversion is class ``1``."""
    X, y = multiclass_clf_data
    model = _fit(model_class, X, y)

    np.testing.assert_allclose(
        _predict_tree_ensemble(convert_tree_model(model), X[:10]),
        _predict_tree_ensemble(convert_tree_model(model, class_label=1), X[:10]),
    )


@pytest.mark.parametrize("model_class", CLASSIFIERS)
def test_out_of_range_class_label_raises(model_class, multiclass_clf_data):
    """Test an out-of-range class label fails loudly instead of silently picking a class."""
    X, y = multiclass_clf_data
    model = _fit(model_class, X, y)

    with pytest.raises(ValueError, match="out of range"):
        convert_tree_model(model, class_label=7)


@pytest.mark.parametrize("model_class", [*REGRESSORS, *CLASSIFIERS])
def test_conversion_uses_sklearn_split_semantics(model_class, multiclass_clf_data):
    """Test the split comparison is read from treelite rather than assumed."""
    X, y = multiclass_clf_data
    model = _fit(model_class, X, y)

    for tree in convert_tree_model(model):
        assert tree.decision_type == "<="


@pytest.mark.parametrize("model_class", REGRESSORS)
def test_split_less_trees_convert_to_constants(model_class):
    """Test bare root-leaf trees, which boosters emit once the loss stops improving, convert."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(60, 4))
    y = np.full(60, 3.0)  # a constant target gives every booster nothing left to split on
    model = _fit(model_class, X, y)

    converted = convert_tree_model(model)

    assert all(tree.n_nodes == 1 for tree in converted)
    assert all(tree.n_features_in_tree == 0 for tree in converted)
    np.testing.assert_allclose(_predict_tree_ensemble(converted, X[:10]), model.predict(X[:10]))


def test_hist_gradient_boosting_missing_values_are_routed():
    """Test ``children_missing`` reproduces the model's own NaN routing."""
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = make_regression(n_samples=200, n_features=5, random_state=RANDOM_STATE)
    X[rng.random(X.shape) < 0.2] = np.nan
    model = _fit(HistGradientBoostingRegressor, X, y)

    converted = convert_tree_model(model)

    assert np.isnan(X[:20]).any()
    np.testing.assert_allclose(
        _predict_tree_ensemble(converted, X[:20]),
        model.predict(X[:20]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_categorical_splits_raise_a_clear_error():
    """Test categorical splits, which shapiq's TreeModel cannot represent, fail loudly."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.integers(0, 4, size=(300, 3)).astype(float)
    y = (X[:, 0] + X[:, 1] > 3).astype(int)
    model = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_depth=MAX_DEPTH,
        max_iter=N_ESTIMATORS,
        early_stopping=False,
        categorical_features=[0, 1],
    ).fit(X, y)

    with pytest.raises(ValueError, match="categorical splits"):
        convert_tree_model(model)


def test_generic_converter_handles_vector_valued_leaves(multiclass_clf_data):
    """Test the generic treelite converter still handles model families it is not registered for.

    Classification forests store a per-class vote vector per leaf and average their trees, a
    layout none of the four registered gradient boosting classes produce. scikit-learn forests
    keep their own faster converter, so this only guards the generic code path.
    """
    treelite = pytest.importorskip("treelite")

    X, y = multiclass_clf_data
    model = RandomForestClassifier(
        random_state=RANDOM_STATE, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS
    ).fit(X, y)
    expected = model.predict_proba(X[:20])

    for class_label in range(3):
        converted = convert_treelite_model(
            treelite.sklearn.import_model(model), class_label=class_label
        )
        assert len(converted) == N_ESTIMATORS
        np.testing.assert_allclose(
            _predict_tree_ensemble(converted, X[:20].astype(np.float32)),
            expected[:, class_label],
            rtol=1e-6,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# pipeline wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_class_path", MODEL_CLASS_PATHS)
def test_model_classes_are_registered_as_tree_models(model_class_path):
    """Test the four gradient boosting classes are recognized throughout the pipeline."""
    assert model_class_path in SUPPORTED_MODELS
    _, model_type = get_predict_function_and_model_type(Mock(), model_class_path)
    assert model_type == "tree"


@pytest.mark.parametrize("model_class", [*REGRESSORS, *CLASSIFIERS])
def test_validate_tree_model_accepts_gradient_boosting(model_class, multiclass_clf_data):
    """Test ``validate_tree_model`` converts the models instead of rejecting them."""
    X, y = multiclass_clf_data
    model = _fit(model_class, X, y)

    trees = validate_tree_model(model)

    assert isinstance(trees, list)
    assert all(safe_isinstance(tree, TREE_MODEL_PATH) for tree in trees)


@pytest.mark.parametrize("model_class", [*REGRESSORS, *CLASSIFIERS])
def test_explainer_dispatches_to_tree_explainer(model_class, multiclass_clf_data):
    """Test ``shapiq.Explainer`` picks the TreeExplainer for these models."""
    X, y = multiclass_clf_data
    model = _fit(model_class, X, y)

    explainer = shapiq.Explainer(model=model, data=X, index="SV", max_order=1)

    assert isinstance(explainer, shapiq.TreeExplainer)


# ---------------------------------------------------------------------------
# TreeExplainer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", ["SV", "k-SII"])
@pytest.mark.parametrize("model_class", [*REGRESSORS, *CLASSIFIERS])
def test_tree_explainer_is_efficient(model_class, index, multiclass_clf_data):
    """Test the interaction values plus the baseline add up to the raw model output."""
    X, y = multiclass_clf_data
    model = _fit(model_class, X, y)
    max_order = 1 if index == "SV" else 2
    explainer = shapiq.TreeExplainer(model=model, index=index, max_order=max_order)
    expected = _raw_output(model, X[:5])

    for i in range(5):
        interaction_values = explainer.explain(X[i])
        total = sum(
            value for interaction, value in interaction_values.dict_values.items() if interaction
        )
        assert total + interaction_values.baseline_value == pytest.approx(expected[i], abs=1e-6)


@pytest.mark.parametrize("model_class", CLASSIFIERS)
def test_tree_explainer_class_index_selects_the_class(model_class, multiclass_clf_data):
    """Test ``class_index`` propagates through the explainer to the converter."""
    X, y = multiclass_clf_data
    model = _fit(model_class, X, y)

    for class_index in range(3):
        explainer = shapiq.TreeExplainer(
            model=model, index="SV", max_order=1, class_index=class_index
        )
        expected = _raw_output(model, X[:5], class_index=class_index)
        for i in range(5):
            interaction_values = explainer.explain(X[i])
            total = sum(
                value
                for interaction, value in interaction_values.dict_values.items()
                if interaction
            )
            assert total + interaction_values.baseline_value == pytest.approx(expected[i], abs=1e-6)


@pytest.mark.parametrize("model_class", [*REGRESSORS, *CLASSIFIERS])
def test_tree_explainer_baseline_matches_mean_raw_output(model_class, multiclass_clf_data):
    """Test the path-dependent baseline is the ensemble's weighted mean leaf value."""
    X, y = multiclass_clf_data
    model = _fit(model_class, X, y)
    explainer = shapiq.TreeExplainer(model=model, index="SV", max_order=1)

    expected = sum(tree.empty_prediction for tree in explainer._trees)

    assert explainer.baseline_value == pytest.approx(expected)
    assert np.isfinite(explainer.baseline_value)


@pytest.mark.parametrize("model_class", REGRESSORS)
def test_tree_explainer_interventional_mode(model_class, reg_data):
    """Test interventional mode runs on the converted ensembles and stays efficient."""
    X, y = reg_data
    model = _fit(model_class, X, y)
    explainer = shapiq.TreeExplainer(
        model=model,
        mode="interventional",
        reference_dataset=X[:50],
        index="SV",
        max_order=1,
    )

    interaction_values = explainer.explain(X[0])
    total = sum(
        value for interaction, value in interaction_values.dict_values.items() if interaction
    )

    assert total + interaction_values.baseline_value == pytest.approx(
        model.predict(X[:1])[0], abs=1e-4
    )
