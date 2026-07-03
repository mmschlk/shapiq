"""Tests for shapiq_benchmark.setup (loading datasets and models from strings).

These run fully in-memory: no dataset downloads and no optional model backends,
so they exercise the always-available scikit-learn path and the shipped
hyperparameter configs.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq_benchmark.bench_types import BenchmarkDataset
from shapiq_benchmark.setup import (
    check_for_known_combination,
    infer_data_type,
    load_and_fit_model_from_str,
)


@pytest.fixture
def tiny_dataset() -> BenchmarkDataset:
    """A tiny in-memory classification dataset (no network / data download)."""
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(20, 4))
    x_test = rng.normal(size=(5, 4))
    return BenchmarkDataset(
        x_train=x_train,
        y_train=(x_train[:, 0] > 0).astype(int),
        x_test=x_test,
        y_test=(x_test[:, 0] > 0).astype(int),
        data_type="classification",
    )


def test_load_and_fit_decision_tree(tiny_dataset) -> None:
    """A scikit-learn model fits end-to-end from a string id, no optional deps."""
    model = load_and_fit_model_from_str("decision_tree", tiny_dataset)
    assert isinstance(model, DecisionTreeClassifier)
    preds = model.predict(tiny_dataset.x_test)
    assert preds.shape == (tiny_dataset.x_test.shape[0],)


def test_load_and_fit_unsupported_model_raises(tiny_dataset) -> None:
    """An unknown model id raises a ValueError that lists the supported models."""
    with pytest.raises(ValueError, match="Unsupported model"):
        load_and_fit_model_from_str("not_a_real_model", tiny_dataset)


def test_infer_data_type() -> None:
    """``infer_data_type`` distinguishes classifiers from regressors."""
    assert infer_data_type(DecisionTreeClassifier()) == "classification"
    assert infer_data_type(DecisionTreeRegressor()) == "regression"


def test_check_for_known_combination_reads_shipped_configs() -> None:
    """The shipped optimization JSONs are packaged, found, and parsed."""
    known = check_for_known_combination("california_housing", "xgboost")
    assert isinstance(known, dict)
    assert "n_estimators" in known
    # An unknown (dataset, model) pair resolves to None.
    assert check_for_known_combination("not_a_dataset", "not_a_model") is None
