"""Tests for the seeded model registry."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq_benchmark.datasets import load_dataset
from shapiq_benchmark.models import MODEL_NAMES, TUNED_PRESETS, build_model, fit_model
from tests.shapiq_games.helpers import is_installed

OPTIONAL = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "tabpfn": "tabpfn",
}
CLASSIFICATION_ONLY = {"threshold_nn"}
REGRESSION_ONLY = {"gaussian_process"}


@pytest.fixture(scope="module")
def splits() -> dict[str, object]:
    return {
        "classification": load_dataset("xor", n_samples=200).split(random_state=0),
        "regression": load_dataset("independentlinear60", n_samples=200).split(random_state=0),
    }


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("name", [name for name in MODEL_NAMES if name != "tabpfn"])
def test_every_model_fits_and_is_reproducible(name: str, task: str, splits: dict) -> None:
    if name in OPTIONAL and not is_installed(OPTIONAL[name]):
        pytest.skip(f"{name} is not installed")
    split = splits[task]
    params = {"radius": 100.0} if name == "threshold_nn" else {}
    if (name in CLASSIFICATION_ONLY and task == "regression") or (
        name in REGRESSION_ONLY and task == "classification"
    ):
        with pytest.raises(ValueError, match="only available"):
            build_model(name, task)
        return
    first = fit_model(name, split, random_state=3, **params)
    second = fit_model(name, split, random_state=3, **params)
    np.testing.assert_allclose(first.predict(split.x_test), second.predict(split.x_test))


@pytest.mark.skipif(not is_installed("tabpfn"), reason="tabpfn is not installed")
def test_tabpfn_is_v2_unless_a_version_is_chosen() -> None:
    """Building TabPFN downloads nothing; fitting it is covered by the heavy tests."""
    assert build_model("tabpfn", "regression").model_path.endswith("tabpfn-v2-regressor.ckpt")
    classifier = build_model("tabpfn", "classification", version="v2.5", n_estimators=2)
    assert "v2.5" in classifier.model_path
    assert classifier.n_estimators == 2
    with pytest.raises(ValueError, match="has no TabPFN 'v0.9'"):
        build_model("tabpfn", "regression", version="v0.9")


def test_svm_classifiers_predict_probabilities(splits: dict) -> None:
    model = fit_model("svm", splits["classification"])
    probabilities = model.predict_proba(splits["classification"].x_test)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_unknown_model_and_task_raise() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("not_a_model", "regression")
    with pytest.raises(ValueError, match="task must be"):
        build_model("decision_tree", "clustering")  # type: ignore[arg-type]


def test_random_forest_defaults_and_overrides() -> None:
    forest = build_model("random_forest", "regression", random_state=7)
    assert forest.get_params()["n_estimators"] == 10
    assert forest.get_params()["random_state"] == 7
    assert build_model("random_forest", "regression", n_estimators=3).n_estimators == 3


def test_tuned_presets() -> None:
    model = build_model("random_forest", "regression", preset="tuned", dataset="california_housing")
    assert (
        model.get_params()["n_estimators"]
        == TUNED_PRESETS[("random_forest", "california_housing")]["n_estimators"]
    )
    override = build_model(
        "random_forest", "regression", preset="tuned", dataset="california_housing", n_estimators=2
    )
    assert override.n_estimators == 2
    with pytest.raises(ValueError, match="No tuned preset"):
        build_model("decision_tree", "regression", preset="tuned", dataset="california_housing")
    with pytest.raises(ValueError, match="Unknown preset"):
        build_model("random_forest", "regression", preset="best", dataset="california_housing")
