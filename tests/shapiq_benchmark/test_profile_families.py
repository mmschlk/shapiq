"""Qualified model profiles preserve each shipped construction's payoff semantics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.stats import entropy, mode
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

from shapiq_benchmark import models
from shapiq_benchmark.families import _CoalitionRefit, make_family, profile_compatibility

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def profiled_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three original labels exercise remapping; tiny profiles keep qualification cheap."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(120, 12))
    y = np.digitize(x[:, 0] * x[:, 1] + x[:, 2], [-0.6, 0.6]) * 7 + 5
    monkeypatch.setitem(
        models.DATASETS,
        "fixture",
        {
            "task": "classification",
            "n_features": 12,
            "n_classes": 3,
            "source": "fixture",
        },
    )
    monkeypatch.setattr(models, "load_raw_dataset", lambda _: (x, y, [f"f{i}" for i in range(12)]))
    monkeypatch.setitem(
        models.MODEL_PROFILES,
        "mlp",
        {**models.MODEL_PROFILES["mlp"], "parameters": {"hidden_layer_sizes": (4,), "max_iter": 3}},
    )
    for profile, parameters in {
        "random_forest": {"n_estimators": 4, "min_samples_leaf": 2, "n_jobs": 1},
        "xgboost": {
            "n_estimators": 8,
            "max_depth": 3,
            "early_stopping_rounds": 2,
            "n_jobs": 1,
            "tree_method": "hist",
        },
        "lightgbm": {
            "n_estimators": 8,
            "num_leaves": 7,
            "min_child_samples": 2,
            "learning_rate": 0.1,
            "n_jobs": 1,
            "verbosity": -1,
        },
    }.items():
        monkeypatch.setitem(
            models.MODEL_PROFILES,
            profile,
            {**models.MODEL_PROFILES[profile], "parameters": parameters},
        )


@pytest.mark.parametrize("profile", ["random_forest", "xgboost", "linear"])
@pytest.mark.parametrize("name", ["feature_selection", "data_valuation", "dataset_valuation"])
def test_profiled_retraining_keeps_empty_and_disjoint_pools(
    profiled_data: None, tmp_path: Path, profile: str, name: str
) -> None:
    """Empty utility stays zero; row singletons are constant-label fits with no holdout leakage."""
    with threadpool_limits(limits=1):
        game, metadata = make_family(
            name, dataset="fixture", n_players=11, model_profile=profile, model_cache=str(tmp_path)
        )
        prepared = models.prepare_model(
            "fixture", 11 if name == "feature_selection" else 12, 0, profile, cache_dir=tmp_path
        )
        coalitions = np.vstack((np.zeros(11), np.eye(11), np.ones(11))).astype(bool)
        values = game(coalitions)
        assert values[0] == 0
        assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
        np.testing.assert_array_equal(values, game(coalitions[::-1])[::-1])
        assert set(metadata["train_indices"]).isdisjoint(metadata["test_indices"])
        assert set(metadata["train_indices"]).isdisjoint(metadata["validation_indices"])
        assert metadata["output_scale"] == "accuracy"
        if name == "data_valuation":
            np.testing.assert_array_equal(game.x_train, prepared.x_train[:11])
            np.testing.assert_array_equal(game.x_test, prepared.x_test)
            for i in range(11):
                assert values[i + 1] == np.mean(prepared.y_test == prepared.y_train[i])
        if name == "dataset_valuation":
            assert sorted(i for group in metadata["group_indices"] for i in group) == sorted(
                metadata["train_indices"]
            )
        json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("profile", ["random_forest", "xgboost", "linear", "lightgbm", "mlp"])
def test_profiled_subset_class_mapping(profiled_data: None, profile: str) -> None:
    """Internal class 1 must map to original encoded label 2 when class 1 is absent."""
    prepared = models.prepare_model("fixture", 11, 0, profile)
    refit = _CoalitionRefit(prepared)
    selected = prepared.y_train != 1
    with threadpool_limits(limits=1):
        refit.fit(prepared.x_train[selected], prepared.y_train[selected])
        predictions = refit.predict(prepared.x_test)
    np.testing.assert_array_equal(
        predictions, np.array([0, 2])[refit.model.predict(prepared.x_test).astype(int)]
    )
    assert set(predictions).issubset({0, 2})
    assert refit.template.get_params().get("early_stopping_rounds") is None


@pytest.mark.parametrize(
    "name",
    [
        "local_baseline",
        "local_marginal",
        "local_gaussian",
        "local_copula",
        "global_fidelity",
        "uncertainty",
    ],
)
def test_profiled_predictions_and_uncertainty(
    profiled_data: None, name: str, tmp_path: Path
) -> None:
    """A full coalition recovers its declared probability, fidelity or entropy output."""
    with threadpool_limits(limits=1):
        game, metadata = make_family(
            name,
            dataset="fixture",
            n_players=11,
            model_profile="random_forest",
            model_cache=str(tmp_path),
        )
        prepared = models.prepare_model(
            "fixture",
            11,
            0,
            "random_forest",
            cache_dir=tmp_path,
            feature_rule="continuous" if name in ("local_gaussian", "local_copula") else "all",
        )
        values = game(np.vstack((np.zeros(11), np.ones(11))).astype(bool))
    full = values[1] + (game.normalization_value if game.normalize else 0)
    expected = prepared.predict(prepared.x_test[:1])[0]
    if name == "global_fidelity":
        expected = 0
    elif name == "uncertainty":
        expected = entropy(prepared.model.predict_proba(prepared.x_test[:1])[0], base=2)
    assert full == pytest.approx(expected)
    assert np.isfinite(values).all()
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("profile", ["random_forest", "xgboost", "lightgbm"])
@pytest.mark.parametrize("name", ["pathdependent_tree", "interventional_tree"])
def test_profiled_tree_output_scale(
    profiled_data: None, profile: str, name: str, tmp_path: Path
) -> None:
    """Tree classification uses forest probabilities or selected-round boosted margins."""
    with threadpool_limits(limits=1):
        game, metadata = make_family(
            name, dataset="fixture", n_players=11, model_profile=profile, model_cache=str(tmp_path)
        )
        prepared = models.prepare_model("fixture", 11, 0, profile, cache_dir=tmp_path)
        point = prepared.x_test[:1].astype(np.float32).astype(float)
        if profile == "random_forest":
            expected = prepared.model.predict_proba(point)[0, 1]
        elif profile == "xgboost":
            expected = prepared.model.predict(point, output_margin=True)[0, 1]
        else:
            expected = prepared.model.predict(point, raw_score=True)[0, 1]
        actual = game(np.ones((1, 11), dtype=bool))[0]
        actual += game.normalization_value if game.normalize else 0
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
    assert metadata["output_scale"] == (
        "class probability" if profile == "random_forest" else "class margin"
    )
    json.dumps(metadata, allow_nan=False)


def test_profiled_forest_ensemble_votes(profiled_data: None) -> None:
    """The explicit ensemble has d trees, each voting on the same untouched holdout."""
    game, metadata = make_family(
        "forest_ensemble", dataset="fixture", n_players=11, model_profile="random_forest"
    )
    expected = accuracy_score(game._y_test, mode(game.predictions, axis=0)[0].ravel())
    assert game(np.ones((1, 11), dtype=bool))[0] == expected
    assert len(metadata["ensemble_members"]) == game.n_players == 11
    assert metadata["output_scale"] == "accuracy"


def test_profile_mapping_excludes_incompatible_models_before_fitting() -> None:
    """An arbitrary model cannot silently stand in for a specialized game."""
    assert profile_compatibility("uncertainty", "breast_cancer", "xgboost")
    assert profile_compatibility("product_kernel", "wine", "rbf_svm")
    assert profile_compatibility("pathdependent_tree", "breast_cancer", "mlp")
    assert profile_compatibility("local_marginal", "breast_cancer", "linear") is None
    with pytest.raises(ValueError, match="does not support"):
        make_family("uncertainty", dataset="breast_cancer", n_players=11, model_profile="xgboost")


def test_profiled_heterogeneous_ensemble_member_identity(
    profiled_data: None, tmp_path: Path
) -> None:
    """Members cycle named families, sharing rows/features rather than unrelated seeded splits."""
    with threadpool_limits(limits=1):
        game, metadata = make_family(
            "ensemble",
            dataset="fixture",
            n_players=11,
            model_profile="heterogeneous_ensemble",
            model_cache=str(tmp_path),
        )
    families = ["random_forest", "xgboost", "rbf_svm", "linear"]
    assert [member["profile"] for member in metadata["ensemble_members"]] == [
        families[i % len(families)] for i in range(11)
    ]
    assert game.n_players == 11
    assert game(np.ones((1, 11), dtype=bool))[0] == accuracy_score(
        game._y_test, mode(game.predictions, axis=0)[0].ravel()
    )
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("classification", [False, True])
def test_profiled_product_kernel_matches_fitted_score(
    profiled_data: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, classification: bool
) -> None:
    """Restricting the scaled RBF kernel recovers its fitted full-feature decision function."""
    x, y, names = models.load_raw_dataset("fixture")
    task = "classification" if classification else "regression"
    target = (y > 5).astype(int) if classification else 2 * x[:, 0] + x[:, 1] * x[:, 2]
    monkeypatch.setitem(
        models.DATASETS,
        "fixture",
        {"task": task, "n_features": 12, "n_classes": 2, "source": "fixture"},
    )
    monkeypatch.setattr(models, "load_raw_dataset", lambda _: (x, target, names))
    with threadpool_limits(limits=1):
        game, metadata = make_family(
            "product_kernel",
            dataset="fixture",
            n_players=11,
            model_profile="rbf_svm",
            model_cache=str(tmp_path),
        )
        prepared = models.prepare_model("fixture", 11, 0, "rbf_svm", cache_dir=tmp_path)
    expected = (
        prepared.model.decision_function(prepared.x_test[:1])[0]
        if classification
        else prepared.model.predict(prepared.x_test[:1])[0]
    )
    assert game(np.ones((1, 11), dtype=bool))[0] == pytest.approx(expected)
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("profile", ["random_forest", "xgboost", "linear", "lightgbm", "mlp"])
def test_profiled_regression_retraining_uses_signed_targets(
    profiled_data: None, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    """Regression coalitions retain real targets, including negative values, and negative MSE."""
    x, _, names = models.load_raw_dataset("fixture")
    target = 3 * x[:, 0] - 10
    monkeypatch.setitem(
        models.DATASETS, "fixture", {"task": "regression", "n_features": 12, "source": "fixture"}
    )
    monkeypatch.setattr(models, "load_raw_dataset", lambda _: (x, target, names))
    monkeypatch.setitem(
        models.MODEL_PROFILES,
        "mlp",
        {
            **models.MODEL_PROFILES["mlp"],
            "parameters": {
                **models.MODEL_PROFILES["mlp"]["parameters"],
                "hidden_layer_sizes": (4,),
                "max_iter": 3,
            },
        },
    )
    with threadpool_limits(limits=1):
        game, metadata = make_family(
            "data_valuation", dataset="fixture", n_players=11, model_profile=profile
        )
        values = game(np.vstack((np.zeros(11), np.eye(11), np.ones(11))).astype(bool))
    assert values[0] == 0
    assert np.isfinite(values).all() and (values[1:] <= 0).all()
    assert (game.y_train < 0).all()
    assert metadata["output_scale"] == "negative MSE"
    json.dumps(metadata, allow_nan=False)
