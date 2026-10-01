"""Frozen models keep held-out data independent and share identical artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from shapiq_benchmark import models

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def data(monkeypatch: pytest.MonkeyPatch) -> tuple:
    """A tiny deterministic loader; tests never download or train on the full catalog."""
    rng = np.random.default_rng(12)
    x = rng.normal(size=(120, 12))
    y = (x[:, 0] * x[:, 1] + x[:, 2] > 0).astype(int)
    names = [f"column-{i}" for i in range(12)]
    monkeypatch.setitem(models.DATASETS, "fixture", {"task": "classification", "source": "fixture"})
    monkeypatch.setattr(models, "load_raw_dataset", lambda _: (x, y, names))
    monkeypatch.setitem(
        models.MODEL_PROFILES,
        "random_forest",
        {
            "label": "Test forest",
            "parameters": {"n_estimators": 3, "min_samples_leaf": 2, "n_jobs": 1},
        },
    )
    return x, y, names


def test_reproducible_disjoint_splits_and_cached_model(data: tuple, tmp_path: Path) -> None:
    """Both constructions get byte-authenticated identical models and original row IDs."""
    first = models.prepare_model("fixture", 11, 2, "random_forest", cache_dir=tmp_path)
    second = models.prepare_model("fixture", 11, 2, "random_forest", cache_dir=tmp_path)
    uncached = models.prepare_model("fixture", 11, 2, "random_forest")
    np.testing.assert_array_equal(first.predict(first.x_test), second.predict(second.x_test))
    np.testing.assert_array_equal(first.predict(first.x_test), uncached.predict(uncached.x_test))
    metadata = first.metadata
    train, validation, test = (
        set(metadata[key]) for key in ("train_indices", "validation_indices", "test_indices")
    )
    assert not train & validation and not train & test and not validation & test
    assert train | validation | test == set(range(120))
    assert metadata["background_indices"] == metadata["train_indices"][:16]
    assert metadata["training_rows"] > len(validation)
    assert metadata["quality"]["test"]["dummy"]["log_loss"] > 0
    assert metadata["structure"]["tree_count"] == 3
    assert len(metadata["structure"]["features_used"]) <= 11
    assert first.model.n_jobs == 1
    json.dumps(metadata, allow_nan=False)
    artifact = next(tmp_path.glob("*.joblib"))
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == metadata["model_artifact_sha256"]
    artifact.write_bytes(artifact.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        models.prepare_model("fixture", 11, 2, "random_forest", cache_dir=tmp_path)


def test_medians_ignore_validation_and_test(data: tuple) -> None:
    """Even extreme held-out values cannot alter fitting-row missing-value repair."""
    x, y, _ = data
    fit, test = train_test_split(np.arange(len(x)), test_size=0.2, random_state=0, stratify=y)
    fit, validation = train_test_split(fit, test_size=0.2, random_state=0, stratify=y[fit])
    x[test, 0] = 1e12
    x[validation, 0] = 1e12
    x[fit[0], 0] = np.nan
    x[test[0], 0] = np.nan
    prepared = models.prepare_model("fixture", 12, 0, "random_forest")
    median = np.nanmedian(x[fit, 0])
    assert prepared.metadata["imputation_medians"][0] == median
    assert prepared.x_train[0, 0] == prepared.x_test[0, 0] == median
    assert np.isnan(x[fit[0], 0])


def test_regression_preserves_signed_targets(data: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression targets are not converted into class IDs."""
    x, _, names = data
    y = 4 * x[:, 0] - 2
    monkeypatch.setitem(models.DATASETS, "fixture", {"task": "regression", "source": "fixture"})
    monkeypatch.setattr(models, "load_raw_dataset", lambda _: (x, y, names))
    prepared = models.prepare_model("fixture", 12, 0, "random_forest")
    np.testing.assert_array_equal(prepared.y_train, y[prepared.metadata["train_indices"]])
    assert set(prepared.metadata["quality"]["test"]["model"]) == {"mse", "r2"}
    assert prepared.metadata["output_class"] is None
    assert np.isfinite(prepared.predict(prepared.x_test)).all()


def test_caps_and_seed_are_part_of_identity(
    data: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Changing the training recipe cannot silently reuse a previous fitted model."""
    before = models.prepare_model("fixture", 12, 0, "random_forest", cache_dir=tmp_path)
    monkeypatch.setattr(models, "TRAINING_PROFILE", {**models.TRAINING_PROFILE, "fit_rows_max": 30})
    after = models.prepare_model("fixture", 12, 0, "random_forest", cache_dir=tmp_path)
    other = models.prepare_model("fixture", 12, 1, "random_forest", cache_dir=tmp_path)
    assert after.metadata["training_rows"] == 30
    assert len({item.metadata["model_key"] for item in (before, after, other)}) == 3


@pytest.mark.parametrize(
    "task,classes", [("classification", 2), ("classification", 3), ("regression", 0)]
)
def test_xgboost_correct_objective_and_validation(
    data: tuple, monkeypatch: pytest.MonkeyPatch, task: str, classes: int
) -> None:
    """Real tiny fits cover multiclass labels and validation-based early stopping."""
    pytest.importorskip("xgboost")
    x, _, names = data
    y = np.arange(len(x)) % classes * 3 + 2 if classes else x[:, 0] * 2
    monkeypatch.setitem(models.DATASETS, "fixture", {"task": task, "source": "fixture"})
    monkeypatch.setattr(models, "load_raw_dataset", lambda _: (x, y, names))
    monkeypatch.setitem(
        models.MODEL_PROFILES,
        "xgboost",
        {
            "label": "Test XGBoost",
            "parameters": {
                "n_estimators": 4,
                "max_depth": 2,
                "learning_rate": 0.1,
                "early_stopping_rounds": 2,
                "n_jobs": 1,
            },
        },
    )
    prepared = models.prepare_model("fixture", 11, 0, "xgboost")
    assert 0 <= prepared.metadata["best_iteration"] < 4
    assert prepared.model.n_jobs == 1
    assert np.isfinite(prepared.predict(prepared.x_test)).all()
    if classes:
        assert prepared.metadata["output_class"] == 5
        assert set(prepared.y_train) == set(range(classes))
        np.testing.assert_array_equal(
            prepared.predict(prepared.x_test), prepared.model.predict_proba(prepared.x_test)[:, 1]
        )
    json.dumps(prepared.metadata, allow_nan=False)


def test_valid_artifact_from_wrong_seed_is_rejected(data: tuple, tmp_path: Path) -> None:
    """A valid checksum cannot make another seed's model valid for this cache key."""
    first = models.prepare_model("fixture", 12, 0, "random_forest", cache_dir=tmp_path)
    second = models.prepare_model("fixture", 12, 1, "random_forest", cache_dir=tmp_path)
    for suffix in (".joblib", ".sha256"):
        destination = tmp_path / (first.metadata["model_key"] + suffix)
        source = tmp_path / (second.metadata["model_key"] + suffix)
        destination.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="identity disagrees"):
        models.prepare_model("fixture", 12, 0, "random_forest", cache_dir=tmp_path)


def test_feature_semantics_are_part_of_cache_identity(
    data: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Equal numeric inputs cannot reuse obsolete column names or loader provenance."""
    x, y, names = data
    first = models.prepare_model("fixture", 11, 0, "random_forest", cache_dir=tmp_path)
    renamed = [f"renamed-{name}" for name in names]
    monkeypatch.setattr(models, "load_raw_dataset", lambda _: (x, y, renamed))
    second = models.prepare_model("fixture", 11, 0, "random_forest", cache_dir=tmp_path)
    repeated = models.prepare_model("fixture", 11, 0, "random_forest", cache_dir=tmp_path)
    assert first.metadata["model_key"] != second.metadata["model_key"]
    assert second.metadata == repeated.metadata
    assert second.metadata["dataset_feature_names"] == renamed
    assert second.metadata["feature_names"] == [
        renamed[i] for i in second.metadata["feature_indices"]
    ]
    assert second.metadata["data_arrays"] == {
        "x": {"shape": list(x.shape), "dtype": str(x.dtype)},
        "y": {"shape": list(y.shape), "dtype": str(y.dtype)},
    }
    monkeypatch.setitem(models.DATASETS["fixture"], "source", "replacement.loader")
    third = models.prepare_model("fixture", 11, 0, "random_forest", cache_dir=tmp_path)
    assert third.metadata["model_key"] != second.metadata["model_key"]
    assert third.metadata["dataset_source"] == "replacement.loader"
