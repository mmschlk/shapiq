"""Game quality is determined without estimator scores or held-out-test selection."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from shapiq.imputer.gaussian_imputer import GaussianImputer
from shapiq_benchmark import models
from shapiq_benchmark.families import _retraining_game, make_family
from shapiq_benchmark.games import validate_truth
from shapiq_benchmark.materialize import prepare_families
from shapiq_benchmark.quality import (
    QUALITY_PROTOCOL,
    QualityExclusion,
    clustering_diagnostics,
    imputation_stability,
    model_validation_check,
    payoff_diagnostics,
)
from shapiq_benchmark.structured import _construct
from shapiq_games.benchmark.data_valuation.base import DataValuation

if TYPE_CHECKING:
    from pathlib import Path


def test_clustering_resolution_uses_actual_cluster_count_bound() -> None:
    """A binary coalition may have two clusters despite requesting three."""
    metadata = {
        "class": "shapiq_games.benchmark.unsupervised_cluster.base.ClusterExplanation",
        "background_indices": list(range(128)),
        "parameters": {"cluster_params": {"n_clusters": 3}},
    }
    threshold = 126 / np.finfo(float).eps
    values = np.array([0, threshold, 30, 40])
    original = values.copy()
    quality = {"role": "core", "control_reasons": []}
    clustering_diagnostics(values, metadata, quality)
    assert quality["role"] == "control"
    assert quality["role_before_clustering_check"] == "core"
    assert quality["clustering_numerics"]["coalitions_at_float64_resolution"] == 1
    clustering_diagnostics(values, metadata, quality)
    assert quality["control_reasons"] == ["clustering_variance_at_float64_resolution"]
    np.testing.assert_array_equal(values, original)

    # k=3's smaller threshold must not falsely flag an actual k=2 coalition.
    quality = {"role": "core", "control_reasons": []}
    clustering_diagnostics(np.array([0, threshold * 0.75]), metadata, quality)
    assert quality["role"] == "core"
    assert quality["clustering_numerics"]["coalitions_at_float64_resolution"] == 0


def test_clustering_resolution_is_specific_to_ch_and_requires_row_metadata() -> None:
    """Large payoffs in other games are legitimate; missing evidence is not a pass."""
    quality = {"role": "core", "control_reasons": []}
    clustering_diagnostics(np.array([1e35]), {}, quality)
    assert quality == {"role": "core", "control_reasons": []}
    metadata = {
        "class": "shapiq_games.benchmark.unsupervised_cluster.base.ClusterExplanation",
        "parameters": {"score_method": "silhouette_score"},
    }
    clustering_diagnostics(np.array([1e35]), metadata, quality)
    assert quality == {"role": "core", "control_reasons": []}
    metadata["parameters"] = {}
    with pytest.raises(ValueError, match="recorded clustering rows"):
        clustering_diagnostics(np.array([1e35]), metadata, quality)


def test_empty_jump_and_inactive_players_are_different_controls() -> None:
    """The majority-only pathology has no null players but no nonempty variation."""
    majority = np.full(16, 0.752)
    majority[0] = 0
    result = payoff_diagnostics(majority, 4)
    assert result["inactive_players"] == []
    assert result["nonempty_constant"]
    assert result["empty_indicator_variance_fraction"] == pytest.approx(1)
    assert result["control_reasons"] == ["constant_nonempty_payoffs", "empty_coalition_jump"]
    masks = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool)
    sparse = payoff_diagnostics(masks[:, 2].astype(float) + 100, 4)
    assert sparse["inactive_players"] == [0, 1, 3]
    assert sparse["active_players"] == 1
    assert not sparse["nonempty_constant"]
    constant = payoff_diagnostics(np.ones(16), 4)
    assert constant["empty_indicator_variance_fraction"] is None
    assert constant["active_players"] == 0


def test_quality_selection_uses_validation_not_test() -> None:
    """Making test performance arbitrarily good cannot rescue a weak validation model."""
    metadata = {
        "quality": {
            "validation": {"model": {"mse": 2.0}, "dummy": {"mse": 1.0}},
            "test": {"model": {"mse": 0}, "dummy": {"mse": 1e99}},
        }
    }
    assert not model_validation_check(metadata)["passed"]
    metadata["quality"]["validation"]["model"]["mse"] = 0.9
    metadata["quality"]["test"]["model"]["mse"] = 1e100
    assert model_validation_check(metadata)["passed"]


def test_stratified_row_players_keep_holdout_and_legacy_identity() -> None:
    """A majority-only prefix becomes a reproducible class-complete set only in v2."""
    x = np.arange(60, dtype=float).reshape(-1, 1)
    y = np.r_[np.zeros(45), np.ones(15)].astype(int)
    prepared = models.PreparedModel(
        LogisticRegression(),
        x,
        y,
        x[:0],
        y[:0],
        np.array([[3.0], [55.0]]),
        np.array([0, 1]),
        {
            "task": "classification",
            "model_profile": "linear",
            "model_parameters": {},
            "train_indices": list(range(60)),
            "quality_protocol": QUALITY_PROTOCOL,
        },
    )
    game, metadata = _retraining_game(DataValuation, "data_valuation", prepared, 11, 0)
    repeated, other = _retraining_game(DataValuation, "data_valuation", prepared, 11, 0)
    assert set(game.y_train) == {0, 1}
    np.testing.assert_array_equal(game.y_train, y[metadata["train_indices"]])
    np.testing.assert_array_equal(game.x_train, repeated.x_train)
    np.testing.assert_array_equal(game.x_test, prepared.x_test)
    assert metadata == other
    assert set(game(np.eye(11, dtype=bool))) == {0.5}
    legacy = copy.deepcopy(prepared)
    legacy.metadata.pop("quality_protocol")
    old, old_metadata = _retraining_game(DataValuation, "data_valuation", legacy, 11, 0)
    assert set(old.y_train) == {0}
    assert old_metadata["train_indices"] == list(range(11))


@pytest.fixture
def regression(monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Large signed regression outputs expose an unscaled epsilon/C grid."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(160, 3))
    y = 2e6 + 5e5 * (np.sin(x[:, 0]) + x[:, 1])
    monkeypatch.setitem(
        models.DATASETS, "fixture", {"task": "regression", "source": "fixture", "n_features": 3}
    )
    monkeypatch.setattr(models, "load_raw_dataset", lambda _: (x, y, ["a", "b", "c"]))
    return x, y


def test_scaled_svr_keeps_original_units_and_exact_kernel_truth(regression: tuple) -> None:
    """Both the oracle and analytic solver restore target scale AND additive intercept."""
    prepared = models.prepare_model("fixture", 3, 0, "rbf_svm", quality_protocol=QUALITY_PROTOCOL)
    assert prepared.metadata["model_validation_gate"]["passed"]
    train_y = regression[1][prepared.metadata["train_indices"]]
    scaling = prepared.metadata["structure"]["target_scaling"]
    assert scaling["mean"] == pytest.approx(np.mean(train_y))
    assert scaling["scale"] == pytest.approx(np.std(train_y))
    oracle, truth, arrays, details = _construct(
        prepared, {"oracle": "product_kernel", "index": "SV", "order": 1}
    )
    full = oracle(np.ones((1, 3), dtype=bool))[0]
    assert full == pytest.approx(prepared.predict(prepared.x_test[:1])[0], rel=1e-12)
    assert abs(details["intercept"]) > 1e6
    assert np.max(abs(arrays["alpha"])) > 1e4
    assert validate_truth(oracle, truth, exhaustive=True) < 1e-7
    game, _ = make_family(
        "product_kernel",
        dataset="fixture",
        n_players=3,
        model_profile="rbf_svm",
        quality_protocol=QUALITY_PROTOCOL,
    )
    np.testing.assert_allclose(game(np.eye(3, dtype=bool)), oracle(np.eye(3, dtype=bool)))


def test_failed_validation_model_is_explicit_preflight_exclusion(
    regression: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A constructor cannot silently admit a known failed validation gate."""
    prepared = models.prepare_model("fixture", 3, 0, "rbf_svm", quality_protocol=QUALITY_PROTOCOL)
    prepared.metadata["model_validation_gate"]["passed"] = False
    monkeypatch.setattr("shapiq_benchmark.families.prepare_model", lambda *args, **kwargs: prepared)
    with pytest.raises(QualityExclusion) as caught:
        make_family(
            "local_baseline",
            dataset="fixture",
            n_players=3,
            model_profile="rbf_svm",
            quality_protocol=QUALITY_PROTOCOL,
        )
    assert caught.value.reason == "model_not_better_than_validation_dummy"
    assert "test" not in caught.value.details


def test_diagnostics_preserve_payoffs_and_exact_answers(tmp_path: Path) -> None:
    """A control label does not redefine the game or suppress its ground truth."""
    base = {"id": "control", "family": "unanimity", "n_players": 4}
    old, _ = prepare_families([base], [{"index": "SV", "order": 1}], tmp_path / "old")
    new, _ = prepare_families(
        [{**base, "quality_protocol": QUALITY_PROTOCOL}],
        [{"index": "SV", "order": 1}],
        tmp_path / "new",
    )
    assert old[0]["truth"] == new[0]["truth"]
    assert "game_quality" not in old[0]["metadata"]
    assert new[0]["metadata"]["game_quality"]["role"] == "control"


def test_imputation_probe_leaves_real_rng_and_samples_unchanged() -> None:
    """Probe a real shipped imputer using shared frozen model/data and isolated RNGs."""
    rng = np.random.default_rng(44)
    data = rng.normal(size=(100, 3))
    game = GaussianImputer(
        model=lambda x: x.sum(axis=1),
        data=data,
        x=np.array([2.0, 2.0, 2.0]),
        sample_size=64,
        random_state=4,
    )
    masks = np.eye(3, dtype=bool)
    before = game(masks)
    state = copy.deepcopy(game._rng.bit_generator.state)
    result = imputation_stability(game, seed=7)
    np.testing.assert_array_equal(before, game(masks))
    assert game.random_state == 4 and game.sample_size == 64
    assert game._rng.bit_generator.state == state
    assert result["status"] == "stable"
    assert [level["requested_sample_size"] for level in result["levels"]] == [64, 256]


def test_imputation_probe_flags_variation_created_by_sampling() -> None:
    """A centered linear Gaussian game at zero has zero population payoff everywhere."""
    data = np.random.default_rng(44).normal(size=(100, 3))
    data -= data.mean(axis=0)
    game = GaussianImputer(
        model=lambda x: x.sum(axis=1),
        data=data,
        x=np.zeros(3),
        sample_size=1,
        random_state=4,
    )
    result = imputation_stability(game, seed=7)
    assert result["status"] == "noise_dominated"
    assert result["levels"][0]["noise_ratio"] > result["maximum_noise_ratio"]


def test_unqualified_stochastic_game_is_a_control(tmp_path: Path) -> None:
    """A noisy frozen table alone is not evidence for a meaningful prediction game."""
    games, coverage = prepare_families(
        [
            {
                "id": "random-control",
                "family": "random",
                "n_players": 4,
                "quality_protocol": QUALITY_PROTOCOL,
            }
        ],
        [{"index": "SV", "order": 1}],
        tmp_path,
        instance_seed=0,
    )
    assert games, coverage
    metadata = games[0]["metadata"]
    assert metadata["imputation_stability"]["status"] == "unqualified"
    assert metadata["game_quality"]["role"] == "control"
    assert "unqualified_stochastic_payoffs" in metadata["game_quality"]["control_reasons"]
    assert "synthetic_control" in metadata["game_quality"]["control_reasons"]
