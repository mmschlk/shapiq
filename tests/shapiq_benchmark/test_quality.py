"""Game quality is determined without estimator scores or held-out-test selection."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from shapiq.imputer.gaussian_imputer import GaussianImputer
from shapiq_benchmark import campaign_models as models
from shapiq_benchmark.families import make_family
from shapiq_benchmark.quality import (
    QUALITY_PROTOCOL,
    QualityExclusion,
    clustering_diagnostics,
    imputation_stability,
    model_validation_check,
    payoff_diagnostics,
)


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


def test_failed_validation_model_is_explicit_preflight_exclusion(
    regression: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A constructor cannot silently admit a known failed validation gate."""
    prepared = models.prepare_model("fixture", 3, 0, "linear", quality_protocol=QUALITY_PROTOCOL)
    prepared.metadata["model_validation_gate"]["passed"] = False
    monkeypatch.setattr("shapiq_benchmark.families.prepare_model", lambda *args, **kwargs: prepared)
    with pytest.raises(QualityExclusion) as caught:
        make_family(
            "local_baseline",
            dataset="fixture",
            n_players=3,
            model_profile="linear",
            quality_protocol=QUALITY_PROTOCOL,
        )
    assert caught.value.reason == "model_not_better_than_validation_dummy"
    assert "test" not in caught.value.details


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
