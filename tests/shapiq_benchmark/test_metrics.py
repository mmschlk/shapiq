"""Property tests of the benchmark metrics."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from shapiq import InteractionValues
from shapiq.utils import powerset
from shapiq_benchmark.computers import BruteForceComputer
from shapiq_benchmark.metrics import (
    compare,
    error_metrics,
    faithfulness,
    faithfulness_sample,
    ranking_metrics,
)
from shapiq_games import SOUM


def _values(
    values: np.ndarray, n: int = 5, order: int = 2, baseline: float = 0.0
) -> InteractionValues:
    interactions = list(powerset(range(n), min_size=1, max_size=order))
    return InteractionValues(
        values=np.asarray(values, dtype=float),
        index="k-SII",
        max_order=order,
        min_order=1,
        n_players=n,
        interaction_lookup={interaction: i for i, interaction in enumerate(interactions)},
        baseline_value=baseline,
    )


@pytest.fixture
def truth() -> InteractionValues:
    return _values(np.random.default_rng(0).normal(size=15))


def test_identical_estimate_is_perfect(truth: InteractionValues) -> None:
    metrics = compare(truth, copy.deepcopy(truth), k=5)
    for name in ("mse", "mae", "sse", "sae", "nmse"):
        assert metrics[name] == 0.0
    for name in ("kendall_tau", "spearman", "precision_at_k", "kendall_tau_at_k"):
        assert metrics[name] == pytest.approx(1.0)


def test_rankings_are_invariant_to_positive_scaling(truth: InteractionValues) -> None:
    scaled = _values(3.0 * truth.values)
    metrics = compare(truth, scaled, k=5)
    assert metrics["kendall_tau"] == pytest.approx(1.0)
    assert metrics["precision_at_k"] == pytest.approx(1.0)
    assert metrics["mse"] == pytest.approx(4.0 * np.mean(truth.values**2))
    assert metrics["nmse"] == pytest.approx(4.0)


def test_reversed_ranking(truth: InteractionValues) -> None:
    metrics = compare(truth, _values(-truth.values), k=5)
    assert metrics["kendall_tau"] == pytest.approx(-1.0)
    assert metrics["spearman"] == pytest.approx(-1.0)
    # top-k is by magnitude, so negating everything keeps the same top interactions
    assert metrics["precision_at_k"] == pytest.approx(1.0)


def test_kendall_tau_is_the_textbook_statistic() -> None:
    from scipy.stats import kendalltau

    rng = np.random.default_rng(1)
    truth = rng.normal(size=15)
    estimate = truth + rng.normal(scale=0.5, size=15)
    assert ranking_metrics(truth, estimate)["kendall_tau"] == pytest.approx(
        kendalltau(truth, estimate)[0]
    )


def test_precision_at_k_uses_the_largest_magnitudes() -> None:
    truth = np.array([5.0, -4.0, 0.1, 0.2, 0.3])
    estimate = np.array([5.0, 0.0, 0.1, -4.0, 0.3])
    metrics = ranking_metrics(truth, estimate, k=2)
    assert metrics["precision_at_k"] == pytest.approx(0.5)  # {0, 1} vs {0, 3}


def test_precision_at_k_counts_only_nonzero_estimates_and_shares_ties() -> None:
    truth = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    off_top = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])  # zeros are never selected
    assert ranking_metrics(truth, off_top, k=2)["precision_at_k"] == 0.0
    assert ranking_metrics(truth, np.zeros(6), k=2)["precision_at_k"] == 0.0
    # interactions 0 and 2 tie for the one place: the expected precision of a random pick
    tied = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert ranking_metrics(truth, tied, k=1)["precision_at_k"] == pytest.approx(0.5)


def test_at_k_metrics_are_nan_when_k_covers_every_interaction() -> None:
    truth = np.array([3.0, 2.0, 1.0])
    for k in (3, 10):  # any estimate would pick every interaction
        metrics = ranking_metrics(truth, np.array([0.0, 0.0, 1.0]), k=k)
        assert np.isnan(metrics["precision_at_k"])
        assert np.isnan(metrics["kendall_tau_at_k"])
    # interactions left out of the vectors (0 in both) still count
    metrics = ranking_metrics(truth, np.array([0.0, 0.0, 1.0]), k=3, n_interactions=6)
    assert metrics["precision_at_k"] == pytest.approx(1 / 3)


def test_compare_scores_failed_estimates(truth: InteractionValues) -> None:
    values = truth.values.copy()
    values[3] = np.nan
    assert all(np.isnan(value) for value in compare(truth, _values(values), k=5).values())
    values[3] = np.inf
    metrics = compare(truth, _values(values), k=5)
    assert metrics["mse"] == metrics["sse"] == np.inf


def test_ranking_metrics_ignore_float_noise_among_ties() -> None:
    """A sparse ground truth: 2 non-zero values, the rest zero up to float noise in the estimate."""
    truth = np.zeros(55)
    truth[[3, 20]] = [1.0, -0.5]
    estimate = truth + np.random.default_rng(0).normal(scale=1e-9, size=55)
    metrics = ranking_metrics(truth, estimate, k=10)
    assert metrics["precision_at_k"] == 1.0  # every zero is tied with the 10th largest truth
    assert metrics["kendall_tau"] == pytest.approx(1.0)
    assert metrics["kendall_tau_at_k"] == pytest.approx(1.0)
    worse = ranking_metrics(truth, truth[::-1].copy(), k=2)
    assert worse["precision_at_k"] == 0.0


def test_compare_rejects_orders_beyond_the_ground_truth(truth: InteractionValues) -> None:
    with pytest.raises(ValueError, match="max_order"):
        compare(truth, truth, order=3)


def test_error_metrics_and_degenerate_ground_truth() -> None:
    metrics = error_metrics(np.zeros(4), np.array([1.0, -1.0, 0.0, 0.0]))
    assert metrics["mse"] == 0.5
    assert metrics["mae"] == 0.5
    assert np.isnan(metrics["nmse"])
    assert np.isnan(ranking_metrics(np.zeros(4), np.arange(4.0))["kendall_tau"])


def test_order_zero_and_other_orders_are_ignored(truth: InteractionValues) -> None:
    estimate = copy.deepcopy(truth)
    estimate.baseline_value = 100.0
    assert compare(truth, estimate)["mse"] == 0.0
    single_order = compare(truth, _values(np.r_[truth.values[:5], np.zeros(10)]), order=1)
    assert single_order["mse"] == 0.0


def test_faithfulness_is_one_for_exact_moebius_values() -> None:
    game = SOUM(6, 12, max_interaction_size=2, min_interaction_size=1, random_state=0)
    moebius = BruteForceComputer(game).exact_values("Moebius", 2)
    assert faithfulness(game, moebius) == pytest.approx(1.0)
    shapley = BruteForceComputer(game).exact_values("SV", 1)
    assert faithfulness(game, shapley) < 1.0


def test_faithfulness_of_a_reused_sample_is_the_same() -> None:
    """The runner evaluates the game once per seed and scores every estimate on that sample."""
    game = SOUM(12, 20, max_interaction_size=2, random_state=0)
    shapley = BruteForceComputer(game).exact_values("SV", 1)
    sample = faithfulness_sample(game, n_samples=300, random_state=7)
    assert sample[0].shape == (300, 12)
    reused = faithfulness(game, shapley, sample=sample)
    assert reused == faithfulness(game, shapley, n_samples=300, random_state=7)


def test_interactions_zero_in_both_only_count_in_the_means(truth: InteractionValues) -> None:
    """The same nonzero values among 300 players: errors and rankings agree, means divide by all."""
    estimate = _values(truth.values + np.random.default_rng(1).normal(size=15))
    small = compare(truth, estimate, k=5)

    def padded(values: InteractionValues) -> InteractionValues:
        return InteractionValues(
            values=dict(values.dict_values),
            index="k-SII",
            max_order=2,
            min_order=1,
            n_players=300,
            baseline_value=0.0,
        )

    large = compare(padded(truth), padded(estimate), k=5)
    n_interactions = math.comb(300, 1) + math.comb(300, 2)
    assert large["mse"] == pytest.approx(small["sse"] / n_interactions)
    assert large["mae"] == pytest.approx(small["sae"] / n_interactions)
    for name in ("sse", "sae", "nmse", "kendall_tau", "spearman", "precision_at_k"):
        assert large[name] == pytest.approx(small[name])
    # a count beyond the float range (all orders of 1100 players) gives a mean, not an OverflowError
    assert error_metrics(np.zeros(1), np.ones(1), n_interactions=2**1100)["mse"] == 0.0
