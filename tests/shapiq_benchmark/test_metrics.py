"""Scientific interpretation checks for interaction-only errors."""

from __future__ import annotations

import pytest

from shapiq_benchmark.order_metrics import order_scores


def test_missing_pairs_are_not_hidden_by_accurate_singletons() -> None:
    """A zero-pair estimator has pair nMSE one, even with dominant exact singletons."""
    game = {"n_players": 3, "order": 2, "metadata": {"payoff_std": 10.0}}
    truth = {(0,): 100.0, (0, 1): 1.0}
    scores = order_scores(truth, {(0,): 100.0}, game)
    assert scores["1"]["nmse"] == 0
    assert scores["2"]["nmse"] == 1
    assert scores["2"]["energy_share"] == pytest.approx(1 / 10001)


@pytest.mark.parametrize("value", [0, 1e-12])
def test_weak_pair_truth_is_undefined_for_every_prediction(value: float) -> None:
    """An apparent perfect zero-interaction result cannot bypass the signal guard."""
    game = {"n_players": 3, "order": 2, "metadata": {"payoff_std": 1.0}}
    truth = {(0,): 1.0, (0, 1): value}
    for prediction in ({}, truth, {(0, 1): 1.0}):
        scores = order_scores(truth, prediction, game)
        assert scores["2"]["nmse"] is None
        assert scores["2"]["score_eligible"] is False


def test_order_signal_is_scale_invariant_and_legacy_reference_explicit() -> None:
    """Changing payoff units changes MSE, not normalized error or eligibility."""
    game = {"n_players": 3, "order": 2, "metadata": {"payoff_std": 2.0}}
    truth = {(0,): 3.0, (0, 1): 0.1}
    first = order_scores(truth, {}, game)
    scaled = order_scores(
        {key: value * 100 for key, value in truth.items()},
        {},
        {**game, "metadata": {"payoff_std": 200.0}},
    )
    for degree in first:
        assert scaled[degree]["signal_ratio"] == pytest.approx(first[degree]["signal_ratio"])
        assert scaled[degree]["nmse"] == first[degree]["nmse"]
        assert scaled[degree]["mse"] == pytest.approx(first[degree]["mse"] * 10000)
    legacy = order_scores(truth, {}, {**game, "metadata": {}})
    assert legacy["2"]["signal_reference"] == "full_target_rms"
