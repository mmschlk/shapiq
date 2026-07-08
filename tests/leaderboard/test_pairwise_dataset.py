from __future__ import annotations

import math

import pandas as pd
import pytest

from leaderboard.pdl.pairwise_dataset import (
    aggregate_metric_scores,
    build_pairwise_dataset,
    flatten_runs,
)


def test_mse_lower_score_wins() -> None:
    scores = _scores(metric="mse", score_a=0.1, score_b=0.2)

    pairwise = build_pairwise_dataset(scores)

    assert len(pairwise) == 1
    assert pairwise.iloc[0]["approximator_A"] == "A"
    assert pairwise.iloc[0]["approximator_B"] == "B"
    assert pairwise.iloc[0]["label"] == 1
    assert pairwise.iloc[0]["score_diff"] == pytest.approx(-0.1)


def test_spearman_higher_score_wins() -> None:
    scores = _scores(metric="spearman", score_a=0.7, score_b=0.9)

    pairwise = build_pairwise_dataset(scores)

    assert len(pairwise) == 1
    assert pairwise.iloc[0]["label"] == 0


def test_log_budget_is_created() -> None:
    pairwise = build_pairwise_dataset(_scores(metric="mse", score_a=0.1, score_b=0.2))

    assert pairwise.iloc[0]["log_budget"] == pytest.approx(math.log(100))


def test_pairs_are_only_created_within_same_game_params_context() -> None:
    scores = pd.DataFrame(
        [
            _score_row("A", 0.1, game_params_json='{"x": 1}'),
            _score_row("B", 0.2, game_params_json='{"x": 1}'),
            _score_row("C", 0.3, game_params_json='{"x": 2}'),
        ]
    )

    pairwise = build_pairwise_dataset(scores)

    assert len(pairwise) == 1
    assert set(pairwise.iloc[0][["approximator_A", "approximator_B"]]) == {"A", "B"}
    assert pairwise.iloc[0]["game_params_json"] == '{"x": 1}'


def test_ties_are_skipped() -> None:
    pairwise = build_pairwise_dataset(_scores(metric="mse", score_a=0.1, score_b=0.1))

    assert pairwise.empty


def test_missing_scores_are_skipped() -> None:
    scores = _scores(metric="mse", score_a=0.1, score_b=float("nan"))

    pairwise = build_pairwise_dataset(scores)

    assert pairwise.empty


def test_metrics_outside_requested_set_raise_clear_error() -> None:
    scores = _scores(metric="mae", score_a=0.1, score_b=0.2)

    with pytest.raises(ValueError, match="Unexpected metric"):
        build_pairwise_dataset(scores, metric_names=("mse", "spearman"))


def test_flatten_runs_normalizes_seed_and_flattens_metrics() -> None:
    raw_runs = [
        {
            "game_name": "Game",
            "game_params": {"b": 2, "a": 1},
            "n_players": 3,
            "approximator_name": "A",
            "approximator_params": {"z": 1},
            "index": "SV",
            "max_order": 1,
            "budget": 100,
            "seed": 7,
            "ground_truth_method": "ExactComputer",
            "run_failed": False,
            "metrics": {"mse": 0.1, "spearman": 0.8},
            "runtime_seconds": 1.2,
            "shapiq_version": "test",
            "hardware": {"cpu": "test"},
        },
        {"run_failed": True, "metrics": {"mse": 999.0}},
    ]

    flattened = flatten_runs(raw_runs)

    assert len(flattened) == 1
    assert flattened.iloc[0]["approx_seed"] == 7
    assert flattened.iloc[0]["mse"] == pytest.approx(0.1)
    assert flattened.iloc[0]["spearman"] == pytest.approx(0.8)
    assert flattened.iloc[0]["game_params_json"] == '{"a": 1, "b": 2}'
    assert flattened.iloc[0]["approximator_params_json"] == '{"z": 1}'


def test_aggregate_metric_scores_averages_across_seeds() -> None:
    flattened = pd.DataFrame(
        [
            _flat_row("A", approx_seed=0, mse=0.1, spearman=0.8),
            _flat_row("A", approx_seed=1, mse=0.3, spearman=1.0),
        ]
    )

    scores = aggregate_metric_scores(flattened)
    mse_row = scores[scores["metric"] == "mse"].iloc[0]
    spearman_row = scores[scores["metric"] == "spearman"].iloc[0]

    assert mse_row["score_mean"] == pytest.approx(0.2)
    assert mse_row["n_seeds"] == 2
    assert spearman_row["score_mean"] == pytest.approx(0.9)


def _scores(metric: str, score_a: float, score_b: float) -> pd.DataFrame:
    return pd.DataFrame([_score_row("A", score_a, metric), _score_row("B", score_b, metric)])


def _score_row(
    approximator_name: str,
    score: float,
    metric: str = "mse",
    game_params_json: str = '{"x": 1}',
) -> dict[str, object]:
    return {
        "game_name": "Game",
        "game_params_json": game_params_json,
        "n_players": 3,
        "index": "SV",
        "max_order": 1,
        "ground_truth_method": "ExactComputer",
        "budget": 100,
        "approximator_name": approximator_name,
        "metric": metric,
        "score_mean": score,
        "n_seeds": 1,
    }


def _flat_row(
    approximator_name: str,
    approx_seed: int,
    mse: float,
    spearman: float,
) -> dict[str, object]:
    return {
        "game_name": "Game",
        "game_params_json": '{"x": 1}',
        "n_players": 3,
        "index": "SV",
        "max_order": 1,
        "ground_truth_method": "ExactComputer",
        "budget": 100,
        "approximator_name": approximator_name,
        "approx_seed": approx_seed,
        "mse": mse,
        "spearman": spearman,
    }
