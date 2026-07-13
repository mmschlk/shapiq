from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from leaderboard.runner.benchmark_runner import run_benchmark
from leaderboard.runner.runner_exceptions import NoSuccessfulRunsError
from shapiq.approximator import ProxySHAP
from shapiq_games.synthetic import DummyGame

if TYPE_CHECKING:
    from leaderboard.runner.custom_types import InteractionIndex
    from shapiq import Game


def fake_ground_truth_fn(
    game: Game, index: InteractionIndex, max_order: int, method: str = "ExactComputer"
) -> str:
    """Return a fixed fake ground truth object for benchmark orchestration tests."""
    return "fake_ground_truth"


def fake_experiment_fn(
    *,
    game: Any,
    game_name: str,
    game_params: dict[str, Any],
    ground_truth: Any,
    approximator_class: type,
    index: str,
    max_order: int,
    budget: int,
    approx_seeds: list[int],
) -> list[dict[str, Any]]:
    """Return fixed raw run records and assert that benchmark inputs are forwarded."""
    assert ground_truth == "fake_ground_truth"
    assert game_name == "DummyGame"
    assert game_params == {"n": 3}
    assert index == "SV"
    assert max_order == 1
    assert budget == 100
    assert approx_seeds == [0, 1]

    return [
        {
            "run_id": "run-1",
            "approx_seed": 0,
            "run_failed": False,
            "error_message": None,
            "metrics": {"mse": 0.1},
        },
        {
            "run_id": "run-2",
            "approx_seed": 1,
            "run_failed": False,
            "error_message": None,
            "metrics": {"mse": 0.2},
        },
    ]


def fake_aggregate_fn(raw_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a fixed aggregated result and assert that raw records are forwarded."""
    assert len(raw_results) == 2
    assert raw_results[0]["run_id"] == "run-1"
    assert raw_results[1]["run_id"] == "run-2"

    return {
        "game_name": "DummyGame",
        "budget": 100,
        "metrics": {
            "mse_mean": 0.15,
        },
    }


def failing_aggregate_fn(raw_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Raise a controlled error to test aggregation error propagation."""
    raise NoSuccessfulRunsError from None


def test_run_benchmark_successful_returns_both_raw_and_aggregated_results():
    """Test that run_benchmark orchestrates ground truth, experiments, and aggregation."""
    game = DummyGame(n=3)

    result = run_benchmark(
        game=game,
        game_name="DummyGame",
        game_params={"n": 3},
        max_order=1,
        approx_seeds=[0, 1],
        budget=100,
        index="SV",
        approximator_class=ProxySHAP,
        ground_truth_fn=fake_ground_truth_fn,
        experiment_fn=fake_experiment_fn,
        aggregate_fn=fake_aggregate_fn,
    )

    assert "raw_results" in result
    assert "aggregated_result" in result

    assert len(result["raw_results"]) == 2
    assert result["raw_results"][0]["approx_seed"] == 0
    assert result["raw_results"][1]["approx_seed"] == 1

    assert result["aggregated_result"] == {
        "game_name": "DummyGame",
        "budget": 100,
        "metrics": {
            "mse_mean": 0.15,
        },
    }


def test_run_benchmark_propagates_aggregation_error():
    """Test that run_benchmark propagates aggregation errors."""
    game = DummyGame(n=3)

    with pytest.raises(NoSuccessfulRunsError, match="No successful runs to aggregate."):
        run_benchmark(
            game=game,
            game_name="DummyGame",
            game_params={"n": 3},
            max_order=1,
            approx_seeds=[0, 1],
            budget=100,
            index="SV",
            approximator_class=ProxySHAP,
            ground_truth_fn=fake_ground_truth_fn,
            experiment_fn=fake_experiment_fn,
            aggregate_fn=failing_aggregate_fn,
        )
