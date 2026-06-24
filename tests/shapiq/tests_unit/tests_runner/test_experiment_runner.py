from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from config_manager import UnsupportedApproximatorError
from leaderboard.runner.approximator_registry import APPROXIMATOR_REGISTRY
from leaderboard.runner.experiment_runner import (
    align_interaction_values,
    run_single_experiment_seed,
)
from leaderboard.runner.runner_exceptions import InteractionKeyMismatchError
from shapiq import InteractionValues
from shapiq.approximator import ProxySHAP
from shapiq_games.synthetic import DummyGame


@dataclass
class MockInteractionValues:
    values: np.ndarray
    interaction_lookup: dict[tuple[int, ...], int]


def test_align_interaction_values_removes_empty_interaction_and_aligns_values():
    """Test if interaction values are aligned correctly and the empty interaction is removed."""
    ground_truth = MockInteractionValues(
        values=np.array([0.0, 10.0, 20.0, 30.0]),
        interaction_lookup={
            (): 0,
            (0,): 1,
            (1,): 2,
            (0, 1): 3,
        },
    )

    approx_values = MockInteractionValues(
        values=np.array([300.0, 100.0, 0.0, 200.0]),
        interaction_lookup={
            (0, 1): 0,
            (0,): 1,
            (): 2,
            (1,): 3,
        },
    )

    gt_values_aligned, approx_values_aligned = align_interaction_values(ground_truth, approx_values)

    assert np.array_equal(gt_values_aligned, np.array([10.0, 20.0, 30.0]))
    assert np.array_equal(approx_values_aligned, np.array([100.0, 200.0, 300.0]))


def test_align_interaction_values_raises_on_key_mismatch():
    """Test that mismatching interaction keys raise an error."""
    ground_truth = MockInteractionValues(
        values=np.array([0.0, 10.0, 20.0]),
        interaction_lookup={
            (): 0,
            (0,): 1,
            (1,): 2,
        },
    )

    approx_values = MockInteractionValues(
        values=np.array([0.0, 10.0, 30.0]),
        interaction_lookup={
            (): 0,
            (0,): 1,
            (0, 1): 2,
        },
    )

    with pytest.raises(InteractionKeyMismatchError):
        align_interaction_values(ground_truth, approx_values)


def test_align_interaction_values_removes_empty_interaction():
    """Test that empty interactions are removed from aligned arrays."""
    ground_truth = MockInteractionValues(
        values=np.array([0.0]),
        interaction_lookup={
            (): 0,
        },
    )

    approx_values = MockInteractionValues(
        values=np.array([0.0]),
        interaction_lookup={
            (): 0,
        },
    )

    gt_values_aligned, approx_values_aligned = align_interaction_values(ground_truth, approx_values)

    assert np.array_equal(gt_values_aligned, np.array([]))
    assert np.array_equal(approx_values_aligned, np.array([]))


def fake_approximate(**kwargs: Any) -> str:
    """Return a fake approximation result for the successful runner path."""
    return "approx_values"


def failing_approximate(**kwargs: Any) -> str:
    """Raise a controlled error to test failed run record creation."""
    available = APPROXIMATOR_REGISTRY.keys()
    name = "Fail"
    raise UnsupportedApproximatorError(name, list(available)) from None


def fake_align(
    ground_truth: InteractionValues, approx_values: Any, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed aligned ground truth and approximation arrays."""
    return np.array([1.0, 2.0]), np.array([1.1, 1.9])


def fake_metrics(
    ground_truth: list[float],
    estimated: list[float],
) -> dict[str, float | None]:
    """Return fixed metric values for testing record creation."""
    return {
        "mse": 0.01,
        "mae": 0.1,
        "mse_normalized": None,
        "spearman": 1.0,
        "kendall_tau": None,
        "precision_at_k": None,
    }


def fake_record_builder(**kwargs: Any) -> dict[str, Any]:
    """Return the record-builder arguments as the produced run record."""
    return kwargs


def test_run_single_experiment_seed_success():
    """Test that a successful single-seed experiment creates a successful run record."""
    record = run_single_experiment_seed(
        game=DummyGame(n=3),
        game_name="DummyGame",
        game_params={"n": 3},
        ground_truth=cast(InteractionValues, object()),
        approximator_class=ProxySHAP,
        index="SV",
        max_order=1,
        budget=100,
        approx_seed=42,
        approximate_fn=fake_approximate,
        align_fn=fake_align,
        metrics_fn=fake_metrics,
        record_builder_fn=fake_record_builder,
    )

    assert record["game_name"] == "DummyGame"
    assert record["game_params"] == {"n": 3}
    assert record["index"] == "SV"
    assert record["max_order"] == 1
    assert record["budget"] == 100
    assert record["approx_seed"] == 42
    assert record["run_failed"] is False
    assert record["error_message"] is None
    assert record["metrics"]["mse"] == 0.01
    assert record["metrics"]["spearman"] == 1.0


def test_run_single_experiment_seed_creates_failed_record_on_error():
    """Test that expected errors produce a failed run record."""
    record = run_single_experiment_seed(
        game=DummyGame(n=3),
        game_name="DummyGame",
        game_params={"n": 3},
        ground_truth=cast(InteractionValues, object()),
        approximator_class=ProxySHAP,
        index="SV",
        max_order=1,
        budget=100,
        approx_seed=42,
        approximate_fn=failing_approximate,
        align_fn=fake_align,
        metrics_fn=fake_metrics,
        record_builder_fn=fake_record_builder,
    )

    assert record["run_failed"] is True
    assert record["approx_seed"] == 42
    assert record["metrics"] is None
