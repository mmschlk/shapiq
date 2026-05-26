from dataclasses import dataclass
from leaderboard.runner.experiment_runner import align_interaction_values
import pytest

from leaderboard.runner.runner_exceptions import InteractionKeyMismatchError
import numpy as np


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

    gt_values_aligned, approx_values_aligned = align_interaction_values(
        ground_truth,
        approx_values,
    )

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

    gt_values_aligned, approx_values_aligned = align_interaction_values(
        ground_truth,
        approx_values,
    )

    assert np.array_equal(gt_values_aligned, np.array([]))
    assert np.array_equal(approx_values_aligned, np.array([]))