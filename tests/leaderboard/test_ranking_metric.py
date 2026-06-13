from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from leaderboard.metrics.ranking_metrics import (  # noqa: E402
    KendallTauMetric,
    PrecisionAtKMetric,
    SpearmanMetric,
)


@pytest.mark.parametrize(
    ("metric", "ground_truth", "estimated", "expected", "kwargs"),
    [
        (
            SpearmanMetric(),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            1.0,
            {},
        ),
        (
            SpearmanMetric(),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([4.0, 3.0, 2.0, 1.0]),
            -1.0,
            {},
        ),
        (
            KendallTauMetric(),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            1.0,
            {},
        ),
        (
            KendallTauMetric(),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([4.0, 3.0, 2.0, 1.0]),
            -1.0,
            {},
        ),
        (
            PrecisionAtKMetric(),
            np.array([10.0, -8.0, 1.0, 0.5]),
            np.array([9.0, -7.0, 0.2, 0.1]),
            1.0,
            {"k": 2},
        ),
        (
            PrecisionAtKMetric(),
            np.array([10.0, -8.0, 1.0, 0.5]),
            np.array([0.1, -7.0, 9.0, 0.2]),
            0.5,
            {"k": 2},
        ),
    ],
)
def test_ranking_metrics_happy_path(metric, ground_truth, estimated, expected, kwargs):
    result = metric.compute(ground_truth, estimated, **kwargs)

    assert isinstance(result.value, float)
    assert result.value == pytest.approx(expected)
    assert result.metric_name == metric.name
    assert result.higher_is_better is True


@pytest.mark.parametrize("metric", [SpearmanMetric(), KendallTauMetric()])
def test_correlation_metrics_return_zero_for_constant_arrays(metric):
    ground_truth = np.array([2.0, 2.0, 2.0])
    estimated = np.array([5.0, 5.0, 5.0])

    result = metric.compute(ground_truth, estimated)

    assert isinstance(result.value, float)
    assert result.value == pytest.approx(0.0)


@pytest.mark.parametrize("invalid_k", [0, -1, -10])
def test_precision_at_k_rejects_non_positive_k(invalid_k):
    metric = PrecisionAtKMetric()

    with pytest.raises(ValueError, match="k must be greater than 0"):
        metric.compute(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            k=invalid_k,
        )


def test_precision_at_k_rejects_shape_mismatch():
    metric = PrecisionAtKMetric()

    with pytest.raises(ValueError, match="ground_truth and estimated must have the same shape"):
        metric.compute(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0]),
            k=2,
        )


def test_precision_at_k_caps_k_at_array_size():
    result = PrecisionAtKMetric().compute(
        np.array([10.0, 1.0]),
        np.array([1.0, 10.0]),
        k=10,
    )

    assert isinstance(result.value, float)
    assert result.value == pytest.approx(1.0)


def test_precision_at_k_returns_zero_for_empty_arrays():
    result = PrecisionAtKMetric().compute(
        np.array([]),
        np.array([]),
        k=5,
    )

    assert isinstance(result.value, float)
    assert result.value == pytest.approx(0.0)
