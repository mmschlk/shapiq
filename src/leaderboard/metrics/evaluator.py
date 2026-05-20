"""Evaluator for computing all registered metrics."""

from __future__ import annotations

from typing import TypeVar

from .registry import METRICS

T = TypeVar("T")


def compute_all_metrics(ground_truth: T | list[T], estimated: T | list[T]) -> dict[str, float]:
    """Compute all registered metrics given ground truth and estimated values."""
    results = {}

    for name, metric in METRICS.items():
        metric_result = metric.compute(
            ground_truth,
            estimated,
        )

        results[name] = metric_result.value

    return results
