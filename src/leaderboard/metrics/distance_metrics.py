"""Distance-based metrics for evaluating the performance of models."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from .base import Metric
from .result import MetricResult

T = TypeVar("T")


class MSEMetric(Metric):
    """Mean Squared Error (MSE) metric for evaluating the performance of models."""

    def __init__(self) -> None:
        """Initialize the MSE metric with its name and whether higher values are better."""
        self.name = "mse"
        self.higher_is_better = False

    def compute(self, ground_truth: T | list[T], estimated: T | list[T]) -> MetricResult:
        """Compute the MSE value given ground truth and estimated values."""
        difference = ground_truth - estimated

        return MetricResult(
            metric_name=self.name,
            value=float(np.mean(difference ** 2)),
            higher_is_better=self.higher_is_better,
        )


class NormalizedMSEMetric(Metric):
    """Normalized Mean Squared Error (MSE) metric for evaluating the performance of models."""

    def __init__(self) -> None:
        """Initialize the Normalized MSE metric with its name and whether higher values are better."""
        self.name = "normalized_mse"
        self.higher_is_better = False

    def compute(self, ground_truth: T | list[T], estimated: T | list[T]) -> MetricResult:
        """Compute the Normalized MSE value given ground truth and estimated values."""
        difference = ground_truth - estimated
        mse = np.mean(difference**2)
        variance = np.var(ground_truth)

        value = float(mse) if variance == 0 else float(mse / variance)

        return MetricResult(
            metric_name=self.name,
            value=value,
            higher_is_better=self.higher_is_better,
        )
