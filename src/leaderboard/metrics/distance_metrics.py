"""Distance-based metrics for evaluating the performance of models."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from .base import Metric
from .result import MetricResult
from .utils import prepare_metric_inputs

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
            value=float(np.mean(difference**2)),
            higher_is_better=self.higher_is_better,
        )


class MAEMetric(Metric):
    """Mean Absolute Error (MAE) metric for evaluating model performance."""

    def __init__(self) -> None:
        """Initialize the MAE metric with its name and whether higher values are better."""
        self.name = "mae"
        self.higher_is_better = False

    def compute(self, ground_truth: T | list[T], estimated: T | list[T]) -> MetricResult:
        """Compute the MAE value given ground truth and estimated values."""
        difference = ground_truth - estimated

        return MetricResult(
            metric_name=self.name,
            value=float(np.mean(np.abs(difference))),
            higher_is_better=self.higher_is_better,
        )


class NormalizedMSEMetric(Metric):
    """Normalized Mean Squared Error (MSE) metric for evaluating the performance of models."""

    def __init__(self) -> None:
        """Initialize the Normalized MSE metric with its name and whether higher values are better."""
        self.name = "mse_normalized"
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

"""R² faithfulness score measuring reconstruction quality.

Defined as 1 - ||estimated - ground_truth||² / ||ground_truth - mean(ground_truth)||²,
following the faithfulness metric in ProxySPEX, Section 3.1, Equation (2).
"""
class R2Metric(Metric):
    """R² faithfulness score measuring reconstruction quality."""

    def __init__(self) -> None:
        """Initialize the R² metric with its name and sort direction."""
        self.name = "r2"
        self.higher_is_better = True

    def compute(self, ground_truth: T | list[T], estimated: T | list[T]) -> MetricResult:
        """Compute R² faithfulness given ground truth and estimated values."""
        ground_truth_array, estimated_array = prepare_metric_inputs(ground_truth, estimated)

        numerator = float(np.sum((estimated_array - ground_truth_array) ** 2))
        denominator = float(np.sum((ground_truth_array - np.mean(ground_truth_array)) ** 2))
        value = np.nan if np.isclose(denominator, 0.0) else 1.0 - numerator / denominator


        return MetricResult(
            metric_name=self.name,
            value=float(value),
            higher_is_better=self.higher_is_better,
        )
