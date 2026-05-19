"""Ranking-based metrics for evaluating the performance of models."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from scipy.stats import spearmanr

from .base import Metric
from .result import MetricResult

T = TypeVar("T")


class SpearmanMetric(Metric):
    """Spearman's rank correlation coefficient metric for evaluating the performance of models."""

    def __init__(self) -> None:
        """Initialize the Spearman metric with its name and whether higher values are better."""
        self.name = "spearman"
        self.higher_is_better = True

    def compute(self, ground_truth: T | list[T], estimated: T | list[T]) -> MetricResult:
        """Compute the Spearman correlation value given ground truth and estimated values."""
        correlation, _ = spearmanr(ground_truth, estimated)

        if np.isnan(correlation):
            correlation = 0.0

        return MetricResult(
            metric_name=self.name,
            value=float(correlation),
            higher_is_better=self.higher_is_better,
        )
