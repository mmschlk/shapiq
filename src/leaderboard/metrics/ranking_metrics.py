"""Ranking-based metrics for evaluating the performance of models."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from scipy.stats import kendalltau, spearmanr

from shapiq.interaction_values import InteractionValues

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


class KendallTauMetric(Metric):
    """Kendall tau rank correlation metric."""

    def __init__(self) -> None:
        """Initialize the Kendall tau metric with its name and sort direction."""
        self.name = "kendall_tau"
        self.higher_is_better = True

    def compute(self, ground_truth: T | list[T], estimated: T | list[T]) -> MetricResult:
        """Compute Kendall tau given prepared ground truth and estimated values."""
        correlation, _ = kendalltau(ground_truth, estimated)

        if np.isnan(correlation):
            correlation = 0.0

        return MetricResult(
            metric_name=self.name,
            value=float(correlation),
            higher_is_better=self.higher_is_better,
        )


class PrecisionAtKMetric(Metric):
    """Top-k overlap metric based on absolute interaction values."""

    def __init__(self) -> None:
        """Initialize the Precision@k metric with its name and sort direction."""
        self.name = "precision_at_k"
        self.higher_is_better = True

    def compute(
        self,
        ground_truth: T | list[T],
        estimated: T | list[T],
        k: int = 10,
    ) -> MetricResult:
        """Compute Precision@k for InteractionValues or prepared arrays."""
        if k <= 0:
            msg = "k must be greater than 0"
            raise ValueError(msg)

        if isinstance(ground_truth, InteractionValues) and isinstance(estimated, InteractionValues):
            gt_top_k = _top_k_interaction_keys(ground_truth, k)
            estimated_top_k = _top_k_interaction_keys(estimated, k)
            denominator = min(k, max(len(gt_top_k), len(estimated_top_k)))
        else:
            gt_array = np.asarray(ground_truth, dtype=float).ravel()
            estimated_array = np.asarray(estimated, dtype=float).ravel()
            if gt_array.shape != estimated_array.shape:
                msg = "ground_truth and estimated must have the same shape"
                raise ValueError(msg)
            denominator = min(k, gt_array.size)
            gt_top_k = _top_k_array_indices(gt_array, denominator)
            estimated_top_k = _top_k_array_indices(estimated_array, denominator)

        value = 0.0 if denominator == 0 else len(gt_top_k & estimated_top_k) / denominator

        return MetricResult(
            metric_name=self.name,
            value=float(value),
            higher_is_better=self.higher_is_better,
        )


def _top_k_array_indices(values: np.ndarray, k: int) -> set[int]:
    if k <= 0:
        return set()
    return set(np.argsort(np.abs(values))[-k:])


def _top_k_interaction_keys(values: InteractionValues, k: int) -> set[tuple[int, ...]]:
    _, interactions = values.get_top_k(len(values.interaction_lookup), as_interaction_values=False)
    return set([interaction for interaction, _ in interactions if interaction != ()][:k])