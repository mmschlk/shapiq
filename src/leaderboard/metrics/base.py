"""Metric abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .result import MetricResult

T = TypeVar("T")


class Metric(ABC):
    """Abstract base class for metrics."""

    name = "base"
    higher_is_better = False

    @abstractmethod
    def compute(self, ground_truth: T | list[T], estimated: T | list[T]) -> MetricResult:
        """Compute the metric value given ground truth and estimated values."""
