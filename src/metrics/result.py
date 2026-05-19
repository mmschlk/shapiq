"""Result class for storing the outcome of metric computations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricResult:
    """Class for storing the result of a metric computation."""

    metric_name: str
    value: float
    higher_is_better: bool
