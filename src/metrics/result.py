from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricResult:
    metric_name: str
    value: float
    higher_is_better: bool
