"""Registry of all available metrics for evaluating the performance of models."""

from __future__ import annotations

from .distance_metrics import MSEMetric, NormalizedMSEMetric
from .ranking_metrics import SpearmanMetric

METRICS = {
    "mse": MSEMetric(),
    "normalized_mse": NormalizedMSEMetric(),
    "spearman": SpearmanMetric(),
}
