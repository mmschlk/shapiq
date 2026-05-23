"""Registry of all available metrics for evaluating the performance of models."""

from __future__ import annotations

from .distance_metrics import MSEMetric, NormalizedMSEMetric, MAEMetric
from .ranking_metrics import SpearmanMetric

METRICS = {
    "mse": MSEMetric(),
    "normalized_mse": NormalizedMSEMetric(),
    "mae": MAEMetric(),
    "spearman": SpearmanMetric(),
}
