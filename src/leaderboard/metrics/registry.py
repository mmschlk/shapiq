"""Registry of all available metrics for evaluating the performance of models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .distance_metrics import MAEMetric, MSEMetric, NormalizedMSEMetric
from .ranking_metrics import KendallTauMetric, PrecisionAtKMetric, SpearmanMetric

if TYPE_CHECKING:
    from .base import Metric


@dataclass(frozen=True)
class MetricSpec:
    """Metadata and implementation for one canonical metric."""

    name: str
    function: Metric
    higher_is_better: bool
    category: str
    description: str


METRIC_KEYS = (
    "mse",
    "mae",
    "mse_normalized",
    "spearman",
    "kendall_tau",
    "precision_at_k",
)

METRIC_ALIASES = {"normalized_mse": "mse_normalized"}

METRIC_SPECS = {
    "mse": MetricSpec(
        name="mse",
        function=MSEMetric(),
        higher_is_better=False,
        category="error",
        description="Mean squared error.",
    ),
    "mae": MetricSpec(
        name="mae",
        function=MAEMetric(),
        higher_is_better=False,
        category="error",
        description="Mean absolute error.",
    ),
    "mse_normalized": MetricSpec(
        name="mse_normalized",
        function=NormalizedMSEMetric(),
        higher_is_better=False,
        category="error",
        description="Mean squared error normalized by reference variance.",
    ),
    "spearman": MetricSpec(
        name="spearman",
        function=SpearmanMetric(),
        higher_is_better=True,
        category="rank_correlation",
        description="Spearman rank correlation.",
    ),
    "kendall_tau": MetricSpec(
        name="kendall_tau",
        function=KendallTauMetric(),
        higher_is_better=True,
        category="rank_correlation",
        description="Kendall tau rank correlation.",
    ),
    "precision_at_k": MetricSpec(
        name="precision_at_k",
        function=PrecisionAtKMetric(),
        higher_is_better=True,
        category="top_k",
        description="Top-k overlap by absolute value.",
    ),
}

METRICS = {name: spec.function for name, spec in METRIC_SPECS.items()}
METRICS.update(
    {alias: METRIC_SPECS[canonical].function for alias, canonical in METRIC_ALIASES.items()}
)
