"""Shared utilities for leaderboard scorers."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev

from leaderboard.metrics.registry import METRIC_SPECS
from leaderboard.scoring.result import ScoringContext


def filter_valid_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return records that did not fail during benchmark execution."""
    return [record for record in records if record.get("run_failed") is not True]


def group_records(
    records: list[dict[str, object]],
    group_keys: list[str],
) -> dict[tuple[object, ...], list[dict[str, object]]]:
    """Group records by comparable benchmark keys."""
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)

    for record in records:
        group_key = tuple(record.get(key) for key in group_keys)
        groups[group_key].append(record)

    return dict(groups)


def get_metric_value(
    record: dict[str, object],
    metric_name: str,
) -> float | None:
    """Extract a metric value from nested or flattened records."""
    metrics = record.get("metrics")
    value = metrics.get(metric_name) if isinstance(metrics, dict) else record.get(metric_name)

    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)

    return None


def get_metric_metadata(
    record: dict[str, object],
    metric_name: str,
) -> dict[str, object]:
    """Extract aggregation metadata for one metric if available."""
    metric_stats = record.get("metric_stats")

    if isinstance(metric_stats, dict):
        stats = metric_stats.get(metric_name)
        if isinstance(stats, dict):
            return dict(stats)

    return {}


def build_context(
    records: list[dict[str, object]],
    group_keys: list[str],
) -> ScoringContext:
    """Build context describing the scored records."""
    return ScoringContext(
        game_names=unique_str_values(records, "game_name"),
        indices=unique_str_values(records, "index"),
        budgets=unique_int_values(records, "budget"),
        metric_names=list(METRIC_SPECS),
        group_keys=list(group_keys),
    )


def aggregate_seeds_in_group(
    records: list[dict[str, object]],
    group_keys: list[str],
) -> list[dict[str, object]]:
    """Aggregate metric values over seeds for each approximator in one group."""
    records_by_approximator: dict[str, list[dict[str, object]]] = defaultdict(list)

    for record in records:
        approximator_name = record.get("approximator_name")
        if not isinstance(approximator_name, str):
            continue
        records_by_approximator[approximator_name].append(record)

    aggregated_records: list[dict[str, object]] = []

    for approximator_name, approximator_records in records_by_approximator.items():
        aggregated_metrics: dict[str, float] = {}
        metric_stats: dict[str, dict[str, object]] = {}

        for metric_name in METRIC_SPECS:
            metric_values = [
                metric_value
                for record in approximator_records
                if (metric_value := get_metric_value(record, metric_name)) is not None
            ]

            if not metric_values:
                continue

            metric_mean = float(mean(metric_values))
            metric_std = float(stdev(metric_values)) if len(metric_values) > 1 else 0.0

            aggregated_metrics[metric_name] = metric_mean
            metric_stats[metric_name] = {
                "mean": metric_mean,
                "std": metric_std,
                "n_records": len(metric_values),
            }

        if not aggregated_metrics:
            continue

        aggregated_record: dict[str, object] = {
            "approximator_name": approximator_name,
            "metrics": aggregated_metrics,
            "metric_stats": metric_stats,
            "n_seed_records": len(approximator_records),
        }

        for key in group_keys:
            aggregated_record[key] = approximator_records[0].get(key)

        aggregated_records.append(aggregated_record)

    return aggregated_records


def unique_str_values(records: list[dict[str, object]], key: str) -> list[str]:
    """Return sorted unique string values for one record key."""
    values = {record[key] for record in records if isinstance(record.get(key), str)}
    return sorted(values)


def unique_int_values(records: list[dict[str, object]], key: str) -> list[int]:
    """Return sorted unique integer values for one record key."""
    values = {
        record[key]
        for record in records
        if isinstance(record.get(key), int) and not isinstance(record.get(key), bool)
    }
    return sorted(values)
