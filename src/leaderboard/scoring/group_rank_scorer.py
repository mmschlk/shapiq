"""Group-wise ranking scorer for comparable benchmark records."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev

from leaderboard.metrics.registry import METRIC_SPECS
from leaderboard.scoring.base import LeaderboardScorer
from leaderboard.scoring.result import (
    GroupScoreRow,
    GroupScoringResult,
    LeaderboardRow,
    ScoringContext,
    ScoringResult,
)


class GroupRankScorer(LeaderboardScorer):
    """Rank approximators inside comparable groups for all available metrics.

    This scorer groups comparable records, ranks
    approximators inside each group for each available metric, and aggregates
    ranks into a simple average-rank leaderboard.
    """

    name = "group_rank"
    higher_is_better = False

    def __init__(
        self,
        group_keys: list[str] | None = None,
    ) -> None:
        """Create a group-wise ranking scorer.

        Args:
            group_keys: Record keys used to form comparable benchmark groups.
        """
        self.group_keys = group_keys or [
            "game_id",
            "game_name",
            "index",
            "max_order",
            "budget",
            "ground_truth_method",
        ]

    def score(self, records: list[dict[str, object]]) -> ScoringResult:
        """Compute group-wise metric rankings and aggregate average ranks."""
        valid_records = [record for record in records if record.get("run_failed") is not True]

        groups = self._group_records(valid_records)
        group_results = self._score_groups(groups)
        leaderboard_rows = self._aggregate_group_ranks(group_results)

        return ScoringResult(
            scorer_name=self.name,
            context=self._build_context(valid_records),
            rows=leaderboard_rows,
            group_results=group_results,
            metadata={
                "n_input_records": len(records),
                "n_valid_records": len(valid_records),
                "n_groups": len(groups),
                "n_group_results": len(group_results),
                "seed_aggregation": "mean",
            },
        )

    def _group_records(
        self,
        records: list[dict[str, object]],
    ) -> dict[tuple[object, ...], list[dict[str, object]]]:
        """Group records by comparable benchmark keys."""
        groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)

        for record in records:
            group_key = tuple(record.get(key) for key in self.group_keys)
            groups[group_key].append(record)

        return dict(groups)

    def _score_groups(
        self,
        groups: dict[tuple[object, ...], list[dict[str, object]]],
    ) -> list[GroupScoringResult]:
        """Rank approximators inside each group for all available metrics."""
        results: list[GroupScoringResult] = []

        for group_key_tuple, group_records in groups.items():
            group_key = dict(zip(self.group_keys, group_key_tuple, strict=True))
            aggregated_records = self._aggregate_seeds_in_group(group_records)

            for metric_name, metric_spec in METRIC_SPECS.items():
                metric_rows = self._build_metric_rows(
                    records=aggregated_records,
                    metric_name=metric_name,
                    higher_is_better=metric_spec.higher_is_better,
                )

                if len(metric_rows) < 2:
                    continue

                results.append(
                    GroupScoringResult(
                        group_key=group_key,
                        metric_name=metric_name,
                        rows=metric_rows,
                        metadata={
                            "n_approximators": len(metric_rows),
                            "n_raw_records": len(group_records),
                            "n_seed_aggregated_records": len(aggregated_records),
                        },
                    )
                )

        return results

    def _build_metric_rows(
        self,
        *,
        records: list[dict[str, object]],
        metric_name: str,
        higher_is_better: bool,
    ) -> list[GroupScoreRow]:
        """Build ranked rows for one seed-aggregated group and one metric."""
        values: list[tuple[str, float, dict[str, object]]] = []

        for record in records:
            approximator_name = record.get("approximator_name")
            if not isinstance(approximator_name, str):
                continue

            metric_value = self._get_metric_value(record, metric_name)
            if metric_value is None:
                continue

            metric_metadata = self._get_metric_metadata(record, metric_name)
            values.append((approximator_name, metric_value, metric_metadata))

        values.sort(
            key=lambda item: item[1],
            reverse=higher_is_better,
        )

        return [
            GroupScoreRow(
                approximator_name=approximator_name,
                metric_name=metric_name,
                metric_value=metric_value,
                rank=rank,
                higher_is_better=higher_is_better,
                metadata=metadata,
            )
            for rank, (approximator_name, metric_value, metadata) in enumerate(values, start=1)
        ]

    def _aggregate_group_ranks(
        self,
        group_results: list[GroupScoringResult],
    ) -> list[LeaderboardRow]:
        """Aggregate group ranks into one average-rank leaderboard."""
        ranks_by_approximator: dict[str, list[float]] = defaultdict(list)

        for group_result in group_results:
            for row in group_result.rows:
                ranks_by_approximator[row.approximator_name].append(float(row.rank))

        rows = [
            LeaderboardRow(
                approximator_name=approximator_name,
                score=float(mean(ranks)),
                higher_is_better=False,
                metadata={
                    "n_rankings": len(ranks),
                },
            )
            for approximator_name, ranks in ranks_by_approximator.items()
            if ranks
        ]

        rows.sort(key=lambda row: row.score)

        return [
            LeaderboardRow(
                approximator_name=row.approximator_name,
                score=row.score,
                higher_is_better=row.higher_is_better,
                rank=rank,
                metadata=row.metadata,
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def _get_metric_value(
        self,
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

    def _build_context(self, records: list[dict[str, object]]) -> ScoringContext:
        """Build context describing the scored records."""
        return ScoringContext(
            game_names=_unique_str_values(records, "game_name"),
            indices=_unique_str_values(records, "index"),
            budgets=_unique_int_values(records, "budget"),
            metric_names=list(METRIC_SPECS),
            group_keys=list(self.group_keys),
        )

    def _aggregate_seeds_in_group(
        self,
        records: list[dict[str, object]],
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
                    if (metric_value := self._get_metric_value(record, metric_name)) is not None
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

            for key in self.group_keys:
                aggregated_record[key] = approximator_records[0].get(key)

            aggregated_records.append(aggregated_record)

        return aggregated_records

    def _get_metric_metadata(
        self,
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


def _unique_str_values(records: list[dict[str, object]], key: str) -> list[str]:
    """Return sorted unique string values for one record key."""
    values = {record[key] for record in records if isinstance(record.get(key), str)}
    return sorted(values)


def _unique_int_values(records: list[dict[str, object]], key: str) -> list[int]:
    """Return sorted unique integer values for one record key."""
    values = {
        record[key]
        for record in records
        if isinstance(record.get(key), int) and not isinstance(record.get(key), bool)
    }
    return sorted(values)
