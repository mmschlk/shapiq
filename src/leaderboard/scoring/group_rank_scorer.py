"""Group-wise ranking scorer for comparable benchmark records."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from leaderboard.metrics.registry import METRIC_SPECS
from leaderboard.scoring.base import LeaderboardScorer
from leaderboard.scoring.result import (
    GroupScoreRow,
    GroupScoringResult,
    LeaderboardRow,
    ScoringResult,
)
from leaderboard.scoring.scorer_utils import (
    aggregate_seeds_in_group,
    build_context,
    filter_valid_records,
    get_metric_metadata,
    get_metric_value,
    group_records,
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
        valid_records = filter_valid_records(records)

        groups = group_records(valid_records, self.group_keys)
        group_results = self._score_groups(groups)
        leaderboard_rows = self._aggregate_group_ranks(group_results)

        return ScoringResult(
            scorer_name=self.name,
            context=build_context(valid_records, self.group_keys),
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

    def _score_groups(
        self,
        groups: dict[tuple[object, ...], list[dict[str, object]]],
    ) -> list[GroupScoringResult]:
        """Rank approximators inside each group for all available metrics."""
        results: list[GroupScoringResult] = []

        for group_key_tuple, group_records in groups.items():
            group_key = dict(zip(self.group_keys, group_key_tuple, strict=True))
            aggregated_records = aggregate_seeds_in_group(
                group_records,
                self.group_keys,
            )

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

            metric_value = get_metric_value(record, metric_name)
            if metric_value is None:
                continue

            metric_metadata = get_metric_metadata(record, metric_name)
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
