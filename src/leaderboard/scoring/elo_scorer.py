from dataclasses import dataclass
from itertools import combinations

from leaderboard.metrics.registry import METRIC_ALIASES, METRIC_SPECS
from leaderboard.scoring import LeaderboardScorer, ScoringResult
from leaderboard.scoring.result import ScoringContext
from leaderboard.scoring.scorer_utils import (
    aggregate_seeds_in_group,
    build_context,
    filter_valid_records,
    get_metric_value,
    group_records,
)


@dataclass(frozen=True)
class PairwiseMatch:
    """One pairwise comparison between two approximators.

    A pairwise match is created from one comparable benchmark group and one
    metric. It stores both the original metric values and the derived match
    scores used by the Elo update.

    Example:
        For MSE, lower values are better. If approximator A has ``mse=0.01``
        and approximator B has ``mse=0.03``, then A wins the comparison.
        The resulting scores are ``score_a=1.0`` and ``score_b=0.0``.

    Attributes:
        approximator_a: Name of the first approximator in the comparison.
        approximator_b: Name of the second approximator in the comparison.
        metric_name: Name of the metric used to decide the match outcome,
            for example ``mse`` or ``spearman``.
        metric_value_a: Aggregated metric value of ``approximator_a`` in the
            comparable group. Usually this is the mean over seeds.
        metric_value_b: Aggregated metric value of ``approximator_b``.
        score_a: Match score for ``approximator_a`` used in the Elo update.
            ``1.0`` means win, ``0.5`` means tie, and ``0.0`` means loss.
        score_b: Match score for ``approximator_b``.
        group_key: Dictionary describing the comparable benchmark group from
            which the match was created.
    """

    approximator_a: str
    approximator_b: str
    metric_name: str
    metric_value_a: float
    metric_value_b: float
    score_a: float
    score_b: float
    group_key: dict[str, object]


class EloScorer(LeaderboardScorer):
    """Scorer based on the Elo system using pairwise approximator comparisons."""

    name = "elo"
    higher_is_better = True

    def __init__(
            self,
            initial_elo: float = 1000.0,
            k_factor: float = 16.0,
            tie_tolerance: float = 0.0,
            group_keys: list[str] | None = None,
            metric_names: list[str] | None = None,
            game_names: list[str] | None = None,
            indices: list[str] | None = None,
            budgets: list[int] | None = None,
    ) -> None:
        """Create an Elo scorer.

        Without filter arguments, the scorer computes a global Elo leaderboard over
        all available comparable groups and metrics. Filter arguments restrict the
        scoring context.

        Args:
            initial_elo: Starting Elo value for each approximator.
            k_factor: Multiplication factor determining Elo gain and loss per match.
            tie_tolerance: Tolerance for treating small metric differences as ties.
            group_keys: Record keys used to form comparable benchmark groups.
            metric_names: Optional metric names to include. ``None`` means all metrics.
            game_names: Optional game names to include. ``None`` means all games.
            indices: Optional interaction indices to include. ``None`` means all indices.
            budgets: Optional budgets to include. ``None`` means all budgets.
        """
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.tie_tolerance = tie_tolerance
        self.group_keys = group_keys or [
            "game_id",
            "game_name",
            "index",
            "max_order",
            "budget",
            "ground_truth_method",
        ]

        self.metric_names = self._normalize_metric_names(metric_names)
        self.game_names = game_names
        self.indices = indices
        self.budgets = budgets

    def score(self, records: list[dict[str, object]]) -> ScoringResult:
        """Compute Elo ratings from benchmark records."""
        valid_records = filter_valid_records(records)
        selected_records = self._filter_records_by_context(valid_records)

        groups = group_records(selected_records, self.group_keys)
        matches = self._build_pairwise_matches(groups)

        # Temporary until Elo update is implemented.
        raise NotImplementedError(
            f"Built {len(matches)} pairwise matches. Elo update is not implemented yet."
        )

    def _build_pairwise_matches(
            self,
            groups: dict[tuple[object, ...], list[dict[str, object]]],
    ) -> list[PairwiseMatch]:
        """Build pairwise matches from selected comparable groups and metrics."""
        matches: list[PairwiseMatch] = []

        for group_key_tuple, group_records in groups.items():
            group_key = dict(zip(self.group_keys, group_key_tuple, strict=True))
            aggregated_records = aggregate_seeds_in_group(group_records, self.group_keys)

            for metric_name in self.metric_names:
                metric_spec = METRIC_SPECS[metric_name]
                metric_records = self._records_with_metric(
                    records=aggregated_records,
                    metric_name=metric_name,
                )

                if len(metric_records) < 2:
                    continue

                for record_a, record_b in combinations(metric_records, 2):
                    match = self._build_pairwise_match(
                        record_a=record_a,
                        record_b=record_b,
                        metric_name=metric_name,
                        higher_is_better=metric_spec.higher_is_better,
                        group_key=group_key,
                    )
                    if match is not None:
                        matches.append(match)

        return matches

    def _records_with_metric(
        self,
        *,
        records: list[dict[str, object]],
        metric_name: str,
    ) -> list[dict[str, object]]:
        """Return records that contain a usable value for the selected metric."""
        return [
            record
            for record in records
            if get_metric_value(record, metric_name) is not None
        ]

    def _build_pairwise_match(
        self,
        *,
        record_a: dict[str, object],
        record_b: dict[str, object],
        metric_name: str,
        higher_is_better: bool,
        group_key: dict[str, object],
    ) -> PairwiseMatch | None:
        """Build one pairwise match from two aggregated approximator records."""
        approximator_a = record_a.get("approximator_name")
        approximator_b = record_b.get("approximator_name")

        if not isinstance(approximator_a, str) or not isinstance(approximator_b, str):
            return None

        metric_value_a = get_metric_value(record_a, metric_name)
        metric_value_b = get_metric_value(record_b, metric_name)

        if metric_value_a is None or metric_value_b is None:
            return None

        score_a, score_b = self._compare_metric_values(
            value_a=metric_value_a,
            value_b=metric_value_b,
            higher_is_better=higher_is_better,
        )

        return PairwiseMatch(
            approximator_a=approximator_a,
            approximator_b=approximator_b,
            metric_name=metric_name,
            metric_value_a=metric_value_a,
            metric_value_b=metric_value_b,
            score_a=score_a,
            score_b=score_b,
            group_key=group_key,
        )

    def _compare_metric_values(
        self,
        *,
        value_a: float,
        value_b: float,
        higher_is_better: bool,
    ) -> tuple[float, float]:
        """Convert two metric values into Elo match scores."""
        if abs(value_a - value_b) <= self.tie_tolerance:
            return 0.5, 0.5

        if higher_is_better:
            if value_a > value_b:
                return 1.0, 0.0
            return 0.0, 1.0

        if value_a < value_b:
            return 1.0, 0.0
        return 0.0, 1.0

    def _normalize_metric_names(self, metric_names: list[str] | None) -> list[str]:
        """Normalize selected metric names and aliases.

        Args:
            metric_names: Selected metric names or aliases. ``None`` means all
                registered metrics.

        Returns:
            Canonical metric names.

        Raises:
            KeyError: If a metric name is unknown.
        """
        if metric_names is None:
            return list(METRIC_SPECS)

        normalized_names = []
        for metric_name in metric_names:
            normalized_name = METRIC_ALIASES.get(metric_name, metric_name)
            if normalized_name not in METRIC_SPECS:
                msg = f"Unknown metric: {metric_name}"
                raise KeyError(msg)
            normalized_names.append(normalized_name)

        return list(dict.fromkeys(normalized_names))

    def _filter_records_by_context(
            self,
            records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Filter records according to the selected scoring context."""
        return [record for record in records if self._record_matches_context(record)]

    def _record_matches_context(self, record: dict[str, object]) -> bool:
        """Return whether a record belongs to the selected scoring context."""
        if self.game_names is not None and record.get("game_name") not in self.game_names:
            return False

        if self.indices is not None and record.get("index") not in self.indices:
            return False

        if self.budgets is not None and record.get("budget") not in self.budgets:
            return False

        return True

    def _build_context(self, records: list[dict[str, object]]) -> ScoringContext:
        """Build context describing the selected Elo scoring subset."""
        context = build_context(records, self.group_keys)

        return ScoringContext(
            game_names=context.game_names,
            indices=context.indices,
            budgets=context.budgets,
            metric_names=list(self.metric_names),
            group_keys=context.group_keys,
        )