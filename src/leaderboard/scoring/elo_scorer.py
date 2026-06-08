from dataclasses import dataclass
from itertools import combinations

from leaderboard.metrics.registry import METRIC_ALIASES, METRIC_SPECS
from leaderboard.scoring import LeaderboardScorer, ScoringResult
from leaderboard.scoring.result import ScoringContext, LeaderboardRow
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

        ratings, stats = self._compute_elo(matches)
        leaderboard_rows = self._build_leaderboard_rows(ratings, stats)

        return ScoringResult(
            scorer_name=self.name,
            context=self._build_context(selected_records),
            rows=leaderboard_rows,
            group_results=[],
            metadata={
                "n_input_records": len(records),
                "n_valid_records": len(valid_records),
                "n_selected_records": len(selected_records),
                "n_groups": len(groups),
                "n_matches": len(matches),
                "initial_elo": self.initial_elo,
                "k_factor": self.k_factor,
                "tie_tolerance": self.tie_tolerance,
                "seed_aggregation": "mean",
                "ordering_strategy": "deterministic",
            },
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

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Compute expected Elo score for A against B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _update_ratings(
            self,
            *,
            rating_a: float,
            rating_b: float,
            score_a: float,
            score_b: float,
    ) -> tuple[float, float]:
        """Update two Elo ratings after one pairwise match."""
        expected_a = self._expected_score(rating_a, rating_b)
        expected_b = self._expected_score(rating_b, rating_a)

        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)

        return new_rating_a, new_rating_b

    def _initialize_match_stats(
            self,
            matches: list[PairwiseMatch],
    ) -> dict[str, dict[str, int]]:
        """Initialize match counters for all approximators occurring in matches.

        Args:
            matches: Pairwise matches from which all participating approximators
                are collected.

        Returns:
            A dictionary mapping each approximator name to initialized match
            counters. Each counter dictionary contains ``n_matches``, ``wins``,
            ``losses``, and ``ties``, all initialized to zero.
        """
        approximators = set()
        for match in matches:
            approximators.add(match.approximator_a)
            approximators.add(match.approximator_b)

        return {
            approximator: {
                "n_matches": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
            }
            for approximator in approximators
        }


    def _compute_elo(
            self,
            matches: list[PairwiseMatch],
    ) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
        """Compute Elo ratings and match statistics from pairwise matches.

        The method starts every approximator at ``self.initial_elo`` and applies
        the matches sequentially. After each match, both participating
        approximators receive updated Elo ratings based on their current ratings,
        expected scores, actual match scores, and ``self.k_factor``.

        Args:
            matches: Pairwise matches used as Elo update events. Each match contains
                two approximators and their actual scores, where ``1.0`` means win,
                ``0.5`` means tie, and ``0.0`` means loss.

        Returns:
            A tuple containing two dictionaries:

            - The first dictionary maps approximator names to their final Elo
              ratings.
            - The second dictionary maps approximator names to match statistics,
              including ``n_matches``, ``wins``, ``losses``, and ``ties``.
        """
        stats = self._initialize_match_stats(matches)
        ratings = {
            approximator: self.initial_elo
            for approximator in stats
        }

        for match in matches:
            rating_a = ratings[match.approximator_a]
            rating_b = ratings[match.approximator_b]

            new_rating_a, new_rating_b = self._update_ratings(
                rating_a=rating_a,
                rating_b=rating_b,
                score_a=match.score_a,
                score_b=match.score_b,
            )

            ratings[match.approximator_a] = new_rating_a
            ratings[match.approximator_b] = new_rating_b

            self._update_match_stats(stats, match)

        return ratings, stats

    def _update_match_stats(
            self,
            stats: dict[str, dict[str, int]],
            match: PairwiseMatch,
    ) -> None:
        """Update win, loss, tie, and match counters for one pairwise match."""
        stats[match.approximator_a]["n_matches"] += 1
        stats[match.approximator_b]["n_matches"] += 1

        if match.score_a == 0.5 and match.score_b == 0.5:
            stats[match.approximator_a]["ties"] += 1
            stats[match.approximator_b]["ties"] += 1
            return

        if match.score_a > match.score_b:
            stats[match.approximator_a]["wins"] += 1
            stats[match.approximator_b]["losses"] += 1
            return

        stats[match.approximator_a]["losses"] += 1
        stats[match.approximator_b]["wins"] += 1

    def _build_leaderboard_rows(
            self,
            ratings: dict[str, float],
            stats: dict[str, dict[str, int]],
    ) -> list[LeaderboardRow]:
        """Build ranked leaderboard rows from Elo ratings.

        Args:
            ratings: Final Elo ratings per approximator.
            stats: Match statistics per approximator.

        Returns:
            Ranked leaderboard rows sorted by Elo rating in descending order.
        """
        leaderboard_rows = []

        for approximator, rating in ratings.items():
            approximator_stats = stats[approximator]

            leaderboard_row = LeaderboardRow(
                approximator_name=approximator,
                score=rating,
                higher_is_better=True,
                rank=None,
                metadata={
                    "n_matches": approximator_stats["n_matches"],
                    "wins": approximator_stats["wins"],
                    "losses": approximator_stats["losses"],
                    "ties": approximator_stats["ties"],
                },
            )
            leaderboard_rows.append(leaderboard_row)

        leaderboard_rows.sort(key=lambda row: row.score, reverse=True)

        return [
            LeaderboardRow(
                approximator_name=row.approximator_name,
                score=row.score,
                higher_is_better=row.higher_is_better,
                rank=rank,
                metadata=row.metadata,
            )
            for rank, row in enumerate(leaderboard_rows, start=1)
        ]