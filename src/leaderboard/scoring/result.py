"""Result data structures for leaderboard scoring."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LeaderboardRow:
    """One row in an aggregated leaderboard scoring result."""

    approximator_name: str
    score: float
    higher_is_better: bool
    rank: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupScoreRow:
    """One scored approximator row inside one comparable group."""

    approximator_name: str
    metric_name: str
    metric_value: float
    rank: int
    higher_is_better: bool
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupScoringResult:
    """Scoring result for one comparable group and one metric."""

    group_key: dict[str, object]
    metric_name: str
    rows: list[GroupScoreRow]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoringContext:
    """Context describing which benchmark subset was scored."""

    game_names: list[str] = field(default_factory=list)
    indices: list[str] = field(default_factory=list)
    budgets: list[int] = field(default_factory=list)
    metric_names: list[str] = field(default_factory=list)
    group_keys: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScoringResult:
    """Complete result of one leaderboard scoring run."""

    scorer_name: str
    context: ScoringContext
    rows: list[LeaderboardRow] = field(default_factory=list)
    group_results: list[GroupScoringResult] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
