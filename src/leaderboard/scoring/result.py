"""Result data structures for leaderboard scoring."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LeaderboardRow:
    """One row in a computed leaderboard scoring result."""
    approximator_name: str
    score: float
    higher_is_better: bool
    rank: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoringResult:
    """Complete result of one leaderboard scoring run."""
    scorer_name: str
    metric_name: str
    rows: list[LeaderboardRow]
    metadata: dict[str, object] = field(default_factory=dict)