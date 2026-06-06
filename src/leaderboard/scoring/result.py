"""Result data structures for leaderboard scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

@dataclass(frozen=True)
class ScoringResult:
    """Complete result of one leaderboard scoring run."""
    scorer_name: str
    context: ScoringContext
    rows: list[LeaderboardRow]
    metadata: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class LeaderboardRow:
    """One row in a computed leaderboard scoring result."""
    approximator_name: str
    score: float
    higher_is_better: bool
    rank: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class ScoringContext:
    """Context describing which benchmark subset was scored."""
    metric_name: str
    game_names: list[str] = field(default_factory=list)
    indices: list[str] = field(default_factory=list)
    budgets: list[int] = field(default_factory=list)
    group_keys: list[str] = field(default_factory=list)