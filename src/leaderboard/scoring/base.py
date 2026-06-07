"""Abstract base class for leaderboard scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from leaderboard.scoring.result import ScoringResult


class LeaderboardScorer(ABC):
    """Abstract base class for leaderboard scoring methods.

    A leaderboard scorer aggregates benchmark run records into a complete
    leaderboard result
    """

    name = "base"
    higher_is_better = True

    @abstractmethod
    def score(self, records: list[dict[str, Any]]) -> ScoringResult:
        """Compute a complete leaderboard scoring result from benchmark records.

        Args:
           records: Benchmark run records

        Returns:
           The complete scoring result, including leaderboard rows and scorer
           metadata.
        """
