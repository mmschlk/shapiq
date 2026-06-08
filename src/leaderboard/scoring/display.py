"""Terminal display helpers for leaderboard scoring results."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from collections.abc import Iterable

    from leaderboard.scoring.result import LeaderboardRow, ScoringResult

def _format_metric_names(result: ScoringResult) -> str:
    """Format metric names from group results or scoring context."""
    group_metrics = {
        group_result.metric_name
        for group_result in result.group_results
    }

    metric_names = sorted(group_metrics) if group_metrics else result.context.metric_names

    return ", ".join(metric_names)


def format_scoring_result(result: ScoringResult) -> str:
    """Format a scoring result as a terminal-friendly ranking table.

    Args:
        result: Complete scoring result produced by a leaderboard scorer.

    Returns:
        Human-readable terminal table as a string.
    """
    used_metrics = sorted({group_result.metric_name for group_result in result.group_results})

    lines = [
        f"Scorer: {result.scorer_name}",
        f"Metrics: {_format_metric_names(result)}",
        f"Games: {', '.join(result.context.game_names) or 'all'}",
        f"Budgets: {_format_values(result.context.budgets) or 'all'}",
        "",
        "Final Ranking:",
        _format_header(),
        _format_separator(),
    ]

    lines.extend(_format_row(row) for row in result.rows)

    if result.metadata:
        lines.extend(["", "Metadata:"])
        for key, value in result.metadata.items():
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def print_scoring_result(
    result: ScoringResult,
    *,
    file: TextIO | None = None,
) -> None:
    """Print a scoring result to the terminal.

    Args:
        result: Complete scoring result produced by a leaderboard scorer.
        file: Optional output stream. Defaults to standard output.
    """
    output = file if file is not None else sys.stdout
    print(format_scoring_result(result), file=output)


def _format_header() -> str:
    """Return the table header."""
    return f"{'Rank':>4}  {'Approximator':<28}  {'Score':>12}  {'Details'}"


def _format_separator() -> str:
    """Return the table separator."""
    return f"{'-' * 4}  {'-' * 28}  {'-' * 12}  {'-' * 30}"


def _format_row(row: LeaderboardRow) -> str:
    """Format one leaderboard row."""
    rank = "-" if row.rank is None else str(row.rank)
    details = _format_metadata(row.metadata)

    return f"{rank:>4}  {row.approximator_name:<28}  {row.score:>12.6g}  {details}"


def _format_metadata(metadata: dict[str, object]) -> str:
    """Format row metadata for compact terminal display."""
    if not metadata:
        return ""

    return ", ".join(f"{key}={value}" for key, value in metadata.items())


def _format_values(values: Iterable[object]) -> str:
    """Format iterable context values."""
    return ", ".join(str(value) for value in values)
