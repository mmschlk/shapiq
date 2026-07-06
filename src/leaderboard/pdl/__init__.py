"""Prototype pairwise decision learning helpers for leaderboard runs."""

from __future__ import annotations

from .pairwise_dataset import aggregate_metric_scores, build_pairwise_dataset, flatten_runs

__all__ = ["aggregate_metric_scores", "build_pairwise_dataset", "flatten_runs"]
