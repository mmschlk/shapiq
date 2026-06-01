"""Evaluator for computing all registered metrics."""

from __future__ import annotations

from .scorer import Scorer


def compute_all_metrics(ground_truth: object, estimated: object) -> dict[str, float | None]:
    """Compute all canonical metrics given ground truth and estimated values."""
    return Scorer().score(ground_truth, estimated)
