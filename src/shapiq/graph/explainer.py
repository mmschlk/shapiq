"""Docstring."""

from __future__ import annotations

from shapiq.explainer.base import Explainer


class GraphExplainer(Explainer):
    """Docstring."""

    def __init__(self) -> None:
        """Docstring."""
        # TO DO
        self.baseline_value = self._compute_baseline_value()

    def _compute_baseline_value(self) -> float:
        """Docstring."""
        # TO DO
        return 0.0
