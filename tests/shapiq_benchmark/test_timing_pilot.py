"""Paired timing diagnostics must preserve the estimator and its strict query cap."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


def pilot_module():
    """Load the standalone diagnostic without executing its CLI."""
    path = Path(__file__).resolve().parents[2] / "benchmark/timing_pilot.py"
    specification = importlib.util.spec_from_file_location("timing_pilot", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_paired_live_and_cached_runs_use_identical_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real estimator sees precisely the same values and coalition sequence in each mode."""
    pilot = pilot_module()
    monkeypatch.setattr(
        pilot,
        "make_family",
        lambda *args, **kwargs: (lambda masks: masks @ np.arange(1.0, 4.0), {}),
    )
    monkeypatch.setattr(pilot, "provenance", dict)
    result = pilot.compare(n_players=3, methods=("KernelSHAP",), relative_budgets=(2,), repeats=2)
    assert len(result["records"]) == 2
    assert result["official_timing"] is False
    for row in result["records"]:
        assert row["queries"] <= row["budget"] == 6
        assert row["queries"] == sum(row["batch_sizes"])
        assert row["maximum_estimate_difference"] == 0
        assert row["estimated_uncached_seconds"] >= 0
        assert row["measured_live_seconds"] > 0


def test_timing_pilot_does_not_relax_query_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    """An over-budget estimator cannot obtain a successful diagnostic record."""
    pilot = pilot_module()
    monkeypatch.setattr(
        pilot, "make_family", lambda *args, **kwargs: (lambda masks: masks.sum(axis=1), {})
    )

    class TooMany:
        def approximate(self, budget: int, game: Callable) -> None:
            game(np.ones((budget + 1, 3)))

    monkeypatch.setattr(pilot, "builtin_factory", lambda *args: TooMany())
    with pytest.raises(RuntimeError, match="budget"):
        pilot.compare(n_players=3, methods=("test",), relative_budgets=(2,), repeats=1)
