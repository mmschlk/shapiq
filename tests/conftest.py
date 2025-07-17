"""Conftest with all pytest plugins for both shapiq and shapiq_games."""

from __future__ import annotations

pytest_plugins = [
    "tests.shapiq.fixtures.games",
    "tests.shapiq.fixtures.models",
    "tests.shapiq.fixtures.data",
    "tests.shapiq.fixtures.interaction_values",
]

pytest_plugins.extend(
    [
        "tests.shapiq_games.fixtures.tabular",
    ]
)
