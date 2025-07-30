"""Configuration for pytest fixtures for shapiq_games."""

from __future__ import annotations

import pytest


@pytest.fixture
def mae_loss():
    """Returns the mean absolute error loss function."""
    from sklearn.metrics import mean_absolute_error

    return mean_absolute_error
