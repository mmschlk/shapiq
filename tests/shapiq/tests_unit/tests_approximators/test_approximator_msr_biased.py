"""Tests for the ProxySPEX approximator."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.approximator.proxy import MSRBiased
from shapiq.game_theory.exact import ExactComputer
from shapiq.interaction_values import InteractionValues


def test_initialization_defaults():
    """Test that MSR-b initializes with correct defaults."""
    n = 10
    msr_b = MSRBiased(n=n)

    # Check ProxySHAP default values
    assert msr_b.n == n
    assert msr_b.max_order == 2
    assert msr_b.index == "SII"


@pytest.mark.parametrize(
    ("n", "index", "max_order"),
    [
        (7, "STII", 2),
        (7, "FBII", 3),
        (20, "FSII", 20),
    ],
)
def test_initialization_custom(n, index, max_order):
    """Test MSR-b initialization with custom parameters."""
    msr_b = MSRBiased(
        n=n,
        index=index,
        max_order=max_order,
    )

    assert msr_b.n == n
    assert msr_b.max_order == (n if max_order is None else max_order)
    assert msr_b.index == index


@pytest.mark.parametrize(
    ("n", "interactions", "budget"),
    [
        (10, {(), (1,), (1, 2)}, 1024),
        (7, {(), (1,), (1, 2)}, 128),
    ],
)
def test_approximate(n, interactions, budget):
    """Test ProxySHAP approximation functionality."""

    def dummy_game(X):
        return np.array(
            [sum(1 for interaction in interactions if all(x[i] for i in interaction)) for x in X]
        )

    # Initialize MSR-b approximator
    msr_b = MSRBiased(n=n, random_state=42, index="SII", max_order=2)

    exact_computer = ExactComputer(game=dummy_game, n_players=n)
    gt_values = exact_computer(index="SII", order=2)
    # Perform approximation
    estimates = msr_b.approximate(budget, dummy_game)

    # Verify the result structure
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == 2
    assert estimates.min_order == 0  # Default top_order is False
    assert estimates.index == "SII"
    assert estimates.estimated
    assert estimates.estimation_budget > 0

    # Check that values are not empty
    assert len(estimates.values) > 0

    for interaction in interactions:
        if interaction == ():
            continue
        assert np.allclose(estimates[interaction], gt_values[interaction], atol=1e-5)
