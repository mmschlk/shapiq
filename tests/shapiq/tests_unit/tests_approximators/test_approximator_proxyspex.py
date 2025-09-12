"""Tests for the ProxySPEX approximator."""

from __future__ import annotations

import pytest

from shapiq.approximator.sparse.proxyspex import ProxySPEX
from shapiq.interaction_values import InteractionValues
from tests.shapiq.markers import skip_if_no_lightgbm


@skip_if_no_lightgbm
@pytest.mark.external_libraries
def test_initialization_defaults():
    """Test that ProxySPEX initializes with correct defaults."""
    n = 10
    proxyspex = ProxySPEX(n=n)

    # Check ProxySPEX default values
    assert proxyspex.n == n
    assert proxyspex.max_order == n  # Default is None, which becomes n
    assert proxyspex.index == "FBII"
    assert proxyspex.top_order is False
    assert proxyspex.transform_type == "fourier"
    assert proxyspex.decoder_type == "proxyspex"  # For proxyspex
    assert proxyspex.decoder_args["max_depth"][1] == n  # Depth n


@pytest.mark.parametrize(
    ("n", "index", "max_order", "top_order"),
    [
        (7, "STII", 2, False),
        (7, "FBII", 3, True),
        (20, "FSII", None, False),
    ],
)
@skip_if_no_lightgbm
@pytest.mark.external_libraries
def test_initialization_custom(n, index, max_order, top_order):
    """Test ProxySPEX initialization with custom parameters."""
    proxyspex = ProxySPEX(
        n=n,
        index=index,
        max_order=max_order,
        top_order=top_order,
    )

    assert proxyspex.n == n
    assert proxyspex.max_order == (n if max_order is None else max_order)
    assert proxyspex.index == index
    assert proxyspex.top_order is top_order
    assert proxyspex.transform_type == "fourier"


@pytest.mark.parametrize(
    ("n", "interactions", "budget"),
    [
        (10, {(), (1,), (1, 2)}, 1000),
        (7, {(), (1,), (1, 2)}, 800),
    ],
)
def test_approximate(n, interactions, budget):
    """Test ProxySPEX approximation functionality."""

    def dummy_game(X):
        return [sum(1 for interaction in interactions if all(x[i] for i in interaction)) for x in X]

    # Initialize SPEX approximator
    proxyspex = ProxySPEX(n=n, random_state=42)

    # Perform approximation
    estimates = proxyspex.approximate(budget, dummy_game)

    # Verify the result structure
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == n
    assert estimates.min_order == 0  # Default top_order is False
    assert estimates.index == "FBII"
    assert estimates.estimated
    assert estimates.estimation_budget > 0

    # Check that values are not empty
    assert len(estimates.values) > 0

    # Check that the target interaction has a non-zero value
    for interaction in interactions:
        assert interaction in estimates.interaction_lookup
        assert abs(estimates[interaction]) > 0
        # The dummy game should return approximately 1.0 for the target interaction
        assert estimates[interaction] == pytest.approx(1.0, abs=0.5)
