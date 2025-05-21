from __future__ import annotations

import pytest

from shapiq.approximator.sparse.spex import SPEX
from shapiq.games.benchmark import DummyGame
from shapiq.interaction_values import InteractionValues


def test_initialization_defaults():
    """Test that SPEX initializes with correct defaults."""
    n = 10
    spex = SPEX(n=n)

    # Check SPEX default values
    assert spex.n == n
    assert spex.max_order == n  # Default is None, which becomes n
    assert spex.index == "FBII"
    assert spex.top_order is False
    assert spex.transform_type == "fourier"
    assert spex.decoder_args["reconstruct_method_channel"] == "identity-siso"  # For soft decoder


@pytest.mark.parametrize(
    "n, index, max_order, top_order, decoder_type",
    [
        (7, "STII", 2, False, "soft"),
        (7, "FBII", 3, True, "hard"),
        (20, "FSII", None, False, "soft"),
    ],
)
def test_initialization_custom(n, index, max_order, top_order, decoder_type):
    """Test SPEX initialization with custom parameters."""
    spex = SPEX(
        n=n,
        index=index,
        max_order=max_order,
        top_order=top_order,
        decoder_type=decoder_type,
    )

    assert spex.n == n
    assert spex.max_order == (n if max_order is None else max_order)
    assert spex.index == index
    assert spex.top_order is top_order
    assert spex.transform_type == "fourier"

    # Check decoder configuration
    if decoder_type.lower() == "soft":
        assert spex.decoder_args["reconstruct_method_channel"] == "identity-siso"
    else:
        assert spex.decoder_args["reconstruct_method_channel"] == "identity"


@pytest.mark.parametrize(
    "n, interaction, budget",
    [
        (10, (1, 2), 1000),
        (7, (0, 3, 5), 800),
    ],
)
def test_approximate(n, interaction, budget):
    """Test SPEX approximation functionality."""
    # Create a game with a specific interaction
    game = DummyGame(n, interaction)

    # Initialize SPEX approximator
    spex = SPEX(n=n, random_state=42)

    # Perform approximation
    estimates = spex.approximate(budget, game)

    # Verify the result structure
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == n
    assert estimates.min_order == 0  # Default top_order is False
    assert estimates.index == "FBII"
    assert estimates.estimated
    assert estimates.estimation_budget > 0

    # Check that game was called within budget
    assert game.access_counter <= budget + 2

    # Check that values are not empty
    assert len(estimates.values) > 0

    # Check that the target interaction has a non-zero value
    if len(interaction) <= n:
        assert interaction in estimates.interaction_lookup
        assert abs(estimates[interaction]) > 0
        # The dummy game should return approximately 1.0 for the target interaction
        assert estimates[interaction] == pytest.approx(1.0, abs=0.5)


def test_spex_vs_sparse():
    """Test that SPEX behaves identically to Sparse with transform_type='fourier'."""
    from shapiq.approximator.sparse import Sparse

    n = 8
    interaction = (1, 3)
    budget = 800
    random_state = 42

    # Create a game
    game = DummyGame(n, interaction)

    # Initialize both approximators with identical parameters
    spex = SPEX(n=n, random_state=random_state)
    sparse = Sparse(
        n=n, index="FBII", transform_type="fourier", decoder_type="soft", random_state=random_state
    )

    # Run approximation with both
    spex_estimates = spex.approximate(budget, game)

    # Reset game counter for fair comparison
    game.access_counter = 0
    sparse_estimates = sparse.approximate(budget, game)

    # Check that both produced similar results
    assert spex_estimates.index == sparse_estimates.index
    assert spex_estimates.max_order == sparse_estimates.max_order
    assert spex_estimates.min_order == sparse_estimates.min_order

    # Check that the interaction is estimated similarly
    assert abs(spex_estimates[interaction] - sparse_estimates[interaction]) < 0.1


@pytest.mark.parametrize(
    "n, interaction, budget, correct_b, correct_t",
    [
        (10, (1, 2), 1000, 3, 5),
        (10, (1, 2), 450, 3, 3),
        (7, (0, 3, 5), 800, 3, 5),
        (7, (0, 3, 5), 300, 3, 2),
    ],
)
def test_sparsity_parameter(n, interaction, budget, correct_b, correct_t):
    """Test SPEX approximation functionality."""
    # Create a game with a specific interaction
    game = DummyGame(n, interaction)

    # Initialize SPEX approximator
    spex = SPEX(n=n, random_state=42)

    # Run approximation with both
    _ = spex.approximate(budget, game)

    assert spex.query_args["b"] == correct_b
    assert spex.transform_error == correct_t


@pytest.mark.parametrize(
    "n, interaction, budget",
    [
        (10, (1, 2), 100),
        (7, (0, 3, 5), 20),
    ],
)
def test_undersampling(n, interaction, budget):
    """Test SPEX approximation functionality."""
    # Create a game with a specific interaction
    game = DummyGame(n, interaction)

    # Initialize SPEX approximator
    spex = SPEX(n=n, random_state=42)

    with pytest.raises(ValueError):
        _ = spex.approximate(budget, game)
