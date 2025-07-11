"""A collection of differently sized interaction values usefull for testing purposes."""

from __future__ import annotations

import copy
import time
from typing import Literal

import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.utils import count_interactions, generate_interaction_lookup


def get_mock_interaction_value(
    n_players: int,
    *,
    n_interactions: int | Literal["all"] = "all",
    min_order: int = 0,
    max_order: int | None = None,
    random_state: int | None = 42,
    index: str = "Moebius",
    baseline_value: float = 0.0,
) -> InteractionValues:
    """Create a Mock InteractionValue object with some user-defined parameters.

    This function is used to create mock InteractionValue objects for our test environment. The
    function creates random interaction values (uniformly distributed between -1 and 1) for the
    given number of players and interactions. The interactions are sampled uniformly over the size
    of the interaction space. If the number of interactions is set to "all", the function will
    create all possible interactions for the given number of players and orders.

    Args:
        n_players: The number of players.
        n_interactions: The number of interactions.
        min_order: The minimum order of the interaction values.
        max_order: The maximum order of the interaction values.
        random_state: The random state to use for the random number generator.
        index: The index of the interaction values. Defaults to "Moebius".
        baseline_value: The baseline value of the interaction values. Defaults to 0.0.

    Returns:
        The mock InteractionValue object.
    """
    rng = np.random.RandomState(random_state)

    if max_order is None:
        max_order = n_players

    # check if n_interactions is higher or equal to all possible interactions
    if isinstance(n_interactions, int):
        n_available = count_interactions(n=n_players, min_order=min_order, max_order=max_order)
        if n_available <= n_interactions:
            n_interactions = "all"

    if n_interactions == "all":
        interaction_lookup = generate_interaction_lookup(
            players=n_players, min_order=min_order, max_order=max_order
        )
    elif isinstance(n_interactions, int):
        # sample n_interactions uniformly over the size the interaction space
        interaction_lookup = {}
        while len(interaction_lookup) < n_interactions:
            order = rng.randint(min_order, max_order + 1)
            interaction = tuple(sorted(rng.choice(n_players, order, replace=False)))
            if interaction not in interaction_lookup:
                interaction_lookup[interaction] = len(interaction_lookup)
    else:
        msg = f"n_interactions must be 'all' or an integer, but got {n_interactions}"
        raise ValueError(msg)

    # create a random array of interaction values with the same size as the interaction lookup
    values = rng.uniform(-1, 1, len(interaction_lookup))

    # return the interaction values as an InteractionValues object
    return InteractionValues(
        values=values,
        index=index,
        interaction_lookup=interaction_lookup,
        n_players=n_players,
        min_order=min_order,
        max_order=max_order,
        estimated=False,
        baseline_value=baseline_value,
    )


@pytest.mark.parametrize(
    ("n_players", "n_interactions", "min_order", "max_order"),
    [
        (4, 5, 0, 4),
        (4, 5, 1, 2),
        (4, "all", 1, 3),
    ],
)
def test_mock_interaction_value(n_players, n_interactions, min_order, max_order):
    """Test the creation of a mock InteractionValue object."""
    iv = get_mock_interaction_value(
        n_players=n_players,
        n_interactions=n_interactions,
        min_order=min_order,
        max_order=max_order,
    )

    assert iv.n_players == n_players
    assert iv.min_order == min_order
    assert iv.max_order == max_order

    if n_interactions == "all":
        n_interactions = count_interactions(n=n_players, max_order=max_order, min_order=min_order)
        assert len(iv.interaction_lookup) == n_interactions


def test_mock_default_values():
    """Test the creation of a mock InteractionValue object with default values."""
    n_players = 4
    iv = get_mock_interaction_value(n_players=n_players)
    assert iv.n_players == n_players
    assert iv.min_order == 0
    assert iv.max_order == n_players
    assert iv.index == "Moebius"
    assert iv.baseline_value == 0.0
    assert iv.estimated is False
    n_interactions = count_interactions(n=n_players, min_order=0, max_order=n_players)
    assert len(iv.interaction_lookup) == n_interactions


def test_random_state_correctness():
    """Test that random states work properly for mock InteractionValues."""
    iv1 = get_mock_interaction_value(n_players=5, random_state=42, n_interactions=4)
    iv2 = get_mock_interaction_value(n_players=5, random_state=42, n_interactions=4)
    assert np.array_equal(iv1.values, iv2.values)
    assert iv1.interaction_lookup.keys() == iv2.interaction_lookup.keys()
    # iv3 should be different from iv1 and iv2
    iv3 = get_mock_interaction_value(n_players=5, random_state=43, n_interactions=4)
    assert not np.array_equal(iv3.values, iv1.values)
    assert iv3.interaction_lookup.keys() != iv1.interaction_lookup.keys()


def test_mock_is_quick_for_large_params():
    """Tests that the mock InteractionValue is quick to create for large parameters."""
    n_players = 1_000
    n_interactions = 1_000
    min_order = 0
    max_order = 300

    # Create the InteractionValue object and time it
    start_time = time.perf_counter()
    iv = get_mock_interaction_value(
        n_players=n_players,
        n_interactions=n_interactions,
        min_order=min_order,
        max_order=max_order,
    )
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    # Performance check: Should complete in under 0.5 seconds
    assert elapsed < 0.5, f"Mock interaction value creation took too long: {elapsed:.3f}s"

    # Check that the object was created successfully
    assert iv.n_players == n_players
    assert iv.min_order == min_order
    assert iv.max_order == max_order
    assert len(iv.interaction_lookup) == n_interactions


def test_mock_interactions_all_correctly_inferred():
    """Test that the number of interactions is correctly inferred.

    Test that the number of interactions is correctly inferred if n_interactions is larger than
    the number of possible interactions.
    """
    n_players = 4
    n_interactions = 2**n_players * 2  # twice the number of possible interactions
    possible_interactions = count_interactions(n_players, min_order=0, max_order=n_players)
    assert possible_interactions == 2**n_players
    assert n_interactions > possible_interactions
    iv1 = get_mock_interaction_value(n_players=n_players, n_interactions=n_interactions)
    assert len(iv1.interaction_lookup) == possible_interactions
    iv2 = get_mock_interaction_value(n_players=n_players, n_interactions="all")
    assert len(iv2.interaction_lookup) == possible_interactions


_iv_300_300_0_300 = get_mock_interaction_value(
    n_players=300,
    n_interactions=300,
    min_order=0,
    max_order=300,
)

_iv_18_all = get_mock_interaction_value(
    n_players=18,
    n_interactions="all",
)

_iv_10_all = get_mock_interaction_value(
    n_players=10,
    n_interactions="all",
)

_iv_7_all = get_mock_interaction_value(
    n_players=7,
    n_interactions="all",
)


@pytest.fixture
def iv_300_300_0_300():
    """Return an InteractionValue (n_players=300, min_order=0, max_order=300, 300 interactions)."""
    return copy.deepcopy(_iv_300_300_0_300)


@pytest.fixture
def iv_18_all():
    """Return an InteractionValue (n_players=18, min_order=0, max_order=18, 18 interactions)."""
    return copy.deepcopy(_iv_18_all)


@pytest.fixture
def iv_10_all():
    """Return an InteractionValue (n_players=10, min_order=0, max_order=10, 10 interactions)."""
    return copy.deepcopy(_iv_10_all)


@pytest.fixture
def iv_7_all():
    """Return an InteractionValue (n_players=7, min_order=0, max_order=7, 7 interactions)."""
    return copy.deepcopy(_iv_7_all)
