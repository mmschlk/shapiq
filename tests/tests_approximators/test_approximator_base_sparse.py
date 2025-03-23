"""This test module contains all tests regarding the base sparse approximator."""

import pytest
import numpy as np

from shapiq.approximator.sparse import Sparse
from shapiq.games.benchmark import DummyGame
from shapiq.interaction_values import InteractionValues


@pytest.mark.parametrize(
    "n, max_order, index, top_order, transform_type, decoder_type",
    [
        (7, 2, "FBII", False, "fourier", "soft"),
        (7, 2, "FBII", False, "FouRier", "Soft"),
        (7, 2, "FBII", False, "fourier", "Hard"),
        (7, 2, "wrong_index", False, "fourier", "soft"),  # Should raise ValueError
        (7, 2, "wrong_index", False, "fourier", "wrong_dec_type"),  # Should raise ValueError
        (7, 2, "FBII", True, "mobius", None),
        (7, 2, "FBII", True, "Mobius", 'abc'),
        (7, 2, "FBII", False, "invalid_type", None),  # Should raise ValueError
    ],
)

def test_initialization(n, max_order, index, top_order, transform_type, decoder_type):
    """Tests the initialization of the Sparse approximator."""

    if index == "wrong_index":
        with pytest.raises(ValueError):
            _ = Sparse(n,
                       max_order,
                       index=index,
                       top_order=top_order,
                       transform_type=transform_type,
                       decoder_type=decoder_type,
                       )
        return

    if transform_type == "invalid_type":
        with pytest.raises(ValueError):
            _ = Sparse(n,
                       max_order,
                       index=index,
                       top_order=top_order,
                       transform_type=transform_type,
                       decoder_type=decoder_type,
                       )
        return

    if decoder_type == "wrong_dec_type" or (transform_type.lower() == "mobius" and decoder_type is not None):
        with pytest.raises(ValueError):
            _ = Sparse(n,
                       max_order,
                       index=index,
                       top_order=top_order,
                       transform_type=transform_type,
                       decoder_type=decoder_type,
                       )
        return

    approximator = Sparse(
        n,
        max_order,
        index=index,
        top_order=top_order,
        transform_type=transform_type,
        decoder_type=decoder_type,
    )

    assert approximator.n == n
    assert approximator.max_order == max_order
    assert approximator.top_order is top_order
    assert approximator.min_order == (max_order if top_order else 0)
    assert approximator.index == index
    assert approximator.transform_type == transform_type.lower()
    if decoder_type is not None or transform_type.lower() == "fourier":
        channel_method = approximator.decoder_args['reconstruct_method_channel']
        if channel_method is None or decoder_type.lower() == 'soft':
            assert channel_method == 'identity-siso'
        else:
            assert channel_method == 'identity'

@pytest.mark.parametrize(
    "n, index, max_order, budget, transform_type, decoder_type, top_order",
    [
        (20, "STII", 2, 8000, "fourier", "soft", False),
        #(7, "FBII", 2, 100, "fourier", "hard", False),
        #(7, "STI", 2, 100, "fourier", "soft", True),
        #(7, "FBII", 3, 200, "fourier", "hard", False),
        #(7, "STII", 3, 200, "mobius", None, False),
        #(7, "FBII", 2, 100, "mobius", None, True),
    ],
)
def test_approximate(n, index, max_order, budget, transform_type, decoder_type, top_order):
    """Tests the approximation of the Sparse approximator with various configurations."""
    # Create a game with a specific interaction
    interaction = (1, 2)
    game = DummyGame(n, interaction)

    # Initialize the approximator
    approximator = Sparse(
        n,
        max_order,
        index=index,
        top_order=top_order,
        transform_type=transform_type,
        decoder_type=decoder_type,
        random_state=42,
    )

    # Perform approximation with budget
    estimates = approximator.approximate(budget, game)

    # Verify the result structure
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == max_order
    assert estimates.min_order == (max_order if top_order else 0)
    assert estimates.index == index
    assert estimates.estimated
    assert estimates.estimation_budget > 0

    # Check that game was called within budget
    assert game.access_counter <= budget + 2

    # Check that values dictionary is properly populated
    assert len(estimates.values) > 0

    # For non-top_order, check that different interaction orders are present
    if not top_order and max_order >= 2:
        has_first_order = False
        has_second_order = False
        for interaction_key in estimates.interaction_lookup.keys():
            if len(interaction_key) == 1:
                has_first_order = True
            elif len(interaction_key) == 2:
                has_second_order = True

        assert has_first_order
        assert has_second_order

    # Check that the target interaction has a non-zero value if it's within max_order
    if max_order >= 2:
        assert interaction in estimates.interaction_lookup
        assert abs(estimates[interaction]) > 0

