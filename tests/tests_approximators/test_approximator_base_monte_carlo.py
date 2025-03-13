"""This test module contains all tests regarding the base monte-carlo approximator many other
approximators are based on."""

import pytest

from shapiq.approximator.montecarlo import MonteCarlo
from shapiq.games.benchmark import DummyGame
from shapiq.interaction_values import InteractionValues


@pytest.mark.parametrize(
    "n, max_order, index, top_order, stratify_intersection, stratify_coalition_size",
    [
        (7, 2, "SII", False, True, False),
        (7, 2, "wrong_index", False, False, True),
    ],
)
def test_initialization(
    n, max_order, index, top_order, stratify_intersection, stratify_coalition_size
):
    """Tests the initialization of the MonteCarlo approximator."""

    if index == "wrong_index":
        with pytest.raises(ValueError):
            _ = MonteCarlo(n, max_order, index=index, top_order=top_order)
        return

    approximator = MonteCarlo(
        n,
        max_order,
        index=index,
        top_order=top_order,
        stratify_intersection=stratify_intersection,
        stratify_coalition_size=stratify_coalition_size,
    )
    assert approximator.n == n
    assert approximator.max_order == max_order
    assert approximator.top_order is top_order
    assert approximator.min_order == (max_order if top_order else 0)
    assert approximator.iteration_cost == 1
    assert approximator.index == index


@pytest.mark.parametrize(
    "n, index, max_order, budget, stratify_intersection, stratify_coalition_size",
    [
        (7, "SII", 2, 100, False, False),
        (7, "SII", 2, 100, True, False),
        (7, "SII", 2, 100, False, True),
        (7, "SII", 2, 100, True, True),
        (7, "SV", 1, 100, True, True),
        (7, "k-SII", 2, 100, True, True),
        (7, "STII", 2, 100, False, False),
        (7, "FSII", 2, 100, False, False),
        (7, "BII", 2, 100, False, False),
        (7, "CHII", 2, 100, False, False),
    ],
)
def test_approximate(n, index, max_order, budget, stratify_intersection, stratify_coalition_size):
    """Tests the approximation of the MonteCarlo approximation."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = MonteCarlo(
        n,
        max_order,
        index=index,
        random_state=42,
        stratify_intersection=stratify_intersection,
        stratify_coalition_size=stratify_coalition_size,
    )
    estimates = approximator.approximate(budget, game)
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == max_order
    if index == "FSII":
        assert estimates.min_order == max_order
    else:
        assert estimates.min_order == 0

    # check that the budget is respected
    assert game.access_counter <= budget + 2

    # for order 2 (max_order) the interaction between player 1 and 2 is the most important (1.0)
    if index == "SV":
        interaction_estimate = estimates[(interaction[0],)]
    else:
        interaction_estimate = estimates[interaction]

    assert interaction_estimate != 0
