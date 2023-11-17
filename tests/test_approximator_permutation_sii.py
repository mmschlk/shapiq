"""This test module contains all tests regarding the SII permutation sampling approximator."""
import numpy as np
import pytest

from approximator._base import InteractionValues
from approximator.permutation import PermutationSamplingSII
from games import DummyGame


@pytest.mark.parametrize(
    "n, max_order, top_order, expected",
    [
        (3, 1, True, 6),
        (3, 1, False, 6),
        (3, 2, True, 8),
        (3, 2, False, 14),
        (10, 3, False, 120),
    ],
)
def test_initialization(n, max_order, top_order, expected):
    """Tests the initialization of the PermutationSamplingSII approximator."""
    approximator = PermutationSamplingSII(n, max_order, top_order)
    assert approximator.n == n
    assert approximator.max_order == max_order
    assert approximator.top_order == top_order
    assert approximator.min_order == (max_order if top_order else 1)
    assert approximator._iteration_cost == expected


@pytest.mark.parametrize(
    "n, max_order, top_order, budget, batch_size, expected",
    [
        (7, 2, False, 380, 10, {"iteration_cost": 38, "access_counter": 380}),
        (7, 2, False, 500, 10, {"iteration_cost": 38, "access_counter": 494}),
    ]
)
def test_approximate(n, max_order, top_order, budget, batch_size, expected):
    """Tests the approximation of the PermutationSamplingSII approximator."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = PermutationSamplingSII(n, max_order, top_order)
    assert approximator._iteration_cost == expected["iteration_cost"]
    sii_estimates = approximator.approximate(budget, game, batch_size=batch_size)
    assert isinstance(sii_estimates, InteractionValues)
    assert len(sii_estimates.values) == max_order

    # check that the budget is respected
    assert game.access_counter <= budget
    assert game.access_counter == expected["access_counter"]

    # check that the estimates are correct
    # for order 1 player 1 and 2 are the most important the rest should be somewhat equal
    # for order 2 the interaction between player 1 and 2 is the most important
    first_order: np.ndarray = sii_estimates.values[1]
    # sort the players by their importance
    sorted_players: np.ndarray = np.argsort(first_order)[::-1]
    assert (sorted_players[0] == 1 or sorted_players[0] == 2) and \
           (sorted_players[1] == 1 or sorted_players[1] == 2)
    for player_one in sorted_players[2:]:
        for player_two in sorted_players[2:]:
            pytest.approx(player_one, player_two, 0.1)

    second_order: np.ndarray = sii_estimates.values[2]
    pytest.approx(second_order[interaction], 1.0, 0.0001)

    # check efficiency
    efficiency = np.sum(first_order)
    pytest.approx(efficiency, 2.0, 0.001)
