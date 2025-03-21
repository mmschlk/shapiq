"""This test module contains all tests regarding the FSII regression approximator."""

from copy import copy, deepcopy

import numpy as np
import pytest

from shapiq import ExactComputer
from shapiq.approximator import RegressionFBII
from shapiq.games.benchmark import DummyGame


@pytest.mark.parametrize(
    "n, max_order",
    [
        (3, 1),
        (3, 1),
        (3, 2),
        (3, 2),
        (7, 2),  # used in subsequent tests
        (10, 3),
    ],
)
def test_initialization(n, max_order):
    """Tests the initialization of the RegressionFBII approximator."""
    approximator = RegressionFBII(n, max_order)
    assert approximator.n == n
    assert approximator.max_order == max_order
    assert approximator.top_order is False
    assert approximator.min_order == 0
    assert approximator.iteration_cost == 1
    assert approximator.index == "FBII"

    approximator_copy = copy(approximator)
    approximator_deepcopy = deepcopy(approximator)
    approximator_deepcopy.index = "something"
    assert approximator_copy == approximator  # check that the copy is equal
    assert approximator_deepcopy != approximator  # check that the deepcopy is not equal
    approximator_string = str(approximator)
    assert repr(approximator) == approximator_string
    assert hash(approximator) == hash(approximator_copy)
    assert hash(approximator) != hash(approximator_deepcopy)


def test_extreme_weight_initialisation():
    """Tests if the attributes and properties of approximators are set correctly."""

    # In local tests this number still did not trigger an OverflowError
    n_players = 1000
    game = DummyGame(n=n_players, interaction=(1, 2))
    approximator = RegressionFBII(n=game.n_players, max_order=1, random_state=42)
    approximator.approximate(200, game)

    # This should trigger a warning
    n_players = 2000
    game = DummyGame(n=n_players, interaction=(1, 2))
    with pytest.warns(UserWarning):
        # We approximate weights very extreme
        approximator = RegressionFBII(n=game.n_players, max_order=1, random_state=42)
        approximator.approximate(200, game)


def test_approximate_bv_equality(cooking_game):
    """Tests the approximation of the RegressionFBII approximator to be equal to BV."""
    n_players = 3
    game = cooking_game
    exact_computer = ExactComputer(n_players, game)
    banzhaf = exact_computer("BV")
    fbii_exac = exact_computer("FBII", order=1)
    approximator = RegressionFBII(n=n_players, max_order=1)

    fbii_approx = approximator.approximate(budget=2**n_players, game=game)
    assert np.allclose(fbii_exac.values, fbii_approx.values, atol=1e-2)

    # disregard the baseline value
    assert np.allclose(banzhaf.values[1:], fbii_approx.values[1:], atol=1e-2)


def test_approximate_fbii(paper_game):
    """Tests the approximation of the RegressionFBII approximator to be equal to the results from
    http://jmlr.org/papers/v24/22-0202.html.
    """

    n_players = 11
    game = paper_game
    approximator = RegressionFBII(n=n_players, max_order=2)
    banzhaf_fbii = approximator.approximate(budget=2**n_players, game=game)

    assert np.allclose(np.round(banzhaf_fbii.values[1:12], 2), 1.08, atol=1e-4)
    assert np.allclose(banzhaf_fbii.values[12:], -0.113, atol=1e-2)
