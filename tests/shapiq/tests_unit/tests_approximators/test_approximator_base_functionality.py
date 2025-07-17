"""This test module tests base functionality of approximators."""

from __future__ import annotations

from copy import copy, deepcopy

import pytest

from shapiq.approximator import Approximator, KernelSHAP
from shapiq.games.benchmark import DummyGame
from tests.shapiq.utils import get_concrete_class


def test_approximator_init():
    """Tests if the attributes and properties of approximators are set correctly."""
    n_players = 5
    game = DummyGame(n=n_players, interaction=(1, 2))

    approximator = KernelSHAP(n=game.n_players, max_order=1, random_state=42)
    assert approximator.n == n_players

    # test representation
    representation = repr(approximator)
    assert "KernelSHAP" in representation
    assert str(approximator) == representation

    # test equality
    approximator_copy = copy(approximator)
    approximator_deepcopy = deepcopy(approximator)
    approximator_deepcopy.index = "something"
    assert approximator_copy == approximator  # check that the copy is equal
    assert approximator_deepcopy != approximator  # check that the deepcopy is not equal

    assert hash(approximator) == hash(approximator_copy)
    assert hash(approximator) != hash(approximator_deepcopy)
    with pytest.raises(TypeError):
        _ = approximator == 1

    id_approx = approximator.approximator_id
    assert hash(approximator) == id_approx

    # test if approximator can be called
    approximator(budget=4, game=game)


def test_abstract_approximator():
    """Tests if the attributes and properties of approximators are set correctly."""
    approx = get_concrete_class(Approximator)(n=7, max_order=2, index="SII", top_order=False)
    assert approx.n == 7
    assert approx.max_order == 2
    assert approx.index == "SII"
    assert approx.top_order is False

    with pytest.raises(NotImplementedError):
        approx.approximate(budget=100, game=lambda x: x)

    with pytest.raises(NotImplementedError):
        approx(game=lambda x: x, budget=100)

    wrong_index = "something"
    with pytest.raises(ValueError):
        _ = get_concrete_class(Approximator)(n=7, max_order=2, index=wrong_index, top_order=False)
