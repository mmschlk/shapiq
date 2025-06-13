"""This test script tests the RandomGame class from the synthetic module."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.games.benchmark import RandomGame


@pytest.mark.parametrize("n_players", [10, 100])
def test_random_game(n_players):
    """Test the RandomGame class with different numbers of players."""
    n_coalitions = 100

    game = RandomGame(n_players, random_state=None)
    coalitions = np.zeros((n_coalitions, n_players), dtype=bool)
    values = game(coalitions)
    assert values.shape == (n_coalitions,)
    assert np.all(values >= 0)
    assert np.all(values <= 100)
    assert not np.all(values == 0)
    assert not np.all(values == 100)

    values_second_time = game(coalitions)
    assert values_second_time.shape == (n_coalitions,)
    assert not np.all(values == values_second_time)

    # with random state
    game_one = RandomGame(n_players, random_state=1)
    game_two = RandomGame(n_players, random_state=2)

    values_one = game_one(coalitions)
    values_one_second_time = game_one(coalitions)
    values_two = game_two(coalitions)

    assert np.all(values_one == values_one_second_time)  # should be the same
    assert not np.all(values_one == values_two)
