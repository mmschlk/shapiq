"""This test module contains the tests for the DummyGame class."""
import numpy as np
import pytest

from games import DummyGame


@pytest.mark.parametrize(
    "n, interaction, expected",
    [
        (
            3,
            (1, 2),
            {
                (): 0,
                (0,): 1 / 3,
                (1,): 1 / 3,
                (2,): 1 / 3,
                (0, 1): 2 / 3,
                (0, 2): 2 / 3,
                (1, 2): 2 / 3 + 1,
                (0, 1, 2): 3 / 3 + 1,
            },
        ),
        (
            4,
            (1, 2),
            {
                (): 0,
                (0,): 1 / 4,
                (1,): 1 / 4,
                (2,): 1 / 4,
                (3,): 1 / 4,
                (0, 1): 2 / 4,
                (1, 2): 2 / 4 + 1,
                (2, 3): 2 / 4,
                (0, 1, 2): 3 / 4 + 1,
                (1, 2, 3): 3 / 4 + 1,
                (0, 1, 2, 3): 4 / 4 + 1,
            },
        ),
    ],
)
def test_dummy_game(n, interaction, expected):
    """Test the DummyGame class."""
    game = DummyGame(n=n, interaction=interaction)
    for coalition in expected.keys():
        x_input = np.zeros(shape=(n,), dtype=bool)
        x_input[list(coalition)] = True
        assert game(x_input)[0] == expected[coalition]

    string_game = str(game)
    assert repr(game) == string_game


def test_dummy_game_access_counts():
    """Test how often the game was called."""
    game = DummyGame(n=10, interaction=(1, 2))
    assert game.access_counter == 0
    game(np.asarray([True, False, False, False, False, False, False, False, False, False]))
    assert game.access_counter == 1
    game(np.asarray([True, False, False, False, False, False, False, False, False, False]))
    assert game.access_counter == 2
    game(
        np.asarray(
            [
                [True, False, False, False, False, False, False, False, False, False],
                [False, True, False, False, False, False, False, False, False, False],
                [False, False, True, False, False, False, False, False, False, False],
            ]
        )
    )
    assert game.access_counter == 5
