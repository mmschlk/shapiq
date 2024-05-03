"""This test module tests the SOUM Game class"""

import pytest
import numpy as np

from shapiq.games.benchmark import DummyGame, SOUM, UnanimityGame


def test_soum_interations():
    """Test SOUM interactions."""

    for i in range(10):
        # run 100 times
        n = np.random.randint(low=2, high=30)
        M = np.random.randint(low=1, high=150)
        interaction = np.random.randint(2, size=n)

        # Unanimity Test
        u_game = UnanimityGame(interaction)
        assert u_game.game_name == "UnanimityGame_Game"
        coalition_matrix = np.random.randint(2, size=(M, n))
        u_game_values = u_game(coalition_matrix)
        assert len(u_game_values) == M
        assert u_game.n_players == n
        assert np.sum(u_game.interaction_binary) == len(u_game.interaction)
        assert np.sum(interaction) == len(u_game.interaction)

        # SOUM Test
        n_interactions = np.random.randint(low=1, high=200)

        soum = SOUM(n, n_interactions)
        assert soum.game_name == "SOUM_Game"
        soum_values = soum(coalition_matrix)
        assert len(soum_values) == M
        assert len(soum.linear_coefficients) == n_interactions
        assert len(soum.unanimity_games) == n_interactions
        assert soum.max_interaction_size == n
        assert soum.min_interaction_size == 0

        min_interaction_size = np.random.randint(low=1, high=n)

        soum_2 = SOUM(n, n_interactions, min_interaction_size=min_interaction_size)
        soum_2_values = soum_2(coalition_matrix)
        assert len(soum_2_values) == M
        assert len(soum_2.linear_coefficients) == n_interactions
        assert len(soum_2.unanimity_games) == n_interactions
        for i, base_game in soum_2.unanimity_games.items():
            assert len(base_game.interaction) >= min_interaction_size
        assert soum_2.max_interaction_size == n
        assert soum_2.min_interaction_size == min_interaction_size

        max_interaction_size = np.random.randint(low=1, high=n)

        soum_3 = SOUM(n, n_interactions, max_interaction_size=max_interaction_size)
        soum_3_values = soum_3(coalition_matrix)
        assert len(soum_3_values) == M
        assert len(soum_3.linear_coefficients) == n_interactions
        assert len(soum_3.unanimity_games) == n_interactions
        for i, base_game in soum_3.unanimity_games.items():
            assert len(base_game.interaction) <= max_interaction_size
        assert soum_3.max_interaction_size == max_interaction_size
        assert soum_3.min_interaction_size == 0

        min_interaction_size = np.random.randint(low=0, high=n - 1)
        max_interaction_size = np.random.randint(low=min_interaction_size + 1, high=n)

        soum_4 = SOUM(
            n,
            n_interactions,
            min_interaction_size=min_interaction_size,
            max_interaction_size=max_interaction_size,
        )
        soum_4_values = soum_4(coalition_matrix)
        assert len(soum_4_values) == M
        assert len(soum_4.linear_coefficients) == n_interactions
        assert len(soum_4.unanimity_games) == n_interactions
        for i, base_game in soum_4.unanimity_games.items():
            assert (len(base_game.interaction) <= max_interaction_size) and (
                len(base_game.interaction) >= min_interaction_size
            )
        assert soum_4.max_interaction_size == max_interaction_size
        assert soum_4.min_interaction_size == min_interaction_size


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
    assert game.game_name == "DummyGame_Game"


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
