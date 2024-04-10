from games.soum import SOUM, UnanimityGame
import numpy as np


def test_soum_interations():
    """Test SOUM interactions."""
    n = 10
    M = 100
    interaction = np.random.randint(2, size=n)

    # Unanimity Test
    u_game = UnanimityGame(interaction)
    coalition_matrix = np.random.randint(2, size=(M, n))
    u_game_values = u_game(coalition_matrix)
    assert len(u_game_values) == M
    assert u_game.n == n
    assert np.sum(u_game.interaction_binary) == len(u_game.interaction)
    assert np.sum(interaction) == len(u_game.interaction)

    # SOUM Test
    n_interactions = 150

    soum = SOUM(n, n_interactions)
    soum_values = soum(coalition_matrix)
    assert len(soum_values) == M
    assert len(soum.linear_coefficients) == n_interactions
    assert len(soum.unanimity_games) == n_interactions
    assert soum.max_interaction_size == n
    assert soum.min_interaction_size == 0

    min_interaction_size = 5
    soum_2 = SOUM(n, n_interactions, min_interaction_size=min_interaction_size)
    soum_2_values = soum_2(coalition_matrix)
    assert len(soum_2_values) == M
    assert len(soum_2.linear_coefficients) == n_interactions
    assert len(soum_2.unanimity_games) == n_interactions
    for i, base_game in soum_2.unanimity_games.items():
        assert len(base_game.interaction) >= min_interaction_size
    assert soum_2.max_interaction_size == n
    assert soum_2.min_interaction_size == min_interaction_size

    max_interaction_size = 7
    soum_3 = SOUM(n, n_interactions, max_interaction_size=max_interaction_size)
    soum_3_values = soum_3(coalition_matrix)
    assert len(soum_3_values) == M
    assert len(soum_3.linear_coefficients) == n_interactions
    assert len(soum_3.unanimity_games) == n_interactions
    for i, base_game in soum_3.unanimity_games.items():
        assert len(base_game.interaction) <= max_interaction_size
    assert soum_3.max_interaction_size == max_interaction_size
    assert soum_3.min_interaction_size == 0

    min_interaction_size = 3
    max_interaction_size = 7
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
