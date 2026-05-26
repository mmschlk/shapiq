from leaderboard.runner.game_factory import create_game_from_config
from shapiq_games.synthetic import SOUM


def test_create_game_from_config():
    """Test that the game factory creates a SOUM game with default parameters."""
    run_config = {
        "game": "SOUM",
        "game_seed": 42,
        "max_order": 2,
    }

    base_config = {}

    game, game_params = create_game_from_config(
        run_config=run_config,
        base_config=base_config,
    )

    assert isinstance(game, SOUM)

    assert game_params == {
        "n": 10,
        "n_basis_games": 20,
        "min_interaction_size": 1,
        "max_interaction_size": 2,
        "random_state": 42,
    }

    assert game.n_players == 10