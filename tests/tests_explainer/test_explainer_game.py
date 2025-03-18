"""This test module tests the GameExplainer class."""

from shapiq.explainer.game import GameExplainer
from shapiq.games.benchmark import DummyGame


def test_init():
    """Test the initialization of the GameExplainer."""
    n_players = 7
    game = DummyGame(n=n_players, interaction=(1, 2))

    explainer = GameExplainer(game=game)
    assert explainer.game == game
    assert explainer.n_players == n_players
