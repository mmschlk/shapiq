"""This script is used to time and profile the regression estimations in large player domains."""

from shapiq.games.benchmark import DummyGame

if __name__ == "__main__":

    n_players = 100

    game = DummyGame(n=n_players, interaction=(1, 2))
