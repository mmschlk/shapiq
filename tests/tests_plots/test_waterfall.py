"""This module contains all tests for the waterfall plots."""

import matplotlib.pyplot as plt
import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.plot import waterfall_plot


def test_waterfall_concrete():
    import numpy as np

    import shapiq

    class CookingGame(shapiq.Game):
        def __init__(self):
            self.characteristic_function = {
                (): 10,
                (0,): 4,
                (1,): 3,
                (2,): 2,
                (0, 1): 9,
                (0, 2): 8,
                (1, 2): 7,
                (0, 1, 2): 15,
            }
            super().__init__(
                n_players=3,
                player_names=["Alice", "Bob", "Charlie"],  # Optional list of names
                normalization_value=self.characteristic_function[()],  # 0
                normalize=False,
            )

        def value_function(self, coalitions: np.ndarray) -> np.ndarray:
            """Defines the worth of a coalition as a lookup in the characteristic function."""
            output = []
            for coalition in coalitions:
                output.append(self.characteristic_function[tuple(np.where(coalition)[0])])
            return np.array(output)

    cooking_game = CookingGame()

    from shapiq import ExactComputer

    # create an ExactComputer object for the cooking game
    exact_computer = ExactComputer(n_players=cooking_game.n_players, game_fun=cooking_game)

    # compute the Shapley Values for the game
    sv_exact = exact_computer(index="k-SII", order=2)
    print(sv_exact.dict_values)

    # visualize the Shapley Values
    from shapiq.plot import waterfall_plot

    waterfall_plot(sv_exact, show=True)


def test_waterfall_plot(interaction_values_list: list[InteractionValues]):
    """Test the waterfall plot function."""
    iv = interaction_values_list[0]
    n_players = iv.n_players
    feature_names = [f"feature-{i}" for i in range(n_players)]
    feature_names = np.array(feature_names)

    wp = waterfall_plot(iv, show=False)
    assert wp is not None
    assert isinstance(wp, plt.Axes)
    plt.close()

    wp = waterfall_plot(iv, show=False, feature_names=feature_names)
    assert isinstance(wp, plt.Axes)
    plt.close()

    iv = iv.get_n_order(1)
    wp = waterfall_plot(iv, show=False)
    assert isinstance(wp, plt.Axes)
    plt.close()

    wp = iv.plot_waterfall(show=False)
    assert isinstance(wp, plt.Axes)
    plt.close()

    # test show=True
    output = iv.plot_waterfall(show=True)
    assert output is None
    plt.close("all")
