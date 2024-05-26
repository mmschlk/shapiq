"""This module contains the benchmark DatasetValuation games."""

from typing import Optional, Union

from .._setup import GameBenchmarkSetup
from .base import DatasetValuation


class CaliforniaHousing(DatasetValuation):
    """The California Housing dataset as a DatasetValuation game.

    Args:
        n_players: The number of players in the game. Defaults to 10.
        model_name: The model to use for the game. Must be 'decision_tree' or 'random_forest'.
            Defaults to 'decision_tree'.
        player_sizes: The sizes of the players. If 'uniform', the players have equal sizes. If
            'increasing', the players have increasing sizes. If 'random', the players have random
            sizes. If a list of floats, the players have the given sizes. Defaults to 'increasing'.
        random_state: The random state to use for shuffling the data. Defaults to 42.

    Note:
        This game uses models from the `sklearn` library. Install the library to use this game.

    Examples:
        >>> import numpy as np
        >>> from shapiq.games.benchmark.dataset_valuation import CaliforniaHousing
        >>> game = CaliforniaHousing(n_players=4)
        >>> game.n_players
        4
        >>> game.player_sizes
        [0.1, 0.2, 0.3, 0.4]
        >>> game_values = game(np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=bool))
        >>> len(game_values)
        3
        >>> # precompute the values and store them
        >>> game.precompute()
        >>> game.save_values("california_housing_values.npz")
        >>> # load the values from the file
        >>> from shapiq.games import Game  # for loading the game via its values
        >>> game = Game(path_to_values="california_housing_values.npz")
        >>> game.n_players
        4
    """

    def __init__(
        self,
        *,
        n_players: int = 10,
        model_name: str = "decision_tree",
        player_sizes: Optional[Union[list[float], str]] = "increasing",
        random_state: Optional[int] = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:

        setup = GameBenchmarkSetup(
            dataset_name="california_housing",
            model_name=model_name,
            random_state=random_state,
            verbose=verbose,
        )

        super().__init__(
            x_train=setup.x_train,
            y_train=setup.y_train,
            x_test=setup.x_test,
            y_test=setup.y_test,
            fit_function=setup.fit_function,
            predict_function=setup.predict_function,
            loss_function=setup.loss_function,
            n_players=n_players,
            player_sizes=player_sizes,
            random_state=random_state,
            empty_data_value=0.0,
            normalize=normalize,
            verbose=verbose,
        )


class BikeSharing(DatasetValuation):
    """The Bike Sharing dataset as a DatasetValuation game.

    Args:
        n_players: The number of players in the game. Defaults to 10.
        model_name: The model to use for the game. Must be 'decision_tree' or 'random_forest'.
            Defaults to 'decision_tree'.
        player_sizes: The sizes of the players. If 'uniform', the players have equal sizes. If
            'increasing', the players have increasing sizes. If 'random', the players have random
            sizes. If a list of floats, the players have the given sizes. Defaults to 'increasing'.
        random_state: The random state to use for shuffling the data. Defaults to 42.

    Note:
        This game uses models from the `sklearn` library. Install the library to use this game.

    Examples:
        >>> import numpy as np
        >>> from shapiq.games.benchmark.dataset_valuation import BikeSharing
        >>> game = BikeSharing(n_players=4)
        >>> game.n_players
        4
        >>> game.player_sizes
        [0.1, 0.2, 0.3, 0.4]
        >>> game_values = game(np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=bool))
        >>> len(game_values)
        3
        >>> # precompute the values and store them
        >>> game.precompute()
        >>> game.save_values("bike_sharing_values.npz")
        >>> # load the values from the file
        >>> from shapiq.games import Game  # for loading the game via its values
        >>> game = Game(path_to_values="bike_sharing_values.npz")
        >>> game.n_players
        4
    """

    def __init__(
        self,
        *,
        n_players: int = 10,
        model_name: str = "decision_tree",
        player_sizes: Optional[Union[list[float], str]] = "increasing",
        random_state: Optional[int] = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:

        setup = GameBenchmarkSetup(
            dataset_name="bike_sharing",
            model_name=model_name,
            random_state=random_state,
            verbose=verbose,
        )

        super().__init__(
            x_train=setup.x_train,
            y_train=setup.y_train,
            x_test=setup.x_test,
            y_test=setup.y_test,
            fit_function=setup.fit_function,
            predict_function=setup.predict_function,
            loss_function=setup.loss_function,
            n_players=n_players,
            player_sizes=player_sizes,
            random_state=random_state,
            empty_data_value=0.0,
            normalize=normalize,
            verbose=verbose,
        )


class AdultCensus(DatasetValuation):
    """The Adult Census dataset as a DatasetValuation game.

    Args:
        n_players: The number of players in the game. Defaults to 10.
        model_name: The model to use for the game. Must be 'decision_tree' or 'random_forest'.
            Defaults to 'decision_tree'.
        player_sizes: The sizes of the players. If 'uniform', the players have equal sizes. If
            'increasing', the players have increasing sizes. If 'random', the players have random
            sizes. If a list of floats, the players have the given sizes. Defaults to 'increasing'.
        random_state: The random state to use for shuffling the data. Defaults to 42.

    Note:
        This game uses models from the `sklearn` library. Install the library to use this game.

    Examples:
        >>> import numpy as np
        >>> from shapiq.games.benchmark.dataset_valuation import AdultCensus
        >>> game = AdultCensus(n_players=4)
        >>> game.n_players
        4
        >>> game.player_sizes
        [0.1, 0.2, 0.3, 0.4]
        >>> game_values = game(np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=bool))
        >>> len(game_values)
        3
        >>> # precompute the values and store them
        >>> game.precompute()
        >>> game.save_values("adult_census_values.npz")
        >>> # load the values from the file
        >>> from shapiq.games import Game  # for loading the game via its values
        >>> game = Game(path_to_values="adult_census_values.npz")
        >>> game.n_players
        4
    """

    def __init__(
        self,
        *,
        n_players: int = 10,
        model_name: str = "decision_tree",
        player_sizes: Optional[Union[list[float], str]] = "increasing",
        random_state: Optional[int] = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:

        setup = GameBenchmarkSetup(
            dataset_name="adult_census",
            model_name=model_name,
            random_state=random_state,
            verbose=verbose,
        )

        super().__init__(
            x_train=setup.x_train,
            y_train=setup.y_train,
            x_test=setup.x_test,
            y_test=setup.y_test,
            fit_function=setup.fit_function,
            predict_function=setup.predict_function,
            loss_function=setup.loss_function,
            n_players=n_players,
            player_sizes=player_sizes,
            random_state=random_state,
            empty_data_value=0.0,
            normalize=normalize,
            verbose=verbose,
        )
