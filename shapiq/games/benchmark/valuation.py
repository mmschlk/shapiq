"""This module contains the benchmark DatasetValuation games."""

from typing import Optional, Union

from shapiq.games.valuation import DatasetValuation
from shapiq.utils.datasets import shuffle_data


def _get_decision_tree_regressor():
    """Get the decision tree regressor model and functions."""
    from sklearn.metrics import r2_score
    from sklearn.tree import DecisionTreeRegressor

    model = DecisionTreeRegressor()

    fit_function = model.fit
    predict_function = model.predict
    loss_function = r2_score
    return fit_function, predict_function, loss_function


def _get_random_forest_regressor():
    """Get the random forest regressor model and functions."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    model = RandomForestRegressor()

    fit_function = model.fit
    predict_function = model.predict
    loss_function = r2_score
    return fit_function, predict_function, loss_function


def _get_decision_tree_classifier():
    """Get the decision tree classifier model and functions."""
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier()

    fit_function = model.fit
    predict_function = model.predict
    loss_function = accuracy_score
    return fit_function, predict_function, loss_function


def _get_random_forest_classifier():
    """Get the random forest classifier model and functions."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    model = RandomForestClassifier()

    fit_function = model.fit
    predict_function = model.predict
    loss_function = accuracy_score
    return fit_function, predict_function, loss_function


class CaliforniaHousing(DatasetValuation):
    """The California Housing dataset as a DatasetValuation game.

    Args:
        path_to_values: The path to load the game values from. If the path is provided, the game
            values are loaded from the given path. Defaults to `None`.
        n_players: The number of players in the game. Defaults to 10.
        model: The model to use for the game. Must be 'decision_tree' or 'random_forest'.
            Defaults to 'decision_tree'.
        player_sizes: The sizes of the players. If 'uniform', the players have equal sizes. If
            'increasing', the players have increasing sizes. If 'random', the players have random
            sizes. If a list of floats, the players have the given sizes. Defaults to 'increasing'.
        random_state: The random state to use for shuffling the data. Defaults to `None`.

    Examples:
        >>> import numpy as np
        >>> from shapiq.games.benchmark.valuation import CaliforniaHousing
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
        >>> game = CaliforniaHousing(path_to_values="california_housing_values.npz")
        >>> game.n_players
        4
    """

    def __init__(
        self,
        path_to_values: str = None,
        n_players: int = 10,
        model: str = "decision_tree",
        player_sizes: Optional[Union[list[float], str]] = "increasing",
        random_state: Optional[int] = None,
    ) -> None:

        if path_to_values is not None:
            super().__init__(path_to_values=path_to_values)
            return

        if model == "decision_tree":
            fit_function, predict_function, loss_function = _get_decision_tree_regressor()
        elif model == "random_forest":
            fit_function, predict_function, loss_function = _get_random_forest_regressor()
        else:
            raise ValueError("Model must be 'decision_tree' or 'random_forest'.")

        from shapiq.datasets import load_california_housing

        x_train, y_train = load_california_housing()
        self.feature_names = list(x_train.columns)
        x_train, y_train = shuffle_data(x_train.values, y_train.values, random_state=random_state)

        super().__init__(
            x_train=x_train,
            y_train=y_train,
            test_size=0.2,
            fit_function=fit_function,
            predict_function=predict_function,
            loss_function=loss_function,
            n_players=n_players,
            player_sizes=player_sizes,
            random_state=random_state,
            empty_value=0.0,
        )


class BikeSharing(DatasetValuation):
    """The Bike Sharing dataset as a DatasetValuation game.

    Args:
        path_to_values: The path to load the game values from. If the path is provided, the game
            values are loaded from the given path. Defaults to `None`.
        n_players: The number of players in the game. Defaults to 10.
        model: The model to use for the game. Must be 'decision_tree' or 'random_forest'.
            Defaults to 'decision_tree'.
        player_sizes: The sizes of the players. If 'uniform', the players have equal sizes. If
            'increasing', the players have increasing sizes. If 'random', the players have random
            sizes. If a list of floats, the players have the given sizes. Defaults to 'increasing'.
        random_state: The random state to use for shuffling the data. Defaults to `None`.

    Examples:
        >>> import numpy as np
        >>> from shapiq.games.benchmark.valuation import BikeSharing
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
        >>> game = BikeSharing(path_to_values="bike_sharing_values.npz")
        >>> game.n_players
        4
    """

    def __init__(
        self,
        path_to_values: str = None,
        n_players: int = 10,
        model: str = "decision_tree",
        player_sizes: Optional[Union[list[float], str]] = "increasing",
        random_state: Optional[int] = None,
    ) -> None:

        if path_to_values is not None:
            super().__init__(path_to_values=path_to_values)
            return

        if model == "decision_tree":
            fit_function, predict_function, loss_function = _get_decision_tree_regressor()
        elif model == "random_forest":
            fit_function, predict_function, loss_function = _get_random_forest_regressor()
        else:
            raise ValueError("Model must be 'decision_tree' or 'random_forest'.")

        from shapiq.datasets import load_bike

        x_train, y_train = load_bike()
        x_train, y_train = shuffle_data(x_train.values, y_train.values, random_state=random_state)

        super().__init__(
            x_train=x_train,
            y_train=y_train,
            test_size=0.2,
            fit_function=fit_function,
            predict_function=predict_function,
            loss_function=loss_function,
            n_players=n_players,
            player_sizes=player_sizes,
            random_state=random_state,
            empty_value=0.0,
        )


class AdultCensus(DatasetValuation):
    """The Adult Census dataset as a DatasetValuation game.

    Args:
        path_to_values: The path to load the game values from. If the path is provided, the game
            values are loaded from the given path. Defaults to `None`.
        n_players: The number of players in the game. Defaults to 10.
        model: The model to use for the game. Must be 'decision_tree' or 'random_forest'.
            Defaults to 'decision_tree'.
        player_sizes: The sizes of the players. If 'uniform', the players have equal sizes. If
            'increasing', the players have increasing sizes. If 'random', the players have random
            sizes. If a list of floats, the players have the given sizes. Defaults to 'increasing'.
        random_state: The random state to use for shuffling the data. Defaults to `None`.

    Examples:
        >>> import numpy as np
        >>> from shapiq.games.benchmark.valuation import AdultCensus
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
        >>> game = AdultCensus(path_to_values="adult_census_values.npz")
        >>> game.n_players
        4
    """

    def __init__(
        self,
        path_to_values: str = None,
        n_players: int = 10,
        model: str = "decision_tree",
        player_sizes: Optional[Union[list[float], str]] = "increasing",
        random_state: Optional[int] = None,
    ) -> None:

        if path_to_values is not None:
            super().__init__(path_to_values=path_to_values)
            return

        if model == "decision_tree":
            fit_function, predict_function, loss_function = _get_decision_tree_classifier()
        elif model == "random_forest":
            fit_function, predict_function, loss_function = _get_random_forest_classifier()
        else:
            raise ValueError("Model must be 'decision_tree' or 'random_forest'.")

        from shapiq.datasets import load_adult_census

        x_train, y_train = load_adult_census()
        x_train, y_train = shuffle_data(x_train.values, y_train.values, random_state=random_state)

        super().__init__(
            x_train=x_train,
            y_train=y_train,
            test_size=0.2,
            fit_function=fit_function,
            predict_function=predict_function,
            loss_function=loss_function,
            n_players=n_players,
            player_sizes=player_sizes,
            random_state=random_state,
            empty_value=0.0,
        )
