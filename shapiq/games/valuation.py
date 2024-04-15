"""This module contains the data valuation games for the shapiq benchmark."""

from typing import Callable, Optional, Union

import numpy as np

from shapiq.games.base import Game


class DatasetValuation(Game):
    """The basis Dataset Valuation Game class.

    The Dataset Valuation Game consists of valuating the worth of individual subsets/chunks of
    datasets towards the whole dataset's performance. Therein, the players are individual subsets
    of rows of the dataset, and the worth of a coalition is the performance of a model on a seperate
    holdout set, trained on the union of the players' subsets.

    Args:
        path_to_values: The path to load the game values from. If the path is provided, the game
            values are loaded from the given path. Defaults to `None`.


    Examples:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from shapiq.datasets import load_bike
        >>> from shapiq.games.valuation import DatasetValuation
        >>> data, target = load_bike()
        >>> model = RandomForestRegressor()

    """

    def __init__(
        self,
        path_to_values: str = None,
        x_train: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        y_train: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        fit_function: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
        predict_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        loss_function: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        n_players: int = 10,
        player_sizes: Optional[Union[list[float], str]] = "uniform",
        random_state: Optional[int] = None,
        empty_value: float = 0.0,
        normalize: bool = True,
    ) -> None:

        # path to values
        if path_to_values is not None:
            super().__init__(path_to_values=path_to_values, normalize=normalize)
            return

        # check if all required functions are given, otherwise
        if (
            x_train is None
            or y_train is None
            or fit_function is None
            or loss_function is None
            or predict_function is None
        ):
            raise ValueError(
                "The 'data', 'target', 'fit_function', 'predict_function', and 'loss_function' "
                "must be provided."
            )

        if isinstance(x_train, list):
            n_players = len(x_train)

        if isinstance(player_sizes, str):
            if player_sizes == "uniform":
                player_sizes = [1 / n_players for _ in range(n_players)]
            elif player_sizes == "increasing":
                player_sizes = [i / n_players for i in range(1, n_players + 1)]
            elif player_sizes == "random":
                player_sizes = np.random.rand(n_players)
            else:
                raise ValueError(
                    "player_sizes must be 'uniform', 'increasing', 'random', or a list."
                )
        else:
            if player_sizes is None:
                player_sizes = [1 / n_players for _ in range(n_players)]
        player_sizes = np.array(player_sizes) / np.sum(player_sizes)

        if random_state is not None:
            np.random.seed(random_state)
        rng = np.random.default_rng(random_state)

        # get the holdout set (if not provided)
        if x_test is None or y_test is None:
            if isinstance(x_train, list):
                raise ValueError("x_test and y_test must be provided if x_train is a list.")
            # randomly split the data into training and test set
            idx = rng.permutation(np.arange(x_train.shape[0]))
            x_train, y_train = x_train[idx], y_train[idx]
            n_holdout = int(test_size * x_train.shape[0])
            x_test, y_test = x_train[:n_holdout], y_train[:n_holdout]
            x_train, y_train = x_train[n_holdout:], y_train[n_holdout:]

        # check if data is a sequence of arrays or a single array
        if isinstance(x_train, list):
            n_players = len(x_train)
            data_sets = {i: x_train[i] for i in range(n_players)}
            target_sets = {i: y_train[i] for i in range(n_players)}
            player_sizes = np.asarray([x_train[i].shape[0] for i in range(n_players)])
            player_sizes = player_sizes / np.sum(player_sizes)
        else:  # data is assumed to be a single array
            n_players = len(player_sizes)
            # shuffle the data and target
            idx = rng.permutation(np.arange(x_train.shape[0]))
            x_train, y_train = x_train[idx], y_train[idx]
            data_sets, target_sets = {}, {}
            start = 0
            for i in range(n_players):
                end = start + int(player_sizes[i] * x_train.shape[0])
                if i >= n_players - 1:
                    end = x_train.shape[0]
                data_sets[i] = x_train[start:end]
                target_sets[i] = y_train[start:end]
                start = end

        self.data_sets = data_sets
        self.target_sets = target_sets
        self.player_sizes = player_sizes

        self._x_test, self._y_test = x_test, y_test
        self._fit_function = fit_function
        self._predict_function = predict_function
        self._loss_function = loss_function

        self.empty_value = empty_value

        super().__init__(n_players=n_players, normalize=normalize, normalization_value=empty_value)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Trains the model on the data subsets denoted in the coalitions. The worth of the
        coalition is the performance of the model on the holdout set.

        Args:
            coalitions: The coalition as a binary matrix of shape `(n_coalitions, n_players)`.

        Returns:
            The worth of the coalition.
        """
        worth = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            if np.sum(coalition) == 0:
                worth[i] = self.empty_value
                continue
            # create the training data for the coalition
            x_train = np.concatenate([self.data_sets[j] for j in np.where(coalition)[0]])
            y_train = np.concatenate([self.target_sets[j] for j in np.where(coalition)[0]])
            self._fit_function(x_train, y_train)
            y_pred = self._predict_function(self._x_test)
            worth[i] = self._loss_function(self._y_test, y_pred)
        return worth
