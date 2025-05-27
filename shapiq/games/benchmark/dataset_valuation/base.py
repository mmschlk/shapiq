"""This module contains the data valuation games for the shapiq benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.games.base import Game

if TYPE_CHECKING:
    from collections.abc import Callable


class DatasetValuation(Game):
    """The basis Dataset Valuation Game class.

    The Dataset Valuation Game consists of valuating the worth of individual subsets/chunks of
    datasets towards the whole dataset's performance. Therein, the players are individual subsets
    of rows of the dataset, and the worth of a coalition is the performance of a model on a separate
    holdout set, trained on the union of the players' subsets. This game is presented in the
    paper by Garrido-Lucero et al. (2024) [1]_.

    References:
          .. [1] Garrido-Lucero, F., Heymann, B., Vono, M., Loiseau, P., Perchet, V. (2024). Advances in Neural Information Processing Systems 37 (NeurIPS 2024) https://proceedings.neurips.cc/paper_files/paper/2024/hash/03cd3cf3f74d4f9ce5958de269960884-Abstract-Conference.html

    """

    def __init__(
        self,
        *,
        x_train: np.ndarray | list[np.ndarray],
        y_train: np.ndarray | list[np.ndarray],
        x_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        test_size: float = 0.2,
        fit_function: Callable[[np.ndarray, np.ndarray], None] | None = None,
        predict_function: Callable[[np.ndarray], np.ndarray] | None = None,
        loss_function: Callable[[np.ndarray, np.ndarray], float] | None = None,
        n_players: int = 10,
        player_sizes: list[float] | str | None = "uniform",
        random_state: int | None = 42,
        normalize: bool = True,
        verbose: bool = False,
        empty_data_value: float = 0.0,
    ) -> None:
        """Initialize the DatasetValuation game.

        Args:
            x_train: The training data used to fit the model.

            y_train: The training labels used to fit the model.

            x_test: The test data used to evaluate the model.

            y_test: The test labels used to evaluate the model.

            test_size: The size of the validation set to be taken from x_train if x_test is missing.
                Defaults to ``0.2``.

            fit_function: The function that fits the model to the training data as a callable
                expecting the training data and labels as input in form of numpy arrays.

            predict_function: The function that predicts the test labels given the test data as a
                callable expecting the test data as input in form of numpy arrays.

            loss_function: A sensible loss function that computes the loss between the predicted and
                true test labels as a callable expecting the true and predicted test labels as input in
                form of numpy arrays.

            n_players: The number of players in the game, i.e. data subsets. Defaults to ``10`` and
                interacts with ``player_sizes``.

            player_sizes: Size of players, i.e. data subsets. Either a list of floats or a string
                indicating the splitting strategy. Can be one of ``{'uniform', 'increasing',
                'random'}``. Defaults to uniform and interacts with ``n_players``.

            random_state: The random state to use for all random operations. Defaults to ``42``.

            normalize: Whether the game values should be normalized. Defaults to ``True``.

            verbose: Whether to print information about the game. Defaults to ``False``.

            empty_data_value: The worth of an empty subset of data. Defaults to ``0.0``.

        """
        # check if all required functions are given, otherwise
        if (
            x_train is None
            or y_train is None
            or fit_function is None
            or loss_function is None
            or predict_function is None
        ):
            msg = (
                "The 'data', 'target', 'fit_function', 'predict_function', and 'loss_function' "
                "must be provided."
            )
            raise ValueError(msg)

        rng = np.random.default_rng(random_state)

        if isinstance(player_sizes, str):
            if player_sizes == "uniform":
                player_sizes = [1 / n_players for _ in range(n_players)]
            elif player_sizes == "increasing":
                player_sizes = [i / n_players for i in range(1, n_players + 1)]
            elif player_sizes == "random":
                player_sizes = rng.random(n_players)
            else:
                msg = "player_sizes must be 'uniform', 'increasing', 'random', or a list."
                raise ValueError(
                    msg,
                )
        elif player_sizes is None:
            player_sizes = [1 / n_players for _ in range(n_players)]
        player_sizes = np.array(player_sizes) / np.sum(player_sizes)

        # get the holdout set (if not provided)
        if x_test is None or y_test is None:
            if isinstance(x_train, list):
                msg = "x_test and y_test must be provided if x_train is a list."
                raise ValueError(msg)
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

        self.empty_data_value = empty_data_value

        super().__init__(
            n_players=n_players,
            normalize=normalize,
            normalization_value=self.empty_data_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Trains the model on the data subsets denoted in the coalitions.

        The worth of the coalition is the performance of the model on the holdout set.

        Args:
            coalitions: The coalition as a binary matrix of shape `(n_coalitions, n_players)`.

        Returns:
            The worth of the coalition.

        """
        worth = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            if np.sum(coalition) == 0:
                worth[i] = self.empty_data_value
                continue
            # create the training data for the coalition
            x_train = np.concatenate([self.data_sets[j] for j in np.where(coalition)[0]])
            y_train = np.concatenate([self.target_sets[j] for j in np.where(coalition)[0]])
            self._fit_function(x_train, y_train)
            y_pred = self._predict_function(self._x_test)
            worth[i] = self._loss_function(self._y_test, y_pred)
        return worth
