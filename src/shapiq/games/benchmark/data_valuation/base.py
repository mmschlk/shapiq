"""This module contains the data valuation games for the shapiq benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.games.base import Game

if TYPE_CHECKING:
    from collections.abc import Callable


class DataValuation(Game):
    """The basis Data Valuation Game class.

    The Data Valuation Game consists of valuating the worth of individual data points towards the
    whole dataset's performance. Therein, the players are individual data points of the dataset, and
    the worth of a coalition is the performance of a model trained only on the data points in the
    coalition. A good summary is presented in [1]_.

    Attributes:
        x_train: The training data used to fit the model.
        y_train: The training labels used to fit the model.
        x_test: The test data used to evaluate the model.
        y_test: The test labels used to evaluate the model.
        empty_data_value: The worth of an empty subset of data.

    References:
        .. [1] Jiang, K., Liang, W., Zou, J. Y (2023) OpenDataVal: a Unified Benchmark for Data Valuation. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2023/hash/5b047c7d862059a5df623c1ce2982fca-Abstract-Datasets_and_Benchmarks.html
    """

    def __init__(
        self,
        *,
        n_data_points: int,
        x_data: np.ndarray,
        y_data: np.ndarray,
        fit_function: Callable[[np.ndarray, np.ndarray], None],
        predict_function: Callable[[np.ndarray], np.ndarray],
        loss_function: Callable[[np.ndarray, np.ndarray], float],
        random_state: int | None = 42,
        normalize: bool = True,
        verbose: bool = False,
        empty_data_value: float = 0.0,
    ) -> None:
        """Initialize the Data Valuation game.

        Args:
            n_data_points: The number of data points to sample from the data.

            x_data: The data used to fit the model.

            y_data: The labels used to fit the model.

            fit_function: The function that fits the model to the training data as a callable
                expecting the training data and labels as input in form of numpy arrays.

            predict_function: The function that predicts the test labels given the test data as a
                callable expecting the test data as input in form of numpy arrays.

            loss_function: A sensible loss function that computes the loss between the predicted and
                true test labels as a callable expecting the true and predicted test labels as input
                in form of numpy arrays.

            random_state: The random state to use for all random operations. Defaults to ``42``.

            normalize: Whether the game values should be normalized. Defaults to ``True``.

            verbose: Whether to print verbose output. Defaults to ``False``.

            empty_data_value: The worth of an empty subset of data. Defaults to ``0.0``.

        """
        # set the random state
        rng = np.random.default_rng(random_state)

        # randomly sample n_data_points from the data and create training and test data
        idx = rng.permutation(np.arange(x_data.shape[0]))
        self.x_train = x_data[idx[:n_data_points]]
        self.y_train = y_data[idx[:n_data_points]]
        self.x_test = x_data[idx[n_data_points:]]
        self.y_test = y_data[idx[n_data_points:]]

        # store the functions
        self._fit_function = fit_function
        self._predict_function = predict_function
        self._loss_function = loss_function

        # initialize the game
        self.empty_data_value = empty_data_value
        super().__init__(
            n_players=n_data_points,
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
            ids = np.where(coalition == 1)[0]
            x_train = self.x_train[ids]
            y_train = self.y_train[ids]
            self._fit_function(x_train, y_train)
            y_pred = self._predict_function(self.x_test)
            worth[i] = self._loss_function(self.y_test, y_pred)
        return worth
