"""This module contains the meta class for all FeatureSelection benchmark games."""

from typing import Any, Callable, Optional

import numpy as np

from shapiq.games.base import Game


class FeatureSelectionGame(Game):
    """The FeatureSelection game.

    The `FeatureSelectionGame` is a game that evaluates the goodness of fit of a model on a subset
    of features. The goodness of fit is determined by a score or loss function that compares the
    model's test set performance.

    Args:
        path_to_values: The path to the pre-computed game values to load. If provided, then the game
            is loaded from the file and no other parameters are used. Defaults to `None`.
        x_train: The training data used to fit the model. Should be a 2d matrix of shape
            (n_samples, n_features). Defaults to `None` but must be provided if `path_to_values` is
            `None`.
        y_train: The training labels used to fit the model. Can be a 1d or 2d matrix of shape
            (n_samples, n_outputs). Defaults to `None` but must be provided if `path_to_values` is
            `None`.
        x_test: The test data used to evaluate the model. Should be the same shape as `x_train`.
            Defaults to `None` but must be provided if `path_to_values` is `None`.
        y_test: The test labels used to evaluate the model. Should be the same shape as `y_train`.
            Defaults to `None` but must be provided if `path_to_values` is `None`.
        fit_function: The function that fits the model to the training data. It should take the
            training data and labels as input. Defaults to `None` but must be provided if
            `path_to_values` is `None`.
        score_function: The function that scores the model's performance on the test data. It should
            take the test data and labels as input. If not provided, then `predict_function` and
            `loss_function` must be provided (if `path_to_values` is `None`).
        predict_function: The function that predicts the test labels given the test data. It should
            take the test data as input. If not provided, then `score_function` must be provided (if
            `path_to_values` is `None`).
        loss_function: The function that computes the loss between the predicted and true test
            labels. It should take the true and predicted test labels as input. If not provided,
            then `score_function` must be provided (if `path_to_values` is `None`).
        empty_value: The value to return when the subset of features is empty. Defaults to 0.0.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.

    Attributes:
        empty_value: The value to return when the subset of features is empty.

    Examples:
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> from shapiq.games.tabular import FeatureSelectionGame
        >>> # create a regression dataset
        >>> x_data, y_data = make_regression(n_samples=100, n_features=10, noise=0.1)
        >>> x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
        >>> # create a decision tree regressor
        >>> model = DecisionTreeRegressor(max_depth=4)
        >>> # create a FeatureSelection game
        >>> game = FeatureSelectionGame(
        ...     x_train=x_train,
        ...     x_test=x_test,
        ...     y_train=y_train,
        ...     y_test=y_test,
        ...     fit_function=model.fit,
        ...     score_function=model.score,
        ... )
        >>> # evaluate the game on a specific coalition
        >>> coalition = np.zeros(shape=(1, 10), dtype=bool)
        >>> coalition[0][0, 1, 2] = True
        >>> value = game(coalition)
        >>> # precompute the game (if needed)
        >>> game.precompute()
        >>> # save and load the game
        >>> game.save("game.pkl")
        >>> new_game = FeatureSelectionGame.load("game.pkl")
    """

    def __init__(
        self,
        *,
        path_to_values: Optional[str] = None,
        x_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        fit_function: Optional[Callable[[np.ndarray, np.ndarray], Any]] = None,
        score_function: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        predict_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        loss_function: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        empty_value: float = 0.0,
        normalize: bool = True,
    ) -> None:
        # TODO: remove path to values logic from subclass
        # check if path is provided
        if path_to_values is not None:
            super().__init__(path_to_values=path_to_values)
            return

        super().__init__(x_train.shape[1], normalization_value=empty_value, normalize=normalize)

        # set datasets
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

        # sanity check on  input params
        if score_function is None:
            if loss_function is None or predict_function is None:
                raise ValueError(
                    "If score function is not provided, then 'predict_function' and 'loss_function'"
                    " must be provided."
                )

        # setup callables
        self._fit_function = fit_function
        self._predict_function = predict_function
        self._loss_function = loss_function
        self._score_function = score_function

        # set empty value
        self.empty_value = empty_value

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Trains and evaluates the model on a coalition (subset) of features. The output of the
            value function is the value of the specified loss function (goodness of fit metric).

        Args:
            coalitions: A one-hot 2d matrix of coalitions denoting the feature selection to train
                and evaluate the model on.

        Returns:
            A vector of loss function values given the subset of features.
        """
        scores = np.zeros(shape=coalitions.shape[0], dtype=float)
        for i in range(len(coalitions)):
            coalition = coalitions[i]  # get coalition
            if sum(coalition) == 0:  # if empty subset then set to empty prediction
                scores[i] = self.empty_value
                continue
            x_train, x_test = self._x_train[:, coalition], self._x_test[:, coalition]
            self._fit_function(x_train, self._y_train)  # fit model
            if self._score_function is not None:
                score = self._score_function(x_test, self._y_test)
            else:
                y_pred = self._predict_function(x_test)  # get y hat prediction
                score = self._loss_function(self._y_test, y_pred)  # compare prediction with gt
            scores[i] = score
        return scores
