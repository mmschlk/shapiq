"""This module contains the metaclass for all FeatureSelection benchmark games."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.games.base import Game

if TYPE_CHECKING:
    from collections.abc import Callable


class FeatureSelection(Game):
    """The FeatureSelection game.

    The `FeatureSelectionGame` is a game that evaluates the goodness of fit of a model on a subset
    of features. The goodness of fit is determined by a score or loss function that compares the
    model's test set performance.

    Attributes:
        empty_features_value: The value to return when the subset of features is empty.

    """

    def __init__(
        self,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        fit_function: Callable[[np.ndarray, np.ndarray], Any],
        score_function: Callable[[np.ndarray, np.ndarray], float] | None = None,
        predict_function: Callable[[np.ndarray], np.ndarray] | None = None,
        loss_function: Callable[[np.ndarray, np.ndarray], float] | None = None,
        empty_features_value: float = 0.0,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the FeatureSelection game.

        Args:
            x_train: The training data used to fit the model. Should be a 2d matrix of shape
                ``(n_samples, n_features)``.

            y_train: The training labels used to fit the model. Can be a 1d or 2d matrix of shape
                ``(n_samples, n_outputs)``.

            x_test: The test data used to evaluate the model. Should be the same shape as
                ``x_train``.

            y_test: The test labels used to evaluate the model. Should be the same shape as
                ``y_train``.

            fit_function: The function that fits the model to the training data. It should take the
                training data and labels as input.

            score_function: The function that scores the model's performance on the test data. It
                should take the test data and labels as input. If not provided, then
                ``predict_function`` and ``loss_function`` must be provided.

            predict_function: The function that predicts the test labels given the test data. It
                should take the test data as input. If not provided, then ``score_function`` must
                be provided.

            loss_function: The function that computes the loss between the predicted and true test
                labels. It should take the true and predicted test labels as input. If not provided,
                then ``score_function`` must be provided.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print information of the game. Defaults to ``False``.

            empty_features_value: The worth of an empty subset of features. Defaults to 0.0.
        """
        self.empty_features_value = empty_features_value
        super().__init__(
            x_train.shape[1],
            normalization_value=self.empty_features_value,
            normalize=normalize,
            verbose=verbose,
        )

        # set datasets
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

        # sanity check on  input params
        if score_function is None and (loss_function is None or predict_function is None):
            msg = (
                "If score function is not provided, then 'predict_function' and 'loss_function'"
                " must be provided."
            )
            raise ValueError(msg)

        # setup callables
        self._fit_function = fit_function
        self._predict_function = predict_function
        self._loss_function = loss_function
        self._score_function = score_function

        # set empty value
        self.normalization_value = empty_features_value

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Trains and evaluates the model on a coalition (subset) of features.

        The output of the value function is the value of the specified loss function (goodness of
        fit metric).

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
                scores[i] = self.empty_features_value
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
