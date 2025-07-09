"""This module contains all tabular machine learning games."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from shapiq.games.base import Game

if TYPE_CHECKING:
    from collections.abc import Callable


class GlobalExplanation(Game):
    """The GlobalExplanation game class.

    The GlobalExplanation game is a benchmark game for global explanation methods. It evaluates the
    worth of coalitions of features towards the model's performance. The players are individual
    features, and the worth of a coalition is the performance of the model on a random subset of the
    data where missing features are removed by setting the feature values to a random value from the
    background data. For more details, we highly recommend reading the SAGE paper [1]_ or the
    related blog post [2]_.

    Attributes:
        empty_loss: The model's prediction on an empty data point (all features missing).
        model: The model to explain as a callable function.
        loss_function: The loss function to use for the game.
        predictions: The model's predictions on the data.
        data: The background data used to fit the imputer.
        data_shuffled: The background data shuffled column wise.
        n_samples_eval: The number of background samples to use for each evaluation of the value
            function.

    References:
        .. [1] Covert, I., Lundberg, S., Lee, S.-L. (2020). Understanding Global Feature Contributions With Additive Importance Measures. https://arxiv.org/abs/2004.00668
        .. [2] https://iancovert.com/blog/understanding-shap-sage/
    """

    def __init__(
        self,
        *,
        data: np.ndarray,
        model: Callable[[np.ndarray], np.ndarray],
        loss_function: Callable[[np.ndarray, np.ndarray], float],
        n_samples_eval: int = 10,
        n_samples_empty: int = 200,
        normalize: bool = True,
        random_state: int | None = 42,
        verbose: bool = False,
    ) -> None:
        """Initialize the GlobalExplanation game.

        Args:
            data: The background data used to fit the imputer. Should be a 2d matrix of shape
                ``(n_samples, n_features)``.

            model: The model to explain as a callable function expecting data points as input and
                returning the model's predictions. The input should be a 2d matrix of shape
                ``(n_samples, n_features)`` and the output a 1d vector of shape ``(n_samples,)``.

            loss_function: The loss function to use for the game as a callable function that takes the
                true values and the predictions as input and returns the loss.

            n_samples_eval: The number of background samples to use for each evaluation of the value
                function. The number of model evaluations is ``n_samples_eval * n_coalitions``.
                Defaults to ``10``.

            n_samples_empty: The number of samples to use for the empty subset of features. Defaults
                to ``200``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print information of the game. Defaults to ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        self._random_state = random_state
        self._rng = np.random.default_rng(self._random_state)
        self.n_samples_eval = n_samples_eval  # how many samples to evaluate for each coalition

        self.data = copy.deepcopy(data)
        self._n_samples = self.data.shape[0]
        # shuffle the data column wise (shuffle each column independently)
        self.data_shuffled = copy.deepcopy(self.data)
        for i in range(self.data_shuffled.shape[1]):
            self._rng.shuffle(self.data_shuffled[:, i])

        # get the model, loss function, and predictions
        self.model = model
        self.loss_function = loss_function
        self.predictions = self.model(self.data)

        # get empty prediction
        n_empty_samples = min(n_samples_empty, self.data_shuffled.shape[0])
        idx = self._rng.choice(n_empty_samples, size=self.n_samples_eval, replace=False)
        empty_subset, predictions = self.data_shuffled[idx], self.predictions[idx]
        empty_predictions = self.model(empty_subset)  # model call
        self.empty_loss: float = self.loss_function(predictions, empty_predictions)

        # init the base game
        super().__init__(
            data.shape[1],
            normalize=normalize,
            normalization_value=self.empty_loss,
            verbose=verbose,
        )

    def value_function(self, coalitions: np.ndarray[bool]) -> np.ndarray:
        """Return the worth of the coalitions for the global explanation game.

        The worth of a coalition in the global explanation game is the performance of the model as
        measured by the loss function on a random subset of the data where the features not part of
        the coalition are replaced by shuffled values from the background data.

        Args:
            coalitions: The coalitions as a one-hot matrix for which the game is to be evaluated.

        Returns:
            The worth of the coalitions as a vector of length `n_coalitions`.

        """
        worth = np.zeros(coalitions.shape[0], dtype=float)
        for i, coalition in enumerate(coalitions):
            if not any(coalition):
                worth[i] = self.empty_loss
                continue
            # get the subset of the data
            idx = self._rng.choice(self._n_samples, size=self.n_samples_eval, replace=False)
            subset = self.data[idx].copy()
            predictions = self.predictions[idx]
            # replace the features not part of the subset
            subset[:, ~coalition] = self.data_shuffled[idx][:, ~coalition]
            # get the predictions of the model on the subset
            subset_predictions = self.model(subset)
            # get the loss of the model on the subset
            worth[i] = self.loss_function(predictions, subset_predictions)
        return worth
