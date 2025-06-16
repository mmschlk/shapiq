"""Base class for all Imputers."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from shapiq.explainer import utils
from shapiq.games.base import Game

if TYPE_CHECKING:
    from shapiq.utils import Model


class Imputer(Game):
    """Base class for Imputers.

    Attributes:
        n_features: The number of features in the data (equals the number of players in the game).
        data: The background data to use for the imputer.
        model: The model to impute missing values for as a callable function.
        sample_size: The number of samples to draw from the background data.
        random_state: The random state to use for sampling.
        empty_prediction: The model's prediction on an empty data point (all features missing).

    Properties:
        x: The explanation point to use the imputer on.

    """

    @abstractmethod
    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        x: np.ndarray | None = None,
        *,
        sample_size: int | None = 100,
        categorical_features: list[int] | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initializes the base imputer.

        Args:
            model: The model to explain as a callable function expecting a data points as input and
                returning the model's predictions.

            data: The background data to use for the explainer as a 2-dimensional array with shape
                ``(n_samples, n_features)``.

            x: The explanation point to use the imputer on either as a 2-dimensional array with
                shape ``(1, n_features)`` or as a vector with shape ``(n_features,)``.

            sample_size: The number of samples to draw from the background data. Defaults to ``100``
                but is usually overwritten in the subclasses.

            categorical_features: A list of indices of the categorical features in the background
                data.

            random_state: The random state to use for sampling. Defaults to ``None``.

            verbose: A flag to enable verbose imputation, which will print a progress bar for model
                evaluation. Note that this can slow down the imputation process. Defaults to
                ``False``.

        """
        if callable(model) and not hasattr(model, "_predict_function"):
            self._predict_function = utils.predict_callable
        # shapiq.Explainer adds a _shapiq_predict_function to the model to make it callable
        elif hasattr(model, "_shapiq_predict_function"):
            self._predict_function = model._shapiq_predict_function  # noqa: SLF001
        else:
            msg = "The model must be callable or have a predict function."
            raise ValueError(msg)
        self.model = model
        # check if data is a vector
        if data.ndim == 1:
            data = data.reshape(1, data.shape[0])
        self.data = data
        self.sample_size = sample_size
        self.empty_prediction: float = 0.0  # will be overwritten in the subclasses
        self.n_features = self.data.shape[1]
        self._cat_features: list = [] if categorical_features is None else categorical_features
        self.random_state = random_state
        self._rng = np.random.default_rng(self.random_state)

        # fit x
        self._x: np.ndarray | None = None  # will be overwritten @ fit
        if x is not None:
            self.fit(x)

        # init the game
        # developer note: the normalization_value needs to be set in the subclass
        super().__init__(n_players=self.n_features, normalize=False, verbose=verbose)

    @property
    def x(self) -> np.ndarray | None:
        """Returns the explanation point if it is set."""
        return self._x.copy() if self._x is not None else None

    def set_random_state(self, random_state: int | None = None) -> None:
        """Sets the random state for the imputer and the model.

        Args:
            random_state: The random state to set. Defaults to ``None``, which will set a not
                deterministic random state.

        """
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Provides a unified prediction interface.

        Args:
            x: The data point to predict the model's output for.

        Returns:
            The model's prediction for the given data point as a vector.

        """
        return self._predict_function(self.model, x)

    def fit(self, x: np.ndarray) -> Imputer:
        """Fits the imputer to the explanation point.

        Args:
            x: The explanation point to use the imputer on either as a 2-dimensional array with
                shape ``(1, n_features)`` or as a vector with shape ``(n_features,)``.

        Returns:
            The fitted imputer.

        """
        self._x = x.copy()
        if self._x.ndim == 1:
            self._x = self._x.reshape(1, x.shape[0])
        return self

    def insert_empty_value(self, outputs: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        """Inserts the empty value into the outputs.

        Args:
            outputs: The model's predictions on the imputed data points.
            coalitions: The coalitions for which the model's predictions were made.

        Returns:
            The model's predictions with the empty value inserted for the empty coalitions.

        """
        outputs[~np.any(coalitions, axis=1)] = self.empty_prediction
        return outputs
