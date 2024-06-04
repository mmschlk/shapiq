"""Base class for all imputers."""

from abc import abstractmethod
from typing import Optional

import numpy as np

from ...explainer import utils
from ..base import Game


class Imputer(Game):
    """Base class for imputers.

    Args:
        model: The model to explain as a callable function expecting a data points as input and
            returning the model's predictions.
        data: The background data to use for the explainer as a 2-dimensional array
            with shape ``(n_samples, n_features)``.
        categorical_features: A list of indices of the categorical features in the background data.
        random_state: The random state to use for sampling. Defaults to ``None``.
    """

    @abstractmethod
    def __init__(
        self,
        model,
        data: np.ndarray,
        categorical_features: list[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if callable(model):
            self._predict_function = utils.predict_callable
        else:  # shapiq.Explainer
            self._predict_function = model._predict_function
        self.model = model
        self.data = data
        self._n_features = self.data.shape[1]
        self._cat_features: list = [] if categorical_features is None else categorical_features
        self._random_state = random_state
        self._rng = np.random.default_rng(self._random_state)

        # the normalization_value needs to be set in the subclass
        super().__init__(n_players=self._n_features, normalize=False)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Provides a unified prediction interface."""
        return self._predict_function(self.model, x)
