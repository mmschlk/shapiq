"""Base class for imputers."""

from abc import abstractmethod
from typing import Callable, Optional

import numpy as np


class Imputer:
    """Base class for imputers.

    Args:
        model: The model to explain as a callable function expecting a data points as input and
            returning the model's predictions.
        background_data: The background data to use for the explainer as a two-dimensional array
            with shape (n_samples, n_features).
        categorical_features: A list of indices of the categorical features in the background data.
        random_state: The random state to use for sampling. Defaults to `None`.
    """

    @abstractmethod
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        background_data: np.ndarray,
        categorical_features: list[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self._model = model
        self._background_data = background_data
        self._n_features = self._background_data.shape[1]
        self._cat_features: list = [] if categorical_features is None else categorical_features
        self._random_state = random_state
        self._rng = np.random.default_rng(self._random_state)

    @abstractmethod
    def __call__(self, subsets: np.ndarray[bool]) -> np.ndarray[float]:
        """Imputes the missing values of a data point and calls the model.

        Args:
            subsets: A boolean array indicating which features are present (`True`) and which are
                missing (`False`). The shape of the array must be (n_subsets, n_features).

        Returns:
            The model's predictions on the imputed data points. The shape of the array is
            (n_subsets, n_outputs).
        """
        raise NotImplementedError("Method `__call__` must be implemented in a subclass.")
