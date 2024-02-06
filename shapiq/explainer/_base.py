"""This module contains the base explainer classes for the shapiq package."""
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from approximator._interaction_values import InteractionValues
from explainer.imputer.marginal_imputer import MarginalImputer


class Explainer(ABC):
    """The base class for all explainers in the shapiq package.

    Args:
        model: The model to explain as a callable function expecting a data points as input and
            returning the model's predictions.
        background_data: The background data to use for the explainer.
    """

    @abstractmethod
    def __init__(
        self, model: Callable[[np.ndarray], np.ndarray], background_data: np.ndarray
    ) -> None:
        self._model = model
        self._background_data = background_data
        self._n_features = self._background_data.shape[1]
        self._imputer = MarginalImputer(self._model, self._background_data)

    @abstractmethod
    def explain(self, x_explain: np.ndarray) -> InteractionValues:
        """Explains the model's predictions."""
        raise NotImplementedError("Method `explain` must be implemented in a subclass.")
