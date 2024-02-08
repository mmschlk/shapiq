"""This module contains the base explainer classes for the shapiq package."""
from abc import ABC, abstractmethod


import numpy as np

from approximator._interaction_values import InteractionValues


class Explainer(ABC):
    """The base class for all explainers in the shapiq package.

    Args:
        n_features: The number of features in the model.
    """

    @abstractmethod
    def __init__(self, n_features: int) -> None:
        self._n_features: int = n_features

    @abstractmethod
    def explain(self, x_explain: np.ndarray) -> InteractionValues:
        """Explains the model's predictions."""
        raise NotImplementedError("Method `explain` must be implemented in a subclass.")
