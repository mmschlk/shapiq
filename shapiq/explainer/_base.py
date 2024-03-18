"""This module contains the base explainer classes for the shapiq package."""

from abc import ABC, abstractmethod

import numpy as np
from interaction_values import InteractionValues


class Explainer(ABC):
    """The base class for all explainers in the shapiq package. All explainers should inherit from
    this class."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def explain(self, x_explain: np.ndarray) -> InteractionValues:
        """Explains the model's predictions."""
        raise NotImplementedError("Method `explain` must be implemented in a subclass.")
