"""This module contains the base explainer classes for the shapiq package."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np


@dataclass
class Explanation:
    """ This class contains the explanation of the model.

    Attributes:
        interaction_values: The interaction values of the model. Mapping from order to the
            interaction values.
        explanation_type: The type of the explanation. Available types are 'SII', 'nSII', 'STI',
            and 'FSI'.
        order: The maximum order of the explanation.
    """
    interaction_values: dict[int, np.ndarray]
    explanation_type: str
    order: int

    def __post_init__(self) -> None:
        """Checks if the explanation type is valid."""
        if self.explanation_type not in ["SII", "nSII", "STI", "FSI"]:
            raise ValueError(
                f"Explanation type {self.explanation_type} is not valid. "
                f"Available types are 'SII', 'nSII', 'STI', and 'FSI'."
            )
        

class Explainer(ABC):

    def __init__(
            self,
            model: Any,
            X: np.ndarray,
            y: np.ndarray
    ) -> None:
        """Initializes the Explainer class.

        Args:
            model: The model to be explained.
            X: The input data.
            y: The output data.
        """
        self.model = model
        self.X = X
        self.y = y
