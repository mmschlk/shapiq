"""This module contains the base explainer classes for the shapiq package."""
from abc import ABC, abstractmethod
from typing import Any
import warnings


class Explainer(ABC):
    """The base class for all explainers in the shapiq package."""

    @abstractmethod
    def __init__(self) -> None:
        """Initializes the explainer."""
        warnings.warn("Explainer is not implemented yet.")

    @abstractmethod
    def explain(self) -> Any:
        warnings.warn("Explainer is not implemented yet.")
