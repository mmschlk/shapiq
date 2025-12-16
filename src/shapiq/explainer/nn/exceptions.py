"""Custom exceptions used by NN Explainers."""

from __future__ import annotations


class MultiOutputNNError(Exception):
    """Exception that is raised when a user tries to initialize an NN Explainer with a model that has multiple output columns."""

    def __init__(self) -> None:
        """Initializes the exception object."""
        super().__init__(
            "Multi-output nearest-neighbor classifiers are not supported. Make sure to pass the training labels as a 1D vector when calling `model.fit()`."
        )


class UnsupportedKNNWeightsError(Exception):
    """Exception that is raised when a user tries to initialize a KNN Explainer with a model that uses a weights parameter that is not supported."""

    def __init__(self, unsupported_weights: str, allowed_weights: list[str]) -> None:
        """Initializes the exception object."""
        msg = f"KNN model uses unsupported weights parameter {unsupported_weights}. Allowed values are {', '.join(allowed_weights)}"
        super().__init__(msg)
