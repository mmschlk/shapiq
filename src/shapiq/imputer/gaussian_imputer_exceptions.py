"""Custom exceptions for the imputer module."""

from __future__ import annotations


class CategoricalFeatureError(ValueError):
    """Exception raised when categorical features are detected."""

    def __init__(self, feature_indices: list[int]) -> None:
        """Initializes the error with the categorical feature indices.

        Args:
            feature_indices: List of indices of features that are categorical.
        """
        feature_names = [f"f{i + 1}" for i in feature_indices]
        message = (
            f"The following are categorical features: {', '.join(feature_names)}. "
            "Gaussian imputation does not support categorical features."
        )
        super().__init__(message)
        self.feature_indices = feature_indices
