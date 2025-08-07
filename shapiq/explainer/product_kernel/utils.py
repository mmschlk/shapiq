"""Utility functions for dealing with product kernel structures."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.shapiq.typing import Model


# TODO(IsaH57): remove get_X_train, get_alpha, get_gamma? # noqa: TD003
def get_X_train(model: Model) -> np.ndarray:
    """Retrieve the training sample based on the model type.

    Args:
        model: The product kernel model or SVC/SVR model.

    Returns:
        model.X_train_: 2D-array of samples.
    """
    if hasattr(model, "support_vectors_"):  # For SVM/SVR
        return model.support_vectors_

    if hasattr(model, "X_train_"):  # For GP
        return model.X_train_
    msg = "Unsupported model type for Shapley value computation."
    raise ValueError(msg)


def get_alpha(model: Model) -> np.ndarray:
    """Retrieve the alpha values based on the model type.

    Args:
        model: The product kernel model or SVC/SVR model.

    Returns:
        model.alpha_.flatten(): Array of alpha values required for Shapley value computation.
    """
    if hasattr(model, "dual_coef_"):  # For SVM/SVR
        return model.dual_coef_.flatten()
    if hasattr(model, "alpha_"):  # For GP
        return model.alpha_.flatten()
    msg = "Unsupported model type for Shapley value computation."
    raise ValueError(msg)


def get_gamma(model: Model) -> float:
    """Retrieve the gamma parameter based on the model type.

    Args:
        model: The product kernel model or SVC/SVR model.

    Returns:
        Gamma parameter for the RBF kernel.
    """
    if hasattr(model, "_gamma"):  # For SVM/SVR
        return model._gamma  # noqa: SLF001 (uses existing implementation)
    if hasattr(model.kernel_, "length_scale"):  # For GP
        return (2 * (model.kernel_.length_scale**2)) ** -1
    msg = "Unsupported model type for Shapley value computation."
    raise ValueError(msg)


# TODO(IsaH57): maybe move to product_kernel/base.py into ProductKernelModel? # noqa: TD003
def precompute_mu(num_features: int) -> np.ndarray:
    """Precompute mu coefficients for computing Shapley values.

    Args:
        num_features: Number of features.

    Returns:
        List of precomputed mu coefficients.
    """
    unnormalized_factors = [
        (math.factorial(q) * math.factorial(num_features - q - 1)) for q in range(num_features)
    ]

    return np.array(unnormalized_factors) / math.factorial(num_features)
