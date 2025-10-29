"""Conversion functions for the product kernel explainer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVC, SVR

from .base import ProductKernelModel
from .conversion import convert_gp_reg, convert_svm

if TYPE_CHECKING:
    from shapiq.typing import Model


SUPPORTED_MODELS = {
    "sklearn.svm.SVR",
    "sklearn.svm.SVC",
    "sklearn.gaussian_process.GaussianProcessRegressor",
}


def validate_pk_model(
    model: (Model | SVR | SVC | GaussianProcessRegressor),  # pyright: ignore[reportInvalidTypeVarUse]
) -> ProductKernelModel:
    """Validate the product kernel model.

    Args:
        model: The model to validate.

    Returns:
        The validated product kernel model.

    Raises:
        TypeError: If the model is not supported.
    """
    # product kernel model already in the correct format
    if isinstance(model, ProductKernelModel):
        return model

    if isinstance(model, SVR):
        return convert_svm(model)

    if isinstance(model, SVC):
        if model.classes_.shape[0] == 2:
            return convert_svm(model)
        msg = "Only binary SVM classification supported."
        raise TypeError(msg)

    if isinstance(model, GaussianProcessRegressor):
        return convert_gp_reg(model)

    msg = f"Unsupported model type.Supported models are: {SUPPORTED_MODELS}"
    raise TypeError(msg)
