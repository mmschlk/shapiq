"""Conversion functions for the product kernel explainer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.shapiq.utils.modules import safe_isinstance

from .conversion import convert_gp_reg, convert_svm

if TYPE_CHECKING:
    from src.shapiq.typing import Model

    from .base import ProductKernelModel

SUPPORTED_MODELS = {
    "sklearn.svm.SVR",
    "sklearn.svm.SVC",
    "sklearn.gaussian_process.GaussianProcessRegressor",
}


def validate_pk_model(
    model: Model,
    class_label: int | None = None,  # TODO(IsaH57): check how to use (Issue #425)
) -> ProductKernelModel:
    """Validate the product kernel model.

    Args:
        model: The model to validate.
        class_label: The class label to be used for validation. Defaults to None.

    Returns:
        The validated product kernel model.
    """
    class_label = class_label or 1  # default to 1 for classification models

    # product kernel model already in the correct format
    if type(model).__name__ == "ProductKernelModel":
        pk_model = model
    elif safe_isinstance(model, "sklearn.svm.SVR") or (
        safe_isinstance(model, "sklearn.svm.SVC") and model.classes_.shape[0] == 2
    ):
        pk_model = convert_svm(model)
    elif safe_isinstance(model, "sklearn.gaussian_process.GaussianProcessRegressor"):
        pk_model = convert_gp_reg(model)

    # unsupported model types
    elif safe_isinstance(model, "sklearn.svm.SVC") and model.classes_.shape[0] >= 2:
        msg = "Only binary SVM classification supported."
        raise TypeError(msg)
    else:
        msg = f"Unsupported model type.Supported models are: {SUPPORTED_MODELS}"
        raise TypeError(msg)
    return pk_model
