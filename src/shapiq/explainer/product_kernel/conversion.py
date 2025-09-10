"""Functions for converting scikit-learn models to a format used by shapiq."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq.explainer.product_kernel.base import ProductKernelModel

if TYPE_CHECKING:
    from shapiq.typing import Model


def convert_svm(model: Model) -> ProductKernelModel:
    """Converts a scikit-learn SVM model to the product kernel format used by shapiq.

    Args:
        model: The scikit-learn SVM model to convert. Can be either a binary support vector classifier (SVC) or a support vector regressor (SVR).

    Returns:
        ProductKernelModel: The converted model in the product kernel format.

    """
    X_train = model.support_vectors_
    n, d = X_train.shape

    if hasattr(model, "kernel"):
        kernel_type = model.kernel
        if kernel_type != "rbf":
            msg = "Currently only RBF kernel is supported for SVM models."
            raise ValueError(msg)
    else:
        msg = "Kernel type not found in the model. Ensure the model is a valid SVC or SVR."
        raise ValueError(msg)

    return ProductKernelModel(
        alpha=model.dual_coef_.flatten(),
        X_train=X_train,
        n=n,
        d=d,
        gamma=model._gamma,  # noqa: SLF001
        kernel_type=kernel_type,
        intercept=model.intercept_[0],
    )  # TODO (IsaH57): check if gamma is always needed or just when rbf is used (Issue #425)


def convert_gp_reg(model: Model) -> ProductKernelModel:
    """Converts a scikit-learn Gaussian Process Regression model to the product kernel format used by shapiq.

    Args:
        model: The scikit-learn Gaussian Process Regression model to convert.

    Returns:
        ProductKernelModel: The converted model in the product kernel format.

    """
    X_train = model.X_train_
    n, d = X_train.shape

    if hasattr(model, "kernel"):
        kernel_type = model.kernel_.__class__.__name__.lower()  # Get the kernel type as a string
        if kernel_type != "rbf":
            msg = "Currently only RBF kernel is supported for Gaussian Process Regression models."
            raise ValueError(msg)
    else:
        msg = "Kernel type not found in the model. Ensure the model is a valid Gaussian Process Regressor."
        raise ValueError(msg)

    return ProductKernelModel(
        alpha=model.alpha_.flatten(),
        X_train=X_train,
        n=n,
        d=d,
        gamma=(2 * (model.kernel_.length_scale**2)) ** -1,
        kernel_type=kernel_type,
    )
