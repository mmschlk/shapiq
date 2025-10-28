"""Functions for converting scikit-learn models to a format used by shapiq."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.explainer.product_kernel.base import ProductKernelModel

if TYPE_CHECKING:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.svm import SVC, SVR


def convert_svm(model: SVC | SVR) -> ProductKernelModel:
    """Converts a scikit-learn SVM model to the product kernel format used by shapiq.

    Args:
        model: The scikit-learn SVM model to convert. Can be either a binary support vector classifier (SVC) or a support vector regressor (SVR).

    Returns:
        ProductKernelModel: The converted model in the product kernel format.

    """
    X_train = model.support_vectors_
    n, d = X_train.shape

    if hasattr(model, "kernel"):
        kernel_type = model.kernel  # pyright: ignore[reportAttributeAccessIssue]
        if kernel_type != "rbf":
            msg = "Currently only RBF kernel is supported for SVM models."
            raise ValueError(msg)
    else:
        msg = "Kernel type not found in the model. Ensure the model is a valid SVC or SVR."
        raise ValueError(msg)

    return ProductKernelModel(
        alpha=model.dual_coef_.flatten(),  # pyright: ignore[reportAttributeAccessIssue]
        X_train=X_train,
        n=n,
        d=d,
        gamma=model._gamma,  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue] # noqa: SLF001
        kernel_type=kernel_type,
        intercept=model.intercept_[0],
    )


def convert_gp_reg(model: GaussianProcessRegressor) -> ProductKernelModel:
    """Converts a scikit-learn Gaussian Process Regression model to the product kernel format used by shapiq.

    Args:
        model: The scikit-learn Gaussian Process Regression model to convert.

    Returns:
        ProductKernelModel: The converted model in the product kernel format.

    """
    X_train = np.array(model.X_train_)
    n, d = X_train.shape

    if hasattr(model, "kernel"):
        kernel_type = model.kernel_.__class__.__name__.lower()  # Get the kernel type as a string
        if kernel_type != "rbf":
            msg = "Currently only RBF kernel is supported for Gaussian Process Regression models."
            raise ValueError(msg)
    else:
        msg = "Kernel type not found in the model. Ensure the model is a valid Gaussian Process Regressor."
        raise ValueError(msg)

    alphas = np.array(model.alpha_).flatten()
    parameters = (
        model.kernel_.get_params()  # pyright: ignore[reportAttributeAccessIssue]
    )
    if "length_scale" in parameters:
        length_scale = parameters["length_scale"]
    else:
        msg = "Length scale parameter not found in the kernel."
        raise ValueError(msg)

    return ProductKernelModel(
        alpha=alphas,
        X_train=X_train,
        n=n,
        d=d,
        gamma=(2 * (length_scale**2)) ** -1,
        kernel_type=kernel_type,
    )
