"""Functions for converting scikit-learn models to a format used by shapiq."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq.explainer.product_kernel.base import ProductKernelModel

if TYPE_CHECKING:
    from src.shapiq.typing import Model


# TODO(IsaH57): add basic one for SVM? SVR and SVC are same (yet) # noqa: TD003
def convert_svr(model: Model) -> ProductKernelModel:
    """Converts a scikit-learn SVR model to the product kernel format used by shapiq.

    Args:
        model: The scikit-learn SVR model to convert.

    Returns:
        ProductKernelModel: The converted model in the product kernel format.

    """
    X_train = model.support_vectors_
    n, d = X_train.shape  # TODO(IsaH57): rename n and d? (d=num_features) # noqa: TD003
    return ProductKernelModel(
        alpha=model.dual_coef_.flatten(),
        X_train=X_train,
        n=n,
        d=d,
        gamma=model._gamma,  # noqa: SLF001
    )  # TODO (IsaH57): check if gamma is always needed # noqa: TD003


def convert_binsvc(model: Model) -> ProductKernelModel:
    """Converts a scikit-learn SVC model to the product kernel format used by shapiq.

    Args:
        model: The binary scikit-learn SVC model to convert.

    Returns:
        ProductKernelModel: The converted model in the product kernel format.

    """
    X_train = model.support_vectors_
    n, d = X_train.shape
    return ProductKernelModel(
        alpha=model.dual_coef_.flatten(),
        X_train=X_train,
        n=n,
        d=d,
        gamma=model._gamma,  # noqa: SLF001
    )  # TODO (IsaH57): check if gamma is always needed # noqa: TD003


def convert_gp_reg(model: Model) -> ProductKernelModel:
    """Converts a scikit-learn Gaussian Process Regression model to the product kernel format used by shapiq.

    Args:
        model: The scikit-learn Gaussian Process Regression model to convert.

    Returns:
        ProductKernelModel: The converted model in the product kernel format.

    """
    X_train = model.X_train_
    n, d = X_train.shape
    return ProductKernelModel(
        alpha=model.alpha_.flatten(),
        X_train=X_train,
        n=n,
        d=d,
        gamma=(2 * (model.kernel_.length_scale**2)) ** -1,
    )


def convert_gp_binclf(
    model: Model,
) -> ProductKernelModel:
    """Converts a binary scikit-learn Gaussian Process Classifier model to the product kernel format used by shapiq.

    Args:
        model: The binary scikit-learn Gaussian Process Classifier model to convert.

    Returns:
        ProductKernelModel: The converted model in the product kernel format.

    """
    # binary classification has parameter X_train (other than classifier with >2 classes)
    X_train = model.base_estimator_.X_train_
    alpha = (
        model.alpha_.flatten()
    )  # Issue: model doesnt have alpha value #TODO(IsaH57): solve this # noqa: TD003

    n, d = X_train.shape
    return ProductKernelModel(
        alpha=alpha, X_train=X_train, n=n, d=d, gamma=(2 * (model.kernel_.length_scale**2)) ** -1
    )
