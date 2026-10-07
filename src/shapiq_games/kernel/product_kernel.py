"""The product kernel game of kernel models (support vector machines, Gaussian processes)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from shapiq.explainer.product_kernel.validation import validate_pk_model
from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions

if TYPE_CHECKING:
    from shapiq.explainer.product_kernel.base import ProductKernelModel

__all__ = ["ProductKernelGame"]


class ProductKernelGame(Game):
    r"""The product kernel game of a model whose decision function is a weighted sum of kernels.

    For a model :math:`f(x) = \sum_i \alpha_i K(x^i, x) + b` with a product kernel :math:`K`, the
    value of a coalition :math:`S` restricts the kernel to the features in :math:`S`:

    .. math::
        v(S) = \sum_i \alpha_i K(x^i_S, x_S) + b

    The empty coalition has :math:`K = 1`, i.e. :math:`v(\emptyset) = \sum_i \alpha_i + b`
    (Mohammadi et al., 2025, https://arxiv.org/abs/2505.16516). Its exact Shapley values are
    computed by :class:`~shapiq.explainer.product_kernel.ProductKernelExplainer`. Only the RBF
    kernel is supported.

    Attributes:
        model: The model in the :class:`~shapiq.explainer.product_kernel.ProductKernelModel`
            format.
        x: The explained point.

    Examples:
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.svm import SVR
        >>> model = SVR(kernel="rbf").fit(X, y)
        >>> game = ProductKernelGame(model, x=X[0])
        >>> bool(np.isclose(game(game.grand_coalition)[0], model.predict(X[:1])[0]))
        True
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        x: np.ndarray,
        *,
        normalize: bool = False,
    ) -> None:
        """Initialize the product kernel game.

        Args:
            model: A fitted RBF ``SVR``, binary ``SVC``, or ``GaussianProcessRegressor``, or a
                :class:`~shapiq.explainer.product_kernel.ProductKernelModel`.
            x: The explained point of shape ``(n_features,)``.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``False``.
        """
        self.model: ProductKernelModel = validate_pk_model(model)
        if self.model.kernel_type != "rbf":
            msg = f"Kernel type '{self.model.kernel_type}' is not supported, only 'rbf'."
            raise NotImplementedError(msg)
        self.x = np.asarray(x, dtype=float).reshape(-1)
        empty_value = float(np.sum(self.model.alpha)) + float(self.model.intercept)
        super().__init__(self.x.shape[0], normalize=normalize, normalization_value=empty_value)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the decision function with the kernel restricted to the coalition."""
        coalitions = as_bool_coalitions(coalitions)
        alpha = self.model.alpha
        values = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            if not coalition.any():
                values[i] = float(np.sum(alpha)) + float(self.model.intercept)
                continue
            kernel = rbf_kernel(
                X=self.model.X_train[:, coalition],
                Y=self.x[coalition].reshape(1, -1),
                gamma=self.model.gamma,
            )
            values[i] = float((alpha @ kernel).squeeze()) + float(self.model.intercept)
        return values
