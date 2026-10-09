"""The product kernel game of kernel models (support vector machines, Gaussian processes)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.explainer.product_kernel.validation import validate_pk_model
from shapiq.game import Game
from shapiq.utils.modules import safe_isinstance
from shapiq_games._base import as_bool_coalitions

if TYPE_CHECKING:
    from shapiq.explainer.product_kernel.base import ProductKernelModel
    from shapiq.typing import CoalitionMatrix, GameValues

__all__ = ["ProductKernelGame"]

_MAX_ELEMENTS = 2**20  # the most kernel values held at once (coalitions times support vectors)


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
        empty_value: The value of the empty coalition before centering.

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
            model: A fitted RBF ``SVR``, binary ``SVC``, or ``GaussianProcessRegressor`` (without
                ``normalize_y``), or a
                :class:`~shapiq.explainer.product_kernel.ProductKernelModel`.
            x: The explained point of shape ``(n_features,)``.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``False``.

        Raises:
            ValueError: If the model is a Gaussian process with ``normalize_y=True``.
        """
        if getattr(model, "normalize_y", False) and safe_isinstance(
            model, "sklearn.gaussian_process.GaussianProcessRegressor"
        ):
            # shapiq's conversion drops the target scaling, so the game would not match predict
            msg = "Gaussian processes with normalize_y=True are not supported."
            raise ValueError(msg)
        self.model: ProductKernelModel = validate_pk_model(model)
        if self.model.kernel_type != "rbf":
            msg = f"Kernel type '{self.model.kernel_type}' is not supported, only 'rbf'."
            raise NotImplementedError(msg)
        self.x = np.asarray(x, dtype=float).reshape(-1)
        self._squared_distances = (np.asarray(self.model.X_train, dtype=float) - self.x) ** 2
        # with no feature the RBF kernel is one for every support vector
        self.empty_value = float(np.sum(self.model.alpha)) + float(self.model.intercept)
        super().__init__(self.x.shape[0], normalize=normalize, normalization_value=self.empty_value)

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the decision function with the kernel restricted to the coalition."""
        coalitions = as_bool_coalitions(coalitions)
        n_support = self._squared_distances.shape[0]
        step = max(1, _MAX_ELEMENTS // n_support)
        values = np.full(coalitions.shape[0], self.empty_value)
        for start in range(0, coalitions.shape[0], step):
            chunk = coalitions[start : start + step]
            # the squared distance over the coalition's features, added feature by feature
            distances = np.zeros((chunk.shape[0], n_support))
            for feature in range(chunk.shape[1]):
                distances[chunk[:, feature]] += self._squared_distances[:, feature]
            gamma = self.model.gamma
            if gamma is None:  # scikit-learn's default: one over the number of features
                gamma = 1.0 / np.maximum(chunk.sum(axis=1, keepdims=True), 1)
            kernel = np.exp(-gamma * distances)
            decision = np.sum(kernel * self.model.alpha, axis=1) + float(self.model.intercept)
            values[start : start + step] = np.where(chunk.any(axis=1), decision, self.empty_value)
        return values
