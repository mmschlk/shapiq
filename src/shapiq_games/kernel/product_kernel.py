"""The product kernel game of kernel models (support vector machines, Gaussian processes)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from shapiq.explainer.product_kernel.validation import validate_pk_model
from shapiq.game import Game
from shapiq_games._base import ConfigMixin, as_bool_coalitions, resolve_x
from shapiq_games._setup import configure

if TYPE_CHECKING:
    from shapiq.explainer.product_kernel.base import ProductKernelModel

__all__ = ["ProductKernelGame"]


class ProductKernelGame(ConfigMixin, Game):
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

    @classmethod
    def from_config(
        cls,
        *,
        dataset: str,
        model: str = "svm",
        x: int = 0,
        n_train: int = 500,
        random_state: int = 42,
        test_size: float = 0.2,
        model_params: dict[str, Any] | None = None,
        normalize: bool = False,
    ) -> Self:
        """Build the game for a registered dataset.

        The kernel model is fitted on ``n_train`` seeded training points (kernel models scale
        poorly with the number of training points); the explained point is taken from the test
        split.

        Args:
            dataset: The dataset name (regression, or binary classification for ``"svm"``).
            model: ``"svm"`` or ``"gaussian_process"`` (regression only). Defaults to ``"svm"``.
            x: The index of the explained point in the test split. Defaults to ``0``.
            n_train: The number of training points of the kernel model. Defaults to ``500``.
            random_state: The seed of the split, the training sample, and the model.
            test_size: The fraction of the data used as test set. Defaults to ``0.2``.
            model_params: Hyperparameters of the model.
            normalize: Whether to center the game.

        Returns:
            The configured game.
        """
        from shapiq_games.models import build_model

        setup = configure(
            dataset=dataset, model=None, random_state=random_state, test_size=test_size
        )
        split = setup.split
        model_params = dict(model_params or {})
        rng = np.random.default_rng(random_state)
        n = min(n_train, split.x_train.shape[0])
        rows = np.sort(rng.choice(split.x_train.shape[0], size=n, replace=False))
        fitted = build_model(model, split.task, random_state=random_state, **model_params)
        fitted.fit(split.x_train[rows], split.y_train[rows])
        game = cls(fitted, resolve_x(x, split.x_test), normalize=normalize)
        config = {**setup.config, "model": model, "model_params": model_params}
        return game._set_config(**config, x=x, n_train=n_train, normalize=normalize)
