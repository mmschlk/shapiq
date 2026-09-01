"""Implementation of the ProductKernelExplainer class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from shapiq import InteractionValues
from shapiq.explainer.base import Explainer
from shapiq.game_theory import get_computation_index

from .product_kernel import ProductKernelComputer, ProductKernelSHAPIQIndices
from .validation import validate_pk_model

if TYPE_CHECKING:
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.svm import SVC, SVR

    from shapiq.explainer.custom_types import ExplainerIndices
    from shapiq.typing import Model

    from .base import ProductKernelModel


class ProductKernelExplainer(Explainer):
    """The ProductKernelExplainer class for product kernel-based models.

    The ProductKernelExplainer can be used with a variety of product kernel-based models. The explainer can handle both regression and
    classification models. The product-kernel game explained here, and the attribution method
    defined on it, were proposed by [pkex-shapley]_. This explainer computes the same values
    with the faster algorithm of [quadrashap]_, which replaces that paper's
    elementary-symmetric-polynomial recursion by Gauss-Legendre quadrature of a one-dimensional
    integral.


    References:
        .. [pkex-shapley] Majid Mohammadi, Siu Lun Chau and Krikamol Muandet. (2025). Computing Exact Shapley Values in Polynomial Time for Product-Kernel Methods. https://arxiv.org/abs/2505.16516
        .. [quadrashap] Majid Mohammadi, Grigory Reznikov, Pavel Sinitcyn, Krikamol Muandet and Siu Lun Chau. (2026). QuadraSHAP: Stable and Scalable Shapley Values for Product Games via Gauss-Legendre Quadrature. https://arxiv.org/abs/2605.05870

    Attributes:
        model: The product kernel model to explain. Can be a dictionary, a ProductKernelModel, or a list of ProductKernelModels.
             Note that the model will be converted to a ProductKernelModel if it is not already in that format.
             Supported models include scikit-learn's SVR, SVC (binary classification only), and GaussianProcessRegressor.
             Beware that for classification models, the class to explain is set to the predicted class of the model.
             For further details, see the `validate_pk_model` function in `shapiq.explainer.product_kernel.validation`.
        max_order: The maximum interaction order to be computed. Defaults to ``1``.
        min_order: The minimum interaction order to be computed. Defaults to ``0``.
        index: The type of value to be computed, either ``"SV"`` (Shapley value) or ``"BV"``
            (Banzhaf value).
        n_quadrature_points: The number of Gauss-Legendre nodes used for the computation.
    """

    def __init__(
        self,
        model: (
            ProductKernelModel | Model | SVR | SVC | GaussianProcessRegressor  # pyright: ignore[reportInvalidTypeVarUse]
        ),
        *,
        min_order: int = 0,
        max_order: int = 1,
        index: ProductKernelSHAPIQIndices = "SV",
        n_quadrature_points: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the ProductKernelExplainer.

        Args:
            model: A product kernel-based model to explain.

            min_order: The minimum interaction order to be computed. Defaults to ``0``.

            max_order: The maximum interaction order to be computed. An interaction order of ``1``
                corresponds to the Shapley value. Defaults to ``1``.

            index: The type of value to be computed, either ``"SV"`` (Shapley value) or
                ``"BV"`` (Banzhaf value). Defaults to ``"SV"``.

            n_quadrature_points: The number of Gauss-Legendre nodes. Defaults to ``None``,
                which uses the exact bound ``ceil(d / 2)`` for ``d`` features. Lower values
                trade exactness for speed with a geometrically decaying error. Ignored for
                ``"BV"``, which is a single evaluation point.

            class_index: The class index of the model to explain. Defaults to ``None``, which will
                set the class index to ``1`` per default for classification models and is ignored
                for regression models.

            **kwargs: Additional keyword arguments are ignored.

        """
        super().__init__(model, index=cast("ExplainerIndices", index), max_order=max_order)

        if min_order > self._max_order:
            msg = f"min_order must not exceed max_order, got {min_order=}, {self._max_order=}."
            raise ValueError(msg)
        self._min_order = min_order
        self._base_index: str = get_computation_index(self._index)

        # validate model
        self.converted_model = validate_pk_model(model)

        self.explainer = ProductKernelComputer(
            model=self.converted_model,
            max_order=self._max_order,
            min_order=min_order,
            # the computer re-checks this at runtime and rejects unsupported indices
            index=cast("ProductKernelSHAPIQIndices", self._index),
            n_quadrature_points=n_quadrature_points,
        )

        self.empty_prediction = self._compute_baseline_value()

    def explain_function(  # type: ignore[override]
        self,
        x: np.ndarray,
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Compute Shapley or Banzhaf values for all features of an instance.

        Args:
           x: The instance (1D array) for which to compute the values.
           **kwargs: Additional keyword arguments are ignored.

        Returns:
           The interaction values for the instance.
        """
        n_players = self.converted_model.d

        interactions: dict[tuple[int, ...], float] = {}
        if self._max_order == 1:
            # the first-order route skips building the subset lattice altogether
            values = self.explainer.compute_values(x)
            interactions = {(player,): float(values[player]) for player in range(n_players)}
        else:
            interactions = self.explainer.compute_interaction_values(x)

        return InteractionValues(
            values=interactions,
            index=self._base_index,
            min_order=self._min_order,
            max_order=self.max_order,
            n_players=n_players,
            estimated=False,
            baseline_value=self.empty_prediction,
            target_index=self._index,
        )

    def _compute_baseline_value(self) -> float:
        """Computes the baseline value for the explainer.

        Returns:
            The baseline value for the explainer.

        """
        return self.converted_model.alpha.sum() + self.converted_model.intercept
