"""Implementation of the ProductKernelExplainer class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shapiq import InteractionValues
from shapiq.explainer.base import Explainer
from shapiq.game_theory import get_computation_index

from .product_kernel import ProductKernelComputer, ProductKernelSHAPIQIndices
from .validation import validate_pk_model

if TYPE_CHECKING:
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.svm import SVC, SVR

    from shapiq.typing import Model

    from .base import ProductKernelModel


class ProductKernelExplainer(Explainer):
    """The ProductKernelExplainer class for product kernel-based models.

    The ProductKernelExplainer can be used with a variety of product kernel-based models. The explainer can handle both regression and
    classification models. See [pkex-shapley]_ for details.


    References:
        .. [pkex-shapley] Majid Mohammadi and Siu Lun Chau, Krikamol Muandet. (2025). Computing Exact Shapley Values in Polynomial Time for Product-Kernel Methods. https://arxiv.org/abs/2505.16516

    Attributes:
        model: The product kernel model to explain. Can be a dictionary, a ProductKernelModel, or a list of ProductKernelModels.
             Note that the model will be converted to a ProductKernelModel if it is not already in that format.
             Supported models include scikit-learn's SVR, SVC (binary classification only), and GaussianProcessRegressor.
             Beware that for classification models, the class to explain is set to the predicted class of the model.
             For further details, see the `validate_pk_model` function in `shapiq.explainer.product_kernel.validation`.
        max_order: The maximum interaction order to be computed. Defaults to ``1``.
        min_order: The minimum interaction order to be computed. Defaults to ``0``.
        index: The type of interaction to be computed. Currently, only ``"SV"`` is supported.
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
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the ProductKernelExplainer.

        Args:
            model: A product kernel-based model to explain.

            min_order: The minimum interaction order to be computed. Defaults to ``0``.

            max_order: The maximum interaction order to be computed. An interaction order of ``1``
                corresponds to the Shapley value. Defaults to ``1``.

            index: The type of interaction to be computed. Currently, only ``"SV"`` is supported.

            class_index: The class index of the model to explain. Defaults to ``None``, which will
                set the class index to ``1`` per default for classification models and is ignored
                for regression models.

            **kwargs: Additional keyword arguments are ignored.

        """
        if max_order > 1:
            msg = "ProductKernelExplainer currently only supports max_order=1."
            raise ValueError(msg)

        super().__init__(model, index=index, max_order=max_order)

        self._min_order = min_order
        self._max_order = max_order

        self._index = index
        self._base_index: str = get_computation_index(self._index)

        # validate model
        self.converted_model = validate_pk_model(model)

        self.explainer = ProductKernelComputer(
            model=self.converted_model,
            max_order=max_order,
            index=index,
        )

        self.empty_prediction = self._compute_baseline_value()

    def explain_function(
        self,
        x: np.ndarray,
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Compute Shapley values for all features of an instance.

        Args:
           x: The instance (1D array) for which to compute Shapley values.
           **kwargs: Additional keyword arguments are ignored.

        Returns:
           The interaction values for the instance.
        """
        n_players = self.converted_model.d

        # compute the kernel vectors for the instance x
        kernel_vectors = self.explainer.compute_kernel_vectors(self.converted_model.X_train, x)

        shapley_values = {}
        for j in range(self.converted_model.d):
            shapley_values.update({(j,): self.explainer.compute_shapley_value(kernel_vectors, j)})

        return InteractionValues(
            values=shapley_values,
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
