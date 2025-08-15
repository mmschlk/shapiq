"""Implementation of the ProductKernelExplainer class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.shapiq.explainer.base import Explainer
from src.shapiq.explainer.product_kernel.validation import validate_pk_model

from .product_kernel import ProductKernelComputer, ProductKernelSHAPIQIndices

if TYPE_CHECKING:
    import numpy as np

    from src.shapiq.utils.custom_types import Model

    from .base import ProductKernelModel


class ProductKernelExplainer(Explainer):
    """The ProductKernelExplainer class for product kernel-based models.

    The ProductKernelExplainer can be used with a variety of product kernel-based models. The explainer can handle both regression and
    classification models.

    References:
        -- [pkex-shapley] Majid Mohammadi and Siu Lun Chau, Krikamol Muandet. (2025). Computing Exact Shapley Values in Polynomial Time for Product-Kernel Methods. https://arxiv.org/abs/2505.16516

    Attributes:
        model: The product kernel model to explain. Can be a dictionary, a ProductKernelModel, or a list of ProductKernelModels.
        max_order: The maximum interaction order to be computed. Defaults to ``1``.
        min_order: The minimum interaction order to be computed. Defaults to ``0``.
        index: The type of interaction to be computed. Currently, only ``"SV"`` is supported.
        class_index: The class index of the model to explain. Defaults to ``None``, which will set the class index to ``1`` per default for classification models and is ignored for regression models.
    """

    def __init__(
        self,
        model: dict
        | ProductKernelModel
        | list[ProductKernelModel]
        | Model,  # TODO (IsaH57): check if list of models is neded (Issue #425)
        *,
        max_order: int = 1,
        min_order: int = 0,
        index: ProductKernelSHAPIQIndices = "SV",
        class_index: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the ProductKernelExplainer.

        Args:
            model: A product kernel-based model to explain.

            max_order: The maximum interaction order to be computed. An interaction order of ``1``
                corresponds to the Shapley value. Defaults to ``1``.

            min_order: The minimum interaction order to be computed. Defaults to ``0``.

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

        self._min_order: int = min_order
        self._class_label: int | None = class_index

        # validate model
        self.model = validate_pk_model(model, class_label=class_index)

        self.explainer = ProductKernelComputer(
            model=self.model,
            max_order=max_order,
            index=index,
        )

        # TODO(IsaH57): add computation of baseline (Issue #425)
        # self.baseline_value = self._compute_baseline_value() # noqa: ERA001

    def explain_function(
        self,
        x: np.ndarray,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[Any]:
        """Compute Shapley values for all features of an instance.

        This function is explain() in PKeX RBFLocalExplainer().

        Args:
           x: The instance (1D array) for which to compute Shapley values.
           **kwargs: Additional keyword arguments are ignored.

        Returns:
           List of Shapley values, one for each feature.
        """
        # compute the kernel vectors for the instance x
        kernel_vectors = self.explainer.compute_kernel_vectors(self.model.X_train, x)

        shapley_values = []
        for j in range(self.model.d):
            shapley_values.append(self.explainer.compute_shapley_value(kernel_vectors, j))  # noqa: PERF401 (using existing implementation from RKHS-ExactSHAP)

        return shapley_values  # return interaction values object

    def _compute_baseline_value(self) -> float:
        """Computes the baseline value for the explainer.

        Returns:
            The baseline value for the explainer.

        """
        msg = "The method _compute_baseline_value is not yet implemented in the ProductKernelExplainer."
        raise NotImplementedError(msg)
