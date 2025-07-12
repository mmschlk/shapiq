"""Implementation of the ProductKernelExplainer class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shapiq.explainer.base import Explainer
from shapiq.explainer.validation import validate_pk_model

from .product_kernel import ProductKernelSHAPIQ, ProductKernelSHAPIQIndices

if TYPE_CHECKING:
    import numpy as np

    from shapiq.utils.custom_types import Model

    from .base import ProductKernelModel


class ProductKernelExplainer(Explainer):
    """The ProductKernelExplainer class for product kernel-based models.

    The ProductKernelExplainer can be used with a variety of product kernel-based models from the 'PKeX-Shapley' package. The explainer can handle both regression and
    classification models.

    References:
        -- [pkex-shapley] Majid Mohammadi and Siu Lun Chau, Krikamol Muandet. (2025). Computing Exact Shapley Values in Polynomial Time for Product-Kernel Methods. https://arxiv.org/abs/2505.16516
    """

    def __init__(
        self,
        model: dict | ProductKernelModel | list[ProductKernelModel] | Model,
        *,
        max_order: int = 1,  # TODO(IsaH57): change to 2 # noqa: TD003
        min_order: int = 0,
        index: ProductKernelSHAPIQIndices = "SV",  # TODO(IsaH57): choose other default # noqa: TD003
        class_index: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the ProductKernelExplainer.

        Args:
            model: A product kernel-based model to explain.

            # TODO: support shapley interactions:
            max_order: The maximum interaction order to be computed. An interaction order of ``1``
                corresponds to the Shapley value. Any value higher than ``1`` computes the Shapley
                interaction values up to that order. Defaults to ``2``.

            min_order: The minimum interaction order to be computed. Defaults to ``1``.

            index: The type of interaction to be computed. It can be one of
                ``["k-SII", "SII", "STII", "FSII", "BII", "SV"]``. All indices apart from ``"BII"``
                will reduce to the ``"SV"`` (Shapley value) for order 1. Defaults to ``"k-SII"``.

            class_index: The class index of the model to explain. Defaults to ``None``, which will
                set the class index to ``1`` per default for classification models and is ignored
                for regression models.

            **kwargs: Additional keyword arguments are ignored.

        """
        super().__init__(model, index=index, max_order=max_order)

        self._min_order: int = min_order
        self._class_label: int | None = class_index

        # validate model
        self.model = validate_pk_model(model, class_label=class_index)

        self.explainer = ProductKernelSHAPIQ(
            model=self.model,
            max_order=max_order,
            index=index,
        )

        # TODO(IsaH57): add computation # noqa: TD003
        # self.baseline_value = self._compute_baseline_value() # noqa: ERA001

    def explain_function(
        self,
        x: np.ndarray,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[Any]:  # TODO(IsaH57): make return InteractionValues # noqa: TD003
        """Compute Shapley values for all features of an instance.

        This function is explain() in PKeX RBFLocalExplainer().

        Args:
           x: The instance (1D array) for which to compute Shapley values.
           **kwargs: Additional keyword arguments are ignored.

        Returns:
           List of Shapley values, one for each feature.
        """
        # We compute the kernel vectors for the instance x
        kernel_vectors = self.explainer.compute_kernel_vectors(self.model.X_train, x)

        shapley_values = []
        for j in range(self.model.d):
            shapley_values.append(self.explainer.compute_shapley_value(kernel_vectors, j))  # noqa: PERF401 (used existing implementation)
        # TODO(IsaH57): add finalize_computed_interactions() # noqa: TD003
        return shapley_values

    def _compute_baseline_value(self) -> float:
        """Computes the baseline value for the explainer.

        The baseline value is the sum of the empty predictions of all trees in the ensemble.

        Returns:
            The baseline value for the explainer.

        """
        msg = "The method _compute_baseline_value is not yet implemented in the ProductKernelExplainer."
        raise NotImplementedError(msg)
