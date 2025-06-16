"""Implementation of the TreeExplainer class.

The :class:`~shapiq.explainer.tree.explainer.TreeSHAPIQ` uses the
:class:`~shapiq.explainer.tree.treeshapiq.TreeSHAPIQ` algorithm for computing any-order Interactions
for tree ensembles.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from shapiq.explainer.base import Explainer
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions

from .treeshapiq import TreeSHAPIQ, TreeSHAPIQIndices
from .validation import validate_tree_model

if TYPE_CHECKING:
    import numpy as np

    from shapiq.utils.custom_types import Model

    from .base import TreeModel


class TreeExplainer(Explainer):
    """The TreeExplainer class for tree-based models.

    The explainer for tree-based models using the
    :class:`~shapiq.explainer.tree.treeshapiq.TreeSHAPIQ` algorithm. For details, refer to
    `Muschalik et al. (2024)` [Mus24]_.

    TreeSHAP-IQ is an algorithm for computing Shapley Interaction values for tree-based models.
    It is based on the Linear TreeSHAP algorithm by `Yu et al. (2022)` [Yu22]_, but extended to
    compute Shapley Interaction values up to a given order. TreeSHAP-IQ needs to visit each node
    only once and makes use of polynomial arithmetic to compute the Shapley Interaction values
    efficiently.

    The TreeExplainer can be used with a variety of tree-based models, including
    ``scikit-learn``, ``XGBoost``, and ``LightGBM``. The explainer can handle both regression and
    classification models.

    References:
        .. [Yu22] Peng Yu, Chao Xu, Albert Bifet, Jesse Read. (2022). Linear Tree Shap. In: Proceedings of 36th Conference on Neural Information Processing Systems. https://openreview.net/forum?id=OzbkiUo24g
        .. [Mus24] Maximilian Muschalik, Fabian Fumagalli, Barbara Hammer, & Eyke Hüllermeier (2024). Beyond TreeSHAP: Efficient Computation of Any-Order Shapley Interactions for Tree Ensembles. In: Proceedings of the AAAI Conference on Artificial Intelligence, 38(13), 14388-14396. https://doi.org/10.1609/aaai.v38i13.29352

    """

    def __init__(
        self,
        model: dict | TreeModel | list[TreeModel] | Model,
        *,
        max_order: int = 2,
        min_order: int = 0,
        index: TreeSHAPIQIndices = "k-SII",
        class_index: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the TreeExplainer.

        Args:
            model: A tree-based model to explain.

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

        # validate and parse model
        validated_model = validate_tree_model(model, class_label=class_index)
        self._trees: list[TreeModel] | TreeModel = copy.deepcopy(validated_model)
        if not isinstance(self._trees, list):
            self._trees = [self._trees]
        self._n_trees = len(self._trees)

        self._min_order: int = min_order
        self._class_label: int | None = class_index

        # setup explainers for all trees
        self._treeshapiq_explainers: list[TreeSHAPIQ] = [
            TreeSHAPIQ(model=_tree, max_order=self._max_order, index=index) for _tree in self._trees
        ]
        self.baseline_value = self._compute_baseline_value()

    def explain_function(
        self,
        x: np.ndarray,
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Computes the Shapley Interaction values for a single instance.

        Args:
            x: The instance to explain as a 1-dimensional array.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            The interaction values for the instance.

        """
        if len(x.shape) != 1:
            msg = "explain expects a single instance, not a batch."
            raise TypeError(msg)
        # run treeshapiq for all trees
        interaction_values: list[InteractionValues] = []
        for explainer in self._treeshapiq_explainers:
            tree_explanation = explainer.explain(x)
            interaction_values.append(tree_explanation)

        # combine the explanations for all trees
        final_explanation = interaction_values[0]
        if len(interaction_values) > 1:
            for i in range(1, len(interaction_values)):
                final_explanation += interaction_values[i]

        if self._min_order == 0 and final_explanation.min_order == 1:
            final_explanation.min_order = 0
            final_explanation = finalize_computed_interactions(
                final_explanation,
                target_index=self._index,
            )
        return finalize_computed_interactions(
            final_explanation,
            target_index=self._index,
        )

    def _compute_baseline_value(self) -> float:
        """Computes the baseline value for the explainer.

        The baseline value is the sum of the empty predictions of all trees in the ensemble.

        Returns:
            The baseline value for the explainer.

        """
        return sum(
            [treeshapiq.empty_prediction for treeshapiq in self._treeshapiq_explainers],
        )
