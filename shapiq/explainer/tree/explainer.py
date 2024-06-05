"""TreeExplainer class that uses the TreeSHAPIQ algorithm for
computing any-order Shapley Interactions for tree ensembles."""

import copy
import warnings
from typing import Any, Optional, Union

import numpy as np

from shapiq.explainer._base import Explainer
from shapiq.interaction_values import InteractionValues

from .treeshapiq import TreeModel, TreeSHAPIQ
from .validation import validate_tree_model


class TreeExplainer(Explainer):
    """
    The explainer for tree-based models using the TreeSHAP-IQ algorithm.
    For details, refer to `Muschalik et al. (2024) <https://doi.org/10.48550/arXiv.2401.12069>`_.

    TreeSHAP-IQ is an algorithm for computing Shapley Interaction values for tree-based models.
    It is based on the Linear TreeSHAP algorithm by `Yu et al. (2022) <https://doi.org/10.48550/arXiv.2209.08192>`_,
    but extended to compute Shapley Interaction values up to a given order. TreeSHAP-IQ needs to
    visit each node only once and makes use of polynomial arithmetic to compute the Shapley
    Interaction values efficiently.

    Args:
        model: A tree-based model to explain.
        max_order: The maximum interaction order to be computed. An interaction order of ``1``
            corresponds to the Shapley value. Any value higher than ``1`` computes the Shapley
            interaction values up to that order. Defaults to ``2``.
        min_order: The minimum interaction order to be computed. Defaults to ``1``.
        index: The type of interaction to be computed. It can be one of
            ``["k-SII", "SII", "STII", "FSII", "BII", "SV"]``. All indices apart from ``"BII"`` will
            reduce to the ``"SV"`` (Shapley value) for order 1. Defaults to ``"k-SII"``.
        class_label: The class label of the model to explain.
    """

    def __init__(
        self,
        model: Union[dict, TreeModel, Any],
        max_order: int = 2,
        min_order: int = 1,
        index: str = "k-SII",
        class_label: Optional[int] = None,
        **kwargs,
    ) -> None:

        super().__init__(model)

        if index == "SV" and max_order > 1:
            warnings.warn("For index='SV' the max_order is set to 1.")
            max_order = 1
        elif max_order == 1 and index != "SV":
            warnings.warn("For max_order=1 the index is set to 'SV'.")
            index = "SV"

        # validate and parse model
        validated_model = validate_tree_model(model, class_label=class_label)
        self._trees: list[TreeModel] = copy.deepcopy(validated_model)
        if not isinstance(self._trees, list):
            self._trees = [self._trees]
        self._n_trees = len(self._trees)

        self._max_order: int = max_order
        self._min_order: int = min_order
        self._class_label: Optional[int] = class_label

        # setup explainers for all trees
        self._treeshapiq_explainers: list[TreeSHAPIQ] = [
            TreeSHAPIQ(model=_tree, max_order=self._max_order, index=index) for _tree in self._trees
        ]

        # TODO: for the current implementation this is correct for other trees this may vary
        self.baseline_value = sum(
            [treeshapiq.empty_prediction for treeshapiq in self._treeshapiq_explainers]
        )

    def explain(self, x: np.ndarray) -> InteractionValues:
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
        return final_explanation
