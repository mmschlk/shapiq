"""This module contains the TreeExplainer class making use of the TreeSHAPIQ algorithm for
computing any-order Shapley Interactions for tree ensembles."""

import copy
from typing import Any, Optional, Union

import numpy as np

from shapiq.explainer._base import Explainer
from shapiq.interaction_values import InteractionValues

from .treeshapiq import TreeModel, TreeSHAPIQ
from .validation import validate_tree_model


class TreeExplainer(Explainer):
    def __init__(
        self,
        model: Union[dict, TreeModel, Any],
        max_order: int = 2,
        min_order: int = 1,
        interaction_type: str = "k-SII",
        class_label: Optional[int] = None,
        **kwargs,
    ) -> None:

        super().__init__(model)

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
            TreeSHAPIQ(model=_tree, max_order=self._max_order, interaction_type=interaction_type)
            for _tree in self._trees
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
