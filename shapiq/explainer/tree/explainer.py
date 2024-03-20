"""This module contains the TreeExplainer class making use of the TreeSHAPIQ algorithm for
computing any-order Shapley Interactions for tree ensembles."""
import copy
from typing import Any, Optional, Union

import numpy as np
from explainer._base import Explainer
from interaction_values import InteractionValues

from .treeshapiq import TreeModel, TreeSHAPIQ
from .validation import _validate_model


class TreeExplainer(Explainer):
    def __init__(
        self,
        model: Union[dict, TreeModel, Any],
        max_order: int = 2,
        min_order: int = 1,
        class_label: Optional[int] = None,
        output_type: str = "raw",
    ) -> None:
        # validate and parse model
        validated_model = _validate_model(model, class_label=class_label, output_type=output_type)
        self._trees: Union[TreeModel, list[TreeModel]] = copy.deepcopy(validated_model)
        if not isinstance(self._trees, list):
            self._trees = [self._trees]
        self._n_trees = len(self._trees)

        self._max_order: int = max_order
        self._min_order: int = min_order
        self._class_label: Optional[int] = class_label
        self._output_type: str = output_type

        # setup explainers for all trees
        self._treeshapiq_explainers: list[TreeSHAPIQ] = [
            TreeSHAPIQ(model=_tree, max_order=self._max_order, interaction_type="SII")
            for _tree in self._trees
        ]

    def explain(self, x_explain: np.ndarray) -> InteractionValues:
        # run treeshapiq for all trees
        interaction_values: list[InteractionValues] = []
        for explainer in self._treeshapiq_explainers:
            tree_explanation = explainer.explain(x_explain)
            interaction_values.append(tree_explanation)

        # combine the explanations for all trees
        final_explanation = interaction_values[0]
        if len(interaction_values) > 1:
            for i in range(1, len(interaction_values)):
                final_explanation += interaction_values[i]
        return final_explanation
