"""This module contains the TreeExplainer class making use of the TreeSHAPIQ algorithm for
computing any-order Shapley Interactions for tree ensembles."""
import copy
from typing import Any, Union

import numpy as np
from approximator._interaction_values import InteractionValues
from explainer.treeshap_iq._validate import _validate_model
from treeshap_iq import TreeModel, TreeSHAPIQ

from shapiq.explainer._base import Explainer


class TreeExplainer(Explainer):
    def __init__(
        self,
        model: Union[dict, TreeModel, Any],
        max_order: int = 2,
        min_order: int = 1,
        verbose: bool = False,
    ) -> None:
        # validate and parse model
        validated_model = _validate_model(model)  # the parsed and validated model

        self._trees: Union[TreeModel, list[TreeModel]] = copy.deepcopy(validated_model)
        if not isinstance(self._trees, list):
            self._trees = [self._trees]
        self._n_trees = len(self._trees)

        self._max_order = max_order
        self._min_order = min_order

        # setup explainers for all trees
        self._treeshapiq_explainers: list[TreeSHAPIQ] = [
            TreeSHAPIQ(model=_tree, max_order=self._max_order, interaction_type="SII")
            for _tree in self._trees
        ]

    def explain(self, x_explain: np.ndarray) -> InteractionValues:
        interaction_values: list[InteractionValues] = []
        for explainer in self._treeshapiq_explainers:
            interaction_values.append(explainer.explain(x_explain))
        if self._n_trees > 1:
            raise NotImplementedError("Currently only a single tree is usable.")
        return interaction_values[0]
