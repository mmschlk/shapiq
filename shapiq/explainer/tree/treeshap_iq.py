"""This module contains the tree explainer implementation."""
from typing import Any

import numpy as np

from approximator._interaction_values import InteractionValues
from explainer._base import Explainer
from explainer.tree._validate import _validate_model
from explainer.tree.conversion import TreeModel


class TreeExplainer(Explainer):
    """
    The explainer for tree-based models using the TreeSHAP-IQ algorithm.

    Args:
        model: The tree-based model to explain.
        background_data: The background data to use for the explainer.
    """

    def __init__(self, model: Any, background_data=None) -> None:
        validated_model: TreeModel = _validate_model(model)
        edge_representation = _get_edge_tree(validated_model)

    def explain(self, x_explain: np.ndarray) -> InteractionValues:
        pass
