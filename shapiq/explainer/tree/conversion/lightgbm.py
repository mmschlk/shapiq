"""Functions for converting lightgbm decision trees to the format used by
shapiq."""

from typing import Optional

import numpy as np
import pandas as pd

from shapiq.utils import safe_isinstance
from shapiq.utils.types import Model

from ..base import TreeModel


def convert_lightgbm_booster(
    tree_model: Model,
    class_label: Optional[int] = None,
) -> list[TreeModel]:
    """Transforms a lightgbm model to the format used by shapiq.

    Args:
        tree_model: The lightgbm booster to convert.
        class_label: The class label of the model to explain. Only used for classification models.
            Defaults to 0.

    Returns:
        The converted random forest model.
    """
    scaling = 1.0 / tree_model.num_trees()
    df_booster = tree_model.trees_to_dataframe()
    convert_feature_str_to_int = {k: v for v, k in enumerate(tree_model.feature_name())}
    # pandas can't chill https://stackoverflow.com/q/77900971
    with pd.option_context('future.no_silent_downcasting', True):
        df_booster['split_feature'] = df_booster['split_feature']\
            .replace(convert_feature_str_to_int).infer_objects(copy=False)
    return [
        convert_lightgbm_tree(tree, scaling=scaling, class_label=class_label)
        for i, tree in df_booster.groupby("tree_index")
    ]


def convert_lightgbm_tree(
    tree_model: Model, class_label: Optional[int] = None, scaling: float = 1.0
) -> TreeModel:
    """Convert a lightgbm decision tree to the format used by shapiq.

    Args:
        tree_model: The lightgbm decision tree model to convert.
        class_label: The class label of the model to explain. Only used for classification models.
            Defaults to 0.
        scaling: The scaling factor for the tree values.

    Returns:
        The converted decision tree model.
    """
    output_type = "raw"
    # tree_values = tree_model.tree_.value.copy() * scaling
    # set class label if not given and model is a classifier
    # if safe_isinstance(tree_model, "sklearn.tree.DecisionTreeClassifier") and class_label is None:
    #     class_label = 1

    # if class_label is not None:
    #     # turn node values into probabilities
    #     if len(tree_values.shape) == 3:
    #         tree_values = tree_values[:, 0, :]
    #     tree_values = tree_values / np.sum(tree_values, axis=1, keepdims=True)
    #     tree_values = tree_values[:, class_label]
    #     output_type = "probability"
    convert_node_str_to_int = {k: v for v, k in enumerate(tree_model.node_index)}

    # pandas can't chill https://stackoverflow.com/q/77900971
    with pd.option_context('future.no_silent_downcasting', True):
        return TreeModel(
            children_left=tree_model['left_child']\
                .replace(convert_node_str_to_int).infer_objects(copy=False).fillna(-1).astype(int).values,
            children_right=tree_model['right_child']\
                .replace(convert_node_str_to_int).infer_objects(copy=False).fillna(-1).astype(int).values,
            features=tree_model['split_feature'].fillna(-2).astype(int).values,
            thresholds=tree_model['threshold'].values,
            values=tree_model['value'].values,
            node_sample_weight=tree_model['count'].values,
            empty_prediction=None,  # compute empty prediction later
            original_output_type=output_type,
        )