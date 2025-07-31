"""Functions for converting catboost decision trees to the format used by shapiq."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.explainer.tree.base import TreeModel
from shapiq.utils import safe_isinstance

if TYPE_CHECKING:
    from shapiq.typing import Model

SUPPORTED_CATBOOST_MODELS = {"catboost.CatBoostClassifier", "catboost.CatBoostRegressor"}


def convert_catboost(
    tree_model: Model,
    class_label: int | None = None,
) -> list[TreeModel]:
    """Transforms models from the catboost package to the format used by shapiq.

     Note: part of this implementation is taken and adapted from the shap package, where it can be found in shap/explainers/_tree.py.

    Args:
        tree_model: The catboost model to convert.
        class_label: The class label of the model to explain. Only used for classification models.

    Returns:
        The converted catboost model.

    """
    output_type = "raw"
    model_type = "undefined"

    if safe_isinstance(tree_model, "catboost.CatBoostClassifier"):
        model_type = "classifier"
    elif safe_isinstance(tree_model, "catboost.CatBoostRegressor"):
        model_type = "regressor"
    else:
        msg = f"Unsupported model type. Supported mode types are {SUPPORTED_CATBOOST_MODELS}"
        raise ValueError(msg)

    # workaround to get the single trees in the ensemble
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "model.json"
        tree_model.save_model(tmp_file, format="json")
        with Path.open(tmp_file, encoding="utf-8") as fh:
            loaded_cb_model = json.load(fh)

    num_trees = len(loaded_cb_model["oblivious_trees"])

    # determine number of classes or set to 1 for regression
    if model_type == "classifier":
        num_classes = len(loaded_cb_model["model_info"]["class_params"]["class_names"])
    elif model_type == "regressor":
        num_classes = 1

    trees = []
    for tree_index in range(num_trees):
        # get the values for all nodes
        leaf_values_json = loaded_cb_model["oblivious_trees"][tree_index]["leaf_values"]
        node_values = [0] * (len(leaf_values_json) - num_classes) + leaf_values_json
        node_values = np.array(node_values)
        total_nodes = int(len(node_values) / num_classes)
        node_values = node_values.reshape((-1, num_classes))  # reshape to match number of classes

        # get the children
        children_left = list(range(1, total_nodes, 2))
        children_left += [-1] * (total_nodes - len(children_left))
        children_right = list(range(2, total_nodes + 1, 2))
        children_right += [-1] * (total_nodes - len(children_right))

        # add a weight to each node
        leaf_weights_json = loaded_cb_model["oblivious_trees"][tree_index]["leaf_weights"]
        leaf_weights = [0] * (len(leaf_weights_json) - 1) + leaf_weights_json
        leaf_weights[0] = sum(leaf_weights_json)
        for index in range(len(leaf_weights_json) - 2, 0, -1):
            # each node weight is the sum of its children
            leaf_weights[index] = leaf_weights[2 * index + 1] + leaf_weights[2 * index + 2]

        # split features and borders from leafs to the root
        split_features_index_json = []
        borders_json = []

        for elem in loaded_cb_model["oblivious_trees"][tree_index]["splits"]:
            split_type = elem.get("split_type")
            if split_type == "FloatFeature":
                split_feature_index = elem.get("float_feature_index")
                borders_json.append(elem["border"])
            elif split_type == "OneHotFeature":
                split_feature_index = elem.get("cat_feature_index")
                borders_json.append(elem["value"])
            else:
                split_feature_index = elem.get("ctr_target_border_idx")
                borders_json.append(elem["border"])
            split_features_index_json.append(split_feature_index)

        split_features_index = []
        for counter, feature_index in enumerate(split_features_index_json[::-1]):
            split_features_index += [feature_index] * (2**counter)
        split_features_index += [-2] * (total_nodes - len(split_features_index))

        borders = []
        for counter, border in enumerate(borders_json[::-1]):
            borders += [border] * (2**counter)
        borders += [0] * (total_nodes - len(borders))

        if model_type == "classifier" and class_label is not None:
            class_label = 0

        # make probabilities
        if class_label is not None:
            row_sums = np.sum(node_values, axis=1, keepdims=True)
            zero_mask = row_sums == 0
            normalized = np.divide(node_values, row_sums, where=~zero_mask)  # avoid division by 0
            node_values = normalized[:, class_label]
            node_values[zero_mask.flatten()] = 0

            output_type = "probability"

        trees.append(
            TreeModel(
                children_left=np.array(children_left),
                children_right=np.array(children_right),
                features=np.array(split_features_index),
                thresholds=np.array(borders),
                values=node_values,
                node_sample_weight=np.array(leaf_weights),
                empty_prediction=None,  # compute empty prediction later
                original_output_type=output_type,
            )
        )

    return trees
