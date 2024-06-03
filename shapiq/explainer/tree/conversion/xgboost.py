"""Functions for converting xgboost decision trees to the format used by
shapiq."""

from typing import Optional

import numpy as np
import pandas as pd

from shapiq.utils.types import Model

from ..base import TreeModel


def convert_xgboost_booster(
    tree_booster: Model,
    class_label: Optional[int] = None,
) -> list[TreeModel]:
    """Transforms models from the ``xgboost`` package to the format used by ``shapiq``.

    Args:
        tree_booster: The xgboost booster to convert.
        class_label: The class label of the model to explain. Only used for multiclass
            classification models. Defaults to ``0``.

    Returns:
        The converted xgboost booster.
    """
    # https://github.com/shap/shap/blob/77e92c3c110e816b768a0ec2acfbf4cc08ee13db/shap/explainers/_tree.py#L1992
    scaling = 1.0
    booster_df = tree_booster.trees_to_dataframe()
    output_type = "raw"
    if len(booster_df["Tree"].unique()) > tree_booster.num_boosted_rounds():
        # choose only trees for the selected class (xgboost grows n_estimators*n_class trees)
        # approximation for the number of classes in xgboost
        n_class = int(len(booster_df["Tree"].unique()) / tree_booster.num_boosted_rounds())
        if class_label is None:
            class_label = 0
        idc = booster_df["Tree"] % n_class == class_label
        booster_df = booster_df.loc[idc, :]

    #
    if tree_booster.feature_names:
        feature_names = tree_booster.feature_names
    else:
        # xgboost does not store default feature names
        n_features = len(np.setdiff1d(np.unique(booster_df["Feature"]), "Leaf"))
        feature_names = [f"f{i}" for i in range(n_features)]
    convert_feature_str_to_int = {k: v for v, k in enumerate(feature_names)}
    convert_feature_str_to_int["Leaf"] = -2
    # pandas can't chill https://stackoverflow.com/q/77900971
    with pd.option_context("future.no_silent_downcasting", True):
        booster_df.loc[:, "Feature"] = booster_df.loc[:, "Feature"].replace(
            convert_feature_str_to_int
        )
    return [
        _convert_xgboost_tree_as_df(tree_df=tree_df, output_type=output_type, scaling=scaling)
        for _, tree_df in booster_df.groupby("Tree")
    ]


def _convert_xgboost_tree_as_df(
    tree_df: Model,
    output_type: str,
    scaling: float = 1.0,
) -> TreeModel:
    """Convert a xgboost decision tree to the format used by shapiq.

    Args:
        tree_df: The xgboost decision tree model formatted as a data frame.
        output_type: Either "raw" or "probability". Currently unused.
        scaling: The scaling factor for the tree values.

    Returns:
        The converted decision tree model.
    """
    convert_node_str_to_int = {k: v for v, k in enumerate(tree_df["ID"])}

    # pandas can't chill https://stackoverflow.com/q/77900971
    with pd.option_context("future.no_silent_downcasting", True):
        return TreeModel(
            children_left=tree_df["Yes"]
            .replace(convert_node_str_to_int)
            .infer_objects(copy=False)
            .fillna(-1)
            .astype(int)
            .values,
            children_right=tree_df["No"]
            .replace(convert_node_str_to_int)
            .infer_objects(copy=False)
            .fillna(-1)
            .astype(int)
            .values,
            features=tree_df["Feature"].values,
            thresholds=tree_df["Split"].values,
            values=tree_df["Gain"].values * scaling,  # values in non-leaf nodes are not used
            node_sample_weight=tree_df["Cover"].values,
            empty_prediction=None,
            original_output_type=output_type,
        )
