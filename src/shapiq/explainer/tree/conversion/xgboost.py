"""Functions for converting xgboost decision trees to the format used by shapiq."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

from shapiq.explainer.tree.base import TreeModel

if TYPE_CHECKING:
    from shapiq.utils.custom_types import Model


def convert_xgboost_booster(
    tree_booster: Model,
    class_label: int | None = None,
) -> list[TreeModel]:
    """Transforms models from the ``xgboost`` package to the format used by ``shapiq``.

    Args:
        tree_booster: The xgboost booster to convert.
        class_label: The class label of the model to explain. Only used for multiclass
            classification models. Defaults to ``0``.

    Returns:
        The converted xgboost booster.

    """
    try:
        intercept = tree_booster.base_score
        if intercept is None:
            intercept = float(tree_booster.intercept_[0])
        tree_booster = tree_booster.get_booster()
    except AttributeError:
        intercept = 0.0
        warnings.warn(
            "The model does not have a valid base score. Setting the intercept to 0.0."
            "Baseline values of the interaction models might be different.",
            stacklevel=2,
        )

    # https://github.com/shap/shap/blob/77e92c3c110e816b768a0ec2acfbf4cc08ee13db/shap/explainers/_tree.py#L1992
    scaling = 1.0
    booster_df = tree_booster.trees_to_dataframe()

    if tree_booster.feature_names:
        feature_names = tree_booster.feature_names
    else:  # xgboost does not store default feature names
        feature_names = [f"f{i}" for i in range(tree_booster.num_features())]
    convert_feature_str_to_int = {k: v for v, k in enumerate(feature_names)}
    convert_feature_str_to_int["Leaf"] = -2
    booster_df.loc[:, "Feature"] = booster_df.loc[:, "Feature"].replace(convert_feature_str_to_int)

    if len(booster_df["Tree"].unique()) > tree_booster.num_boosted_rounds():
        # choose only trees for the selected class (xgboost grows n_estimators*n_class trees)
        # approximation for the number of classes in xgboost
        n_class = int(len(booster_df["Tree"].unique()) / tree_booster.num_boosted_rounds())
        if class_label is None:
            class_label = 0
        idc = booster_df["Tree"] % n_class == class_label
        booster_df = booster_df.loc[idc, :]

    n_trees = len(booster_df["Tree"].unique())
    intercept /= n_trees
    return [
        _convert_xgboost_tree_as_df(
            tree_df=tree_df,
            intercept=intercept,
            output_type="raw",
            scaling=scaling,
        )
        for _, tree_df in booster_df.groupby("Tree")
    ]


def _convert_xgboost_tree_as_df(
    tree_df: Model,
    intercept: float,
    output_type: Literal["raw", "probability"] = "raw",
    scaling: float = 1.0,
) -> TreeModel:
    """Convert a xgboost decision tree to the format used by shapiq.

    Args:
        tree_df: The xgboost decision tree model formatted as a data frame.
        intercept: The intercept of the model.
        output_type: The type of output to be used. Can be one of ``["raw", "probability"]``.
            Defaults to ``"raw"``.
        scaling: The scaling factor for the tree values.

    Returns:
        The converted decision tree model.

    """
    convert_node_str_to_int = {k: v for v, k in enumerate(tree_df["ID"])}

    values = tree_df["Gain"].values * scaling + intercept  # add intercept to all values
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
        values=values,  # values in non-leaf nodes are not used
        node_sample_weight=tree_df["Cover"].values,
        empty_prediction=None,
        original_output_type=output_type,
    )
