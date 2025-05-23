"""Functions for converting lightgbm decision trees to the format used by shapiq."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from shapiq.explainer.tree.base import TreeModel

if TYPE_CHECKING:
    from shapiq.utils.custom_types import Model


def convert_lightgbm_booster(
    tree_booster: Model,
    class_label: int | None = None,
) -> list[TreeModel]:
    """Transforms models from the ``lightgbm`` package to the format used by ``shapiq``.

    Args:
        tree_booster: The lightgbm booster to convert.
        class_label: The class label of the model to explain. Only used for multiclass
            classification models. Defaults to ``0``.

    Returns:
        The converted lightgbm booster.

    """
    # https://github.com/shap/shap/blob/77e92c3c110e816b768a0ec2acfbf4cc08ee13db/shap/explainers/_tree.py#L1079
    scaling = 1.0
    booster_df = tree_booster.trees_to_dataframe()
    # probabilities are hard and not implemented in shap / lightgbm, see
    # https://stackoverflow.com/q/63490533
    # https://stackoverflow.com/q/41433209
    convert_feature_str_to_int = {k: v for v, k in enumerate(tree_booster.feature_name())}
    if tree_booster.params["objective"] == "multiclass":
        # choose only trees for the selected class (lightgbm grows n_estimators*n_class trees)
        n_class = tree_booster.num_model_per_iteration()
        if class_label is None:
            class_label = 0
        idc = booster_df["tree_index"] % n_class == class_label
        booster_df = booster_df[idc]

    booster_df["split_feature"] = (
        booster_df["split_feature"].replace(convert_feature_str_to_int).infer_objects(copy=False)
    )

    return [
        _convert_lightgbm_tree_as_df(tree_df=tree_df, output_type="raw", scaling=scaling)
        for _, tree_df in booster_df.groupby("tree_index")
    ]


def _convert_lightgbm_tree_as_df(
    tree_df: Model,
    output_type: Literal["raw", "probability"] = "raw",
    scaling: float = 1.0,
) -> TreeModel:
    """Convert a lightgbm decision tree to the format used by shapiq.

    Args:
        tree_df: The lightgbm decision tree model formatted as a data frame.
        output_type: The type of output to be used. Can be one of ``["raw", "probability"]``.
            Defaults to ``"raw"``.
        scaling: The scaling factor for the tree values.

    Returns:
        The converted decision tree model.

    """
    convert_node_str_to_int = {k: v for v, k in enumerate(tree_df.node_index)}

    values = tree_df["value"].values * scaling
    return TreeModel(
        children_left=tree_df["left_child"]
        .replace(convert_node_str_to_int)
        .infer_objects(copy=False)
        .fillna(-1)
        .astype(int)
        .values,
        children_right=tree_df["right_child"]
        .replace(convert_node_str_to_int)
        .infer_objects(copy=False)
        .fillna(-1)
        .astype(int)
        .values,
        features=tree_df["split_feature"].fillna(-2).astype(int).values,
        thresholds=tree_df["threshold"].values,
        values=values,
        node_sample_weight=tree_df["count"].values,
        empty_prediction=None,  # compute empty prediction later
        original_output_type=output_type,  # not used
    )
