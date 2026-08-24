"""Conversion functions for the tree explainer implementation."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from .base import TreeModel
from .conversion import convert_tree_model

if TYPE_CHECKING:
    from shapiq.typing import Model

SUPPORTED_MODELS = {
    "sklearn.tree.DecisionTreeRegressor",
    "sklearn.tree._classes.DecisionTreeRegressor",
    "sklearn.tree.DecisionTreeClassifier",
    "sklearn.tree._classes.DecisionTreeClassifier",
    "sklearn.ensemble.RandomForestClassifier",
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.ensemble.ExtraTreesClassifier",
    "sklearn.ensemble._forest.ExtraTreesClassifier",
    "sklearn.ensemble.RandomForestRegressor",
    "sklearn.ensemble._forest.RandomForestRegressor",
    "sklearn.ensemble.ExtraTreesRegressor",
    "sklearn.ensemble._forest.ExtraTreesRegressor",
    "sklearn.ensemble.IsolationForest",
    "sklearn.ensemble._iforest.IsolationForest",
    "sklearn.ensemble.GradientBoostingClassifier",
    "sklearn.ensemble._gb.GradientBoostingClassifier",
    "sklearn.ensemble.GradientBoostingRegressor",
    "sklearn.ensemble._gb.GradientBoostingRegressor",
    "sklearn.ensemble.HistGradientBoostingClassifier",
    "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier",
    "sklearn.ensemble.HistGradientBoostingRegressor",
    "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor",
    "lightgbm.sklearn.LGBMRegressor",
    "lightgbm.sklearn.LGBMClassifier",
    "lightgbm.basic.Booster",
    "xgboost.sklearn.XGBRegressor",
    "xgboost.sklearn.XGBClassifier",
    "catboost.core.CatBoost",
    "catboost.core.CatBoostRegressor",
    "catboost.core.CatBoostClassifier",
}


def validate_tree_model(
    model: Model,
    class_label: int | None = None,
) -> list[TreeModel]:
    """Validate the model and return its trees in the unified internal format.

    Accepts a single :class:`~shapiq.tree.base.TreeModel`, a list of ``TreeModel`` objects, a
    raw ``dict`` matching the ``TreeModel`` constructor, or any library-native model supported by
    :func:`~shapiq.tree.conversion.convert_tree_model` (scikit-learn, XGBoost, LightGBM, CatBoost).

    Args:
        model: The model to validate.
        class_label: The class label of the model to explain. Only used for classification models.

    Returns:
        The validated trees as a list of :class:`~shapiq.tree.base.TreeModel` instances. Single-tree
        inputs are normalized to a one-item list. The returned trees are owned by the caller:
        ``TreeModel`` inputs are deep-copied, so explainers may mutate them (e.g. via
        :meth:`~shapiq.tree.base.TreeModel.reduce_feature_complexity`) without affecting the
        original model.

    Raises:
        TypeError: If the model type is not supported (raised from the underlying
            ``NotImplementedError`` of :func:`~shapiq.tree.conversion.convert_tree_model`), or if
            a list input contains elements that are not ``TreeModel`` instances.
    """
    tree_model = []
    # direct returns for base tree models and dict as model
    # tree model (is already in the correct format); copied so the caller's object stays untouched
    if type(model).__name__ == "TreeModel":
        tree_model = [copy.deepcopy(model)]
    # list of tree models is copied element-wise for the same reason
    elif type(model).__name__ == "list":
        if not all(type(tree).__name__ == "TreeModel" for tree in model):
            msg = "All elements of a list model input must be TreeModel instances."
            raise TypeError(msg)
        tree_model = [copy.deepcopy(tree) for tree in model]
    # dict as model is parsed to TreeModel (the dict needs to have the correct format and names)
    elif type(model).__name__ == "dict":
        tree_model = [TreeModel(**model)]
    else:
        try:
            result = convert_tree_model(model, class_label=class_label)
        except NotImplementedError as e:
            msg = f"Model type {type(model)} is not supported."
            raise TypeError(msg) from e
        tree_model = result if isinstance(result, list) else [result]
    return tree_model
