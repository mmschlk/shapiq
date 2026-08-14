"""Conversion utilities for scikit-learn tree models to the unified internal tree format."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.ensemble._iforest import (
    _average_path_length,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)

from shapiq.tree.base import TreeModel

from .common import register

if TYPE_CHECKING:
    from sklearn.tree._tree import Tree  # ty: ignore[unresolved-import]


def convert_sklearn_tree(
    model: DecisionTreeRegressor | DecisionTreeClassifier,
    class_label: int | None = None,
    scaling: float = 1.0,
    offset: float = 0.0,
) -> TreeModel:
    """Convert a sklearn DecisionTreeRegressor or DecisionTreeClassifier to the internal tree format.

    For classifiers with ``class_label`` set, leaf values are converted to class probabilities
    for the specified class.

    Args:
        model: The sklearn ``DecisionTreeRegressor`` or ``DecisionTreeClassifier`` to convert.
        class_label: The class index whose probability is extracted as the leaf value.
            Only used for ``DecisionTreeClassifier`` models. Defaults to ``None``.
        scaling: A multiplicative scaling factor applied to all leaf values. Defaults to ``1.0``.
        offset: An additive offset applied to all node values after scaling. Used to fold a
            constant baseline (e.g. the ``init_`` prediction of a gradient boosting ensemble)
            into the tree. Defaults to ``0.0``.

    Returns:
        The tree converted to the internal ``TreeModel`` format.
    """
    tree = model.tree_
    tree_values = tree.value.copy()
    original_output_type = "raw"
    if isinstance(model, DecisionTreeClassifier) and class_label is None:
        class_label = 1
    if isinstance(model, DecisionTreeClassifier) and class_label is not None:
        # turn node values into probabilities
        if len(tree_values.shape) == 3:
            tree_values = tree_values[:, 0, :]
        tree_values = tree_values / np.sum(tree_values, axis=1, keepdims=True)
        tree_values = tree_values[:, class_label]
        original_output_type = "probability"
    tree_values = tree_values.flatten() * scaling + offset
    children_missing = np.where(tree.missing_go_to_left, tree.children_left, tree.children_right)
    return TreeModel(
        children_left=tree.children_left,
        children_right=tree.children_right,
        children_missing=children_missing,
        features=tree.feature,
        thresholds=tree.threshold,
        values=tree_values,
        node_sample_weight=tree.weighted_n_node_samples,
        original_output_type=original_output_type,
    )


def convert_extra_tree(
    tree_model: ExtraTreeClassifier | ExtraTreeRegressor,
    tree_features: np.ndarray,
    class_label: int | None = None,  # noqa: ARG001
    scaling: float = 1.0,
    **_: object,
) -> TreeModel:
    """Convert a scikit-learn ExtraTree to the internal tree format used by shapiq.

    Node values are recalculated via :func:`extra_tree_traversal` using the average-path-length
    correction that makes per-node contributions additive across an ensemble.  Feature indices are
    remapped so that each tree can reference the global feature space even when trained on a feature
    subset.

    Args:
        tree_model: The scikit-learn ``ExtraTreeClassifier`` or ``ExtraTreeRegressor`` to convert.
        tree_features: A 1-D integer array mapping each local tree feature index to its
            corresponding global feature index.
        class_label: The class index whose probability is extracted as the leaf value. Only here for API consistency with other converters; ignored since ExtraTrees don't support multi-class outputs.
        scaling: A multiplicative scaling factor applied to all leaf values. Defaults to ``1.0``.

    Returns:
        The tree converted to the internal ``TreeModel`` format.
    """
    output_type = "raw"
    features_updated, values_updated = extra_tree_traversal(
        tree_model.tree_,
        tree_features,
        normalize=False,
        scaling=1.0,
    )
    values_updated = values_updated * scaling
    values_updated = values_updated.flatten()
    tree = tree_model.tree_
    children_missing = np.where(tree.missing_go_to_left, tree.children_left, tree.children_right)
    return TreeModel(
        children_left=tree.children_left,
        children_right=tree.children_right,
        children_missing=children_missing,
        features=features_updated,
        thresholds=tree.threshold,
        values=values_updated,
        node_sample_weight=tree.weighted_n_node_samples,
        empty_prediction=None,  # pyright: ignore[reportArgumentType] compute empty prediction later
        original_output_type=output_type,
        decision_type="<=",
    )


def extra_tree_traversal(
    tree: Tree,
    tree_features: np.ndarray,
    *,
    normalize: bool = False,
    scaling: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Traverse an ExtraTree and recalculate node values using the average path length.

    Recursively computes the expected path length for every node by combining the depth at which
    the node sits with the average path length of its subtree.  This correction is required to
    make isolation-forest scores additive across trees.

    Args:
        tree: The internal scikit-learn ``Tree`` object to traverse.
        tree_features: A 1-D integer array mapping each local tree feature index to its
            corresponding global feature index.
        normalize: If ``True``, row-normalise the corrected node values so they sum to 1
            across features. Defaults to ``False``.
        scaling: A multiplicative scaling factor applied to all corrected values.
            Defaults to ``1.0``.

    Returns:
        A 2-tuple ``(features, values)`` where ``features`` is an integer array of remapped
        global feature indices and ``values`` is a float array of the corrected node values.
    """
    features = tree.feature.copy()
    corrected_values = tree.value.copy()

    def _recalculate_value(tree: Tree, i: int, level: int = 0) -> float:
        if tree.children_left[i] == -1 and tree.children_right[i] == -1:
            value = level + _average_path_length(np.array([tree.n_node_samples[i]]))[0]
            corrected_values[i, 0] = value
            return value * tree.n_node_samples[i]
        value_left = _recalculate_value(tree, tree.children_left[i], level + 1)
        value_right = _recalculate_value(tree, tree.children_right[i], level + 1)
        corrected_values[i, 0] = (value_left + value_right) / tree.n_node_samples[i]
        return value_left + value_right

    _recalculate_value(tree, 0, 0)
    if normalize:
        corrected_values = (corrected_values.T / corrected_values.sum(1)).T
    corrected_values = corrected_values * scaling
    # re-number the features if each tree gets a different set of features
    features = np.where(features >= 0, tree_features[features], features)
    return features, corrected_values


def convert_isolation_forest_tree(
    tree_model: IsolationForest,
    class_label: int | None = None,  # noqa: ARG001
    **_: object,
) -> list[TreeModel]:
    """Convert a scikit-learn IsolationForest to the internal tree format used by shapiq.

    Each constituent ``ExtraTree`` estimator is converted individually via
    :func:`convert_extra_tree`.  The scaling factor is set to ``1 / n_estimators`` so that the
    per-tree contributions sum to the ensemble score.

    Args:
        tree_model: The fitted scikit-learn ``IsolationForest`` to convert.
        class_label: Ignored; present for API consistency with other converters.
        **_: Ignored; present for API consistency with other converters.

    Returns:
        A list of ``TreeModel`` instances, one per estimator in the forest.
    """
    scaling = 1.0 / len(tree_model.estimators_)
    return [
        convert_extra_tree(
            estimator,
            tree_features,
            scaling=scaling,
        )
        for estimator, tree_features in zip(
            tree_model.estimators_,
            tree_model.estimators_features_,
            strict=False,
        )
    ]


def convert_random_forest_tree(
    tree_model: RandomForestClassifier | RandomForestRegressor,
    class_label: int | None = None,
    **_: object,
) -> list[TreeModel]:
    """Convert a scikit-learn RandomForest to the internal tree format used by shapiq.

    Each constituent ``DecisionTree`` estimator is converted individually via
    :func:`convert_sklearn_tree`.  The scaling factor is set to ``1 / n_estimators`` so that the
    per-tree contributions average to the ensemble prediction.

    Args:
        tree_model: The fitted ``RandomForestClassifier`` or ``RandomForestRegressor`` to convert.
        class_label: The class index whose probability is extracted as the leaf value.
            Only used for ``RandomForestClassifier`` models. Defaults to ``None``.

    Returns:
        A list of ``TreeModel`` instances, one per estimator in the forest.
    """
    scaling = 1.0 / len(tree_model.estimators_)
    return [
        convert_sklearn_tree(
            tree_model.estimators_[i],
            class_label=class_label,
            scaling=scaling,
        )
        for i in range(len(tree_model.estimators_))
    ]


def convert_gradient_boosting_tree(
    tree_model: GradientBoostingClassifier | GradientBoostingRegressor,
    class_label: int | None = None,
    **_: object,
) -> list[TreeModel]:
    """Convert a scikit-learn GradientBoosting model to the internal tree format used by shapiq.

    Each constituent ``DecisionTree`` estimator is converted individually via
    :func:`convert_sklearn_tree`.  The scaling factor is set to ``learning_rate`` so that the
    per-tree contributions sum to the ensemble prediction.

    The raw prediction of the fitted ``init_`` estimator is a constant baseline added to the
    sum of the tree predictions.  It is distributed evenly across the trees as an additive
    offset (mirroring the XGBoost converter) so that the per-tree contributions sum to the
    ensemble's raw prediction.

    Args:
        tree_model: The fitted ``GradientBoostingClassifier`` or ``GradientBoostingRegressor``
            to convert.
        class_label: The class index whose trees are extracted for multiclass models.
            Defaults to ``None``, which selects class ``1``.

    Raises:
        ValueError: If the model was fitted with a custom ``init`` estimator, whose contribution
            depends on the input and therefore cannot be folded into a constant baseline.
    """
    scaling = tree_model.learning_rate
    n_estimators, n_classes = tree_model.estimators_.shape
    if n_classes > 1:
        if class_label is None:
            class_label = 1
        tree_column = class_label
    else:
        tree_column = 0
    offset = _gradient_boosting_init_offset(tree_model, tree_column) / n_estimators
    return [
        convert_sklearn_tree(
            tree_model.estimators_[i, tree_column],
            class_label=class_label,
            scaling=scaling,
            offset=offset,
        )
        for i in range(n_estimators)
    ]


def _gradient_boosting_init_offset(
    tree_model: GradientBoostingClassifier | GradientBoostingRegressor,
    class_label: int,
) -> float:
    """Return the raw prediction of the ``init_`` estimator of a GradientBoosting model.

    Args:
        tree_model: The fitted ``GradientBoostingClassifier`` or ``GradientBoostingRegressor``.
        class_label: The column of the raw init prediction to extract (``0`` for regression and
            binary classification).

    Returns:
        The constant raw-space prediction of the ``init_`` estimator for the selected class.

    Raises:
        ValueError: If the model was fitted with a custom ``init`` estimator, whose contribution
            depends on the input and therefore cannot be folded into a constant baseline.
    """
    if tree_model.init == "zero":
        return 0.0
    if tree_model.init is not None:
        msg = (
            "GradientBoosting models with a custom `init` estimator are not supported: the "
            "init contribution depends on the input and cannot be folded into a constant "
            "baseline. Refit with `init=None` (default) or `init='zero'`."
        )
        raise ValueError(msg)
    # the default init (DummyEstimator) predicts a constant, so any input works here
    n_features = tree_model.n_features_in_  # ty: ignore[unresolved-attribute]
    raw_init = tree_model._raw_predict_init(np.zeros((1, n_features)))  # noqa: SLF001
    return float(raw_init[0, class_label])


def convert_hist_gradient_boosting_tree(
    tree_model: HistGradientBoostingClassifier | HistGradientBoostingRegressor,
    class_label: int | None = None,
    **_: object,
) -> list[TreeModel]:
    """Convert a scikit-learn HistGradientBoosting model to the internal tree format used by shapiq.

    Hist models store their trees as ``TreePredictor`` objects with a structured node array
    instead of sklearn ``Tree`` objects.  The leaf values already include the learning rate
    (shrinkage is applied during fitting), so no scaling is needed.  The constant baseline
    prediction of the model is distributed evenly across the trees as an additive offset so
    that the per-tree contributions sum to the ensemble's raw prediction.

    Args:
        tree_model: The fitted ``HistGradientBoostingClassifier`` or
            ``HistGradientBoostingRegressor`` to convert.
        class_label: The class index whose trees are extracted for multiclass models.
            Defaults to ``None``, which selects class ``1``.

    Returns:
        A list of ``TreeModel`` instances, one per boosting iteration for the selected class.

    Raises:
        ValueError: If the model contains categorical splits, which the internal tree format
            does not support.
    """
    predictors = tree_model._predictors  # noqa: SLF001  # ty: ignore[unresolved-attribute]
    tree_column = 0
    if tree_model.n_trees_per_iteration_ > 1:  # ty: ignore[unresolved-attribute]
        tree_column = 1 if class_label is None else class_label
    baseline = tree_model._baseline_prediction  # noqa: SLF001  # ty: ignore[unresolved-attribute]
    offset = float(np.asarray(baseline).ravel()[tree_column]) / len(predictors)
    return [
        _convert_hist_tree_predictor(iteration[tree_column].nodes, offset)
        for iteration in predictors
    ]


def _convert_hist_tree_predictor(nodes: np.ndarray, offset: float) -> TreeModel:
    """Convert the structured node array of a single Hist ``TreePredictor`` to a ``TreeModel``.

    Args:
        nodes: The structured node array of the ``TreePredictor``
            (``sklearn.ensemble._hist_gradient_boosting.predictor``).
        offset: An additive offset applied to all leaf values.

    Returns:
        The tree converted to the internal ``TreeModel`` format.

    Raises:
        ValueError: If the tree contains categorical splits, which the internal tree format
            does not support.
    """
    if np.any(nodes["is_categorical"]):
        msg = (
            "HistGradientBoosting models with categorical splits are not supported. "
            "Refit the model without `categorical_features`."
        )
        raise ValueError(msg)
    is_leaf = nodes["is_leaf"].astype(bool)
    children_left = np.where(is_leaf, -1, nodes["left"].astype(np.int64))
    children_right = np.where(is_leaf, -1, nodes["right"].astype(np.int64))
    children_missing = np.where(
        nodes["missing_go_to_left"].astype(bool), children_left, children_right
    )
    values = np.where(is_leaf, nodes["value"] + offset, 0.0)
    return TreeModel(
        children_left=children_left,
        children_right=children_right,
        children_missing=children_missing,
        features=nodes["feature_idx"].astype(np.int64),
        thresholds=nodes["num_threshold"].astype(np.float64),
        values=values,
        node_sample_weight=nodes["count"].astype(np.float64),
        original_output_type="raw",
    )


register(DecisionTreeRegressor, convert_sklearn_tree)
register(DecisionTreeClassifier, convert_sklearn_tree)
register(ExtraTreeRegressor, convert_extra_tree)
register(ExtraTreeClassifier, convert_extra_tree)
register(IsolationForest, convert_isolation_forest_tree)
register(RandomForestClassifier, convert_random_forest_tree)
register(RandomForestRegressor, convert_random_forest_tree)
register(ExtraTreesClassifier, convert_random_forest_tree)
register(ExtraTreesRegressor, convert_random_forest_tree)
register(GradientBoostingClassifier, convert_gradient_boosting_tree)
register(GradientBoostingRegressor, convert_gradient_boosting_tree)
register(HistGradientBoostingClassifier, convert_hist_gradient_boosting_tree)
register(HistGradientBoostingRegressor, convert_hist_gradient_boosting_tree)
