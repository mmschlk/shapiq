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
    """
    predictors = tree_model._predictors  # noqa: SLF001  # ty: ignore[unresolved-attribute]
    tree_column = 0
    if tree_model.n_trees_per_iteration_ > 1:  # ty: ignore[unresolved-attribute]
        tree_column = 1 if class_label is None else class_label
    baseline = tree_model._baseline_prediction  # noqa: SLF001  # ty: ignore[unresolved-attribute]
    offset = float(np.asarray(baseline).ravel()[tree_column]) / len(predictors)

    # With categorical features, sklearn routes the input through an internal
    # ColumnTransformer that orders the CATEGORICAL COLUMNS FIRST and ordinal-encodes their
    # raw values; the tree predictors live in that transformed feature space. The internal
    # TreeModel operates on the untransformed input, so feature indices must be mapped back
    # to the original columns and encoded category codes back to raw category values.
    trans_to_orig = None
    raw_categories = None
    known_cat_bitsets = None
    f_idx_map = None
    if any(np.any(iteration[tree_column].nodes["is_categorical"]) for iteration in predictors):
        bin_mapper = tree_model._bin_mapper  # noqa: SLF001  # ty: ignore[unresolved-attribute]
        known_cat_bitsets, f_idx_map = bin_mapper.make_known_categories_bitsets()
        preprocessor = getattr(tree_model, "_preprocessor", None)
        if preprocessor is not None:
            is_cat = np.asarray(
                tree_model.is_categorical_,  # ty: ignore[unresolved-attribute]
                dtype=bool,
            )
            trans_to_orig = np.concatenate([np.flatnonzero(is_cat), np.flatnonzero(~is_cat)])
            raw_categories = list(preprocessor.named_transformers_["encoder"].categories_)
    return [
        _convert_hist_tree_predictor(
            iteration[tree_column],
            offset,
            trans_to_orig=trans_to_orig,
            raw_categories=raw_categories,
            known_cat_bitsets=known_cat_bitsets,
            f_idx_map=f_idx_map,
        )
        for iteration in predictors
    ]


def _bitset_to_categories(bitset: np.ndarray) -> np.ndarray:
    """Decode one of sklearn's ``(8,) uint32`` category bitsets into sorted category codes."""
    words = np.asarray(bitset, dtype=np.uint32)
    # Obtain set bits by shifting each word and masking with 1.
    bits = (words[:, None] >> np.arange(32, dtype=np.uint32)) & 1
    return np.flatnonzero(bits.ravel()).astype(np.int64)


def _hist_node_stored_set(
    nodes: np.ndarray,
    node_id: int,
    raw_left_cat_bitsets: np.ndarray,
    raw_categories: list[np.ndarray] | None,
    known_cat_bitsets: np.ndarray | None,
    f_idx_map: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    """Build the stored category set of one categorical Hist node in raw value space.

    sklearn's routing is three-way: left-bitset members go left, other *known* categories go
    right, and everything else (NaN, unknown, negative) goes to the missing branch. The
    internal binary rule (in set -> left, else -> right, NaN -> missing child) captures this
    exactly by always enumerating the side *opposite* the missing branch, which is finite:

    - missing branch is the right child: store the left bitset; every non-member (including
      unknown categories) falls to the right = missing side.
    - missing branch is the left child: store the *right*-routed set (known categories minus
      the left bitset) and swap the children; every non-member (left-bitset members, unknown
      categories, negatives) then falls to the stored right = actual left = missing side.

    Returns:
        A 2-tuple ``(stored_set, swap_children)``.

    Raises:
        ValueError: If the model was trained on non-integer raw category values, which the
            internal integer-based categorical representation cannot express.
    """
    feature_idx = nodes["feature_idx"][node_id]
    if known_cat_bitsets is None or f_idx_map is None:
        msg = "known_cat_bitsets and f_idx_map are required for categorical Hist trees."
        raise ValueError(msg)
    codes = _bitset_to_categories(raw_left_cat_bitsets[nodes["bitset_idx"][node_id]])
    swap_children = bool(nodes["missing_go_to_left"][node_id])
    if swap_children:
        known = _bitset_to_categories(known_cat_bitsets[f_idx_map[feature_idx]])
        codes = np.setdiff1d(known, codes)
    if raw_categories is not None:
        # Bitsets are in the ordinal-encoded space; translate codes to raw values via the
        # encoder's category list (categorical features occupy the first transformed
        # columns, in original order, matching ``encoder.categories_``).
        raw = np.asarray(raw_categories[feature_idx], dtype=np.float64)[codes]
        raw = raw[np.isfinite(raw)]
        if not np.all(raw == np.floor(raw)):
            msg = (
                "HistGradientBoosting models with non-integer raw category values are not "
                "supported by the internal categorical tree representation."
            )
            raise ValueError(msg)
        return np.sort(raw.astype(np.int64)), swap_children
    return codes, swap_children


def _convert_hist_tree_predictor(
    predictor: object,
    offset: float,
    *,
    trans_to_orig: np.ndarray | None = None,
    raw_categories: list[np.ndarray] | None = None,
    known_cat_bitsets: np.ndarray | None = None,
    f_idx_map: np.ndarray | None = None,
) -> TreeModel:
    """Convert a single Hist ``TreePredictor`` to a ``TreeModel``.

    Categorical splits are decoded from the predictor's ``raw_left_cat_bitsets`` into the
    internal CSR representation (category in set -> left child), translated back into the
    original (untransformed) feature space: feature indices are remapped via
    ``trans_to_orig`` and encoded category codes back to raw values via ``raw_categories``.
    Nodes whose missing branch is the left child store the (finite) right-routed category
    set with swapped children instead, so unknown categories fall through to the missing
    branch exactly like in sklearn (see :func:`_hist_node_stored_set`).

    Residual, documented caveat: fractional feature values at categorical splits are
    truncated (``int(5.7) == 5``) instead of being treated as unknown categories like
    sklearn's ordinal encoder does.

    Args:
        predictor: The ``TreePredictor``
            (``sklearn.ensemble._hist_gradient_boosting.predictor``).
        offset: An additive offset applied to all leaf values.
        trans_to_orig: Mapping from transformed to original feature indices (``None`` when
            the model has no encoding preprocessor).
        raw_categories: Per categorical feature, the raw category values in code order
            (``encoder.categories_``); ``None`` without the preprocessor.
        known_cat_bitsets: Model-level known-category bitsets from
            ``_bin_mapper.make_known_categories_bitsets()``; only needed for categorical trees.
        f_idx_map: Feature-index map accompanying ``known_cat_bitsets``.

    Returns:
        The tree converted to the internal ``TreeModel`` format.
    """
    nodes = predictor.nodes  # ty: ignore[unresolved-attribute]
    is_leaf = nodes["is_leaf"].astype(bool)
    children_left = np.where(is_leaf, -1, nodes["left"].astype(np.int64))
    children_right = np.where(is_leaf, -1, nodes["right"].astype(np.int64))
    children_missing = np.where(
        nodes["missing_go_to_left"].astype(bool), children_left, children_right
    )
    values = np.where(is_leaf, nodes["value"] + offset, 0.0)
    features = nodes["feature_idx"].astype(np.int64)
    if trans_to_orig is not None:
        features = trans_to_orig[features]

    thresholds = nodes["num_threshold"].astype(np.float64)
    cat_values_arr = None
    cat_start_arr = None
    cat_size_arr = None
    categorical_nodes = np.flatnonzero(nodes["is_categorical"])
    if len(categorical_nodes) > 0:
        raw_left_cat_bitsets = predictor.raw_left_cat_bitsets  # ty: ignore[unresolved-attribute]
        n_nodes = len(nodes)
        cat_start = np.zeros(n_nodes, dtype=np.int64)
        cat_size = np.zeros(n_nodes, dtype=np.int64)
        left_sets: list[np.ndarray] = []
        total = 0
        for node_id in categorical_nodes:
            stored_set, swap_children = _hist_node_stored_set(
                nodes, node_id, raw_left_cat_bitsets, raw_categories, known_cat_bitsets, f_idx_map
            )
            if swap_children:
                # the stored set enumerates the right-routed categories; swapping the
                # children restores the internal in-set -> left convention and sends every
                # non-member to the missing (actual left) child. children_missing holds a
                # node id computed from the original orientation, so it is unaffected.
                children_left[node_id], children_right[node_id] = (
                    children_right[node_id],
                    children_left[node_id],
                )
            if len(stored_set) == 0:
                # no finite value may reach the stored left child (e.g. only the
                # missing-values bin routes to one side): encode as a numeric split nothing
                # satisfies (NaN keeps routing via children_missing)
                thresholds[node_id] = -np.inf
                continue
            cat_start[node_id] = total
            cat_size[node_id] = len(stored_set)
            total += len(stored_set)
            left_sets.append(stored_set)
        if left_sets:
            cat_values_arr = np.concatenate(left_sets)
            cat_start_arr = cat_start
            cat_size_arr = cat_size

    return TreeModel(
        children_left=children_left,
        children_right=children_right,
        children_missing=children_missing,
        features=features,
        thresholds=thresholds,
        values=values,
        node_sample_weight=nodes["count"].astype(np.float64),
        original_output_type="raw",
        cat_values=cat_values_arr,
        cat_start=cat_start_arr,
        cat_size=cat_size_arr,
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
