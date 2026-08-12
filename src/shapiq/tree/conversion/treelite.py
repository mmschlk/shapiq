from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from shapiq.tree.base import TreeModel

from .common import register

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import DTypeLike, NDArray

    type SklearnBoostingModel = (
        GradientBoostingRegressor
        | GradientBoostingClassifier
        | HistGradientBoostingRegressor
        | HistGradientBoostingClassifier
    )

# treelite's ``TreeNodeType`` enum, as stored in the per-tree ``node_type`` field.
_LEAF_NODE = 0
_CATEGORICAL_TEST_NODE = 2

# treelite's ``Operator`` enum, as stored in the per-node ``cmp`` field, mapped onto the split
# comparisons ``TreeModel.decision_function`` implements. ``0`` marks leaf nodes (no operator).
_OPERATOR_TO_DECISION_TYPE = {2: "<", 3: "<="}


class TreeliteAccessor(Protocol):
    """Structural type of treelite's header and tree accessors.

    Both accessors expose the model as a struct of arrays through a single ``get_field`` lookup,
    which is all this module needs from them.
    """

    def get_field(self, name: str) -> np.ndarray:
        """Return the named field of the underlying treelite object."""
        ...  # pragma: no cover - protocol stub


class TreeliteModel(Protocol):
    """Structural type of a parsed ``treelite.Model``."""

    num_tree: int

    def get_header_accessor(self) -> TreeliteAccessor:
        """Return the accessor for the ensemble-level fields."""
        ...  # pragma: no cover - protocol stub

    def get_tree_accessor(self, tree_id: int) -> TreeliteAccessor:
        """Return the accessor for the fields of one tree."""
        ...  # pragma: no cover - protocol stub


def _import_treelite() -> ModuleType:
    """Import the optional ``treelite`` dependency with an actionable error message.

    Returns:
        The imported ``treelite`` module.

    Raises:
        ImportError: If ``treelite`` is not installed.
    """
    try:
        import treelite  # ty: ignore[unresolved-import]
    except ImportError as error:  # pragma: no cover - depends on the environment
        msg = (
            "Converting this model requires the optional dependency 'treelite'. "
            "Install it with `pip install treelite`."
        )
        raise ImportError(msg) from error
    return treelite


def _field(
    accessor: TreeliteAccessor, name: str, dtype: DTypeLike | None = None
) -> NDArray[Any] | None:
    """Read one flattened field from a treelite header or tree accessor.

    treelite raises for fields a model does not populate (e.g. ``sum_hess`` on a model that only
    tracks ``data_count``) and returns an empty array for fields that exist but are unused, so
    both cases are collapsed into ``None`` here.

    Args:
        accessor: A treelite header or tree accessor exposing ``get_field``.
        name: The name of the field to read.
        dtype: The dtype to cast the field to. ``None`` keeps treelite's own dtype.

    Returns:
        The field as a flat array, or ``None`` if the model does not populate it.
    """
    try:
        array = np.asarray(accessor.get_field(name)).ravel()
    except Exception:  # noqa: BLE001 - treelite raises a bare TreeliteError for unset fields
        return None
    if array.size == 0:
        return None
    return array.astype(dtype, copy=True) if dtype is not None else array


def _resolve_class_index(n_outputs: int, class_label: int | None) -> int:
    """Resolve which output of a multi-output model to convert.

    Args:
        n_outputs: The number of outputs (classes) the model produces per instance.
        class_label: The requested class index, or ``None`` for shapiq's default.

    Returns:
        The class index to extract. Always ``0`` for single-output models.

    Raises:
        ValueError: If ``class_label`` is out of range for the model.
    """
    if n_outputs <= 1:
        return 0
    class_index = 1 if class_label is None else class_label
    if not 0 <= class_index < n_outputs:
        msg = f"class_label={class_index} is out of range for a model with {n_outputs} classes."
        raise ValueError(msg)
    return class_index


def _decision_type(node_types: NDArray[Any] | None, operators: NDArray[Any] | None) -> str:
    """Derive the split comparison of a tree from treelite's per-node ``cmp`` field.

    ``TreeModel`` stores a single comparison for the whole tree, so a tree that mixes ``<`` and
    ``<=`` splits cannot be represented and is rejected rather than silently mis-routed.

    Args:
        node_types: The per-node ``node_type`` field, used to skip leaves.
        operators: The per-node ``cmp`` field. ``None`` falls back to ``"<="``.

    Returns:
        Either ``"<"`` or ``"<="``.

    Raises:
        ValueError: If the tree mixes comparison operators or uses one shapiq cannot express.
    """
    if operators is None:
        return "<="
    split_mask = operators != 0 if node_types is None else node_types != _LEAF_NODE
    used = set(np.unique(operators[split_mask]).tolist())
    if not used:  # split-less tree: the comparison is never evaluated
        return "<="
    unsupported = used - set(_OPERATOR_TO_DECISION_TYPE)
    if unsupported:
        msg = (
            f"treelite comparison operator(s) {sorted(unsupported)} cannot be represented by "
            "shapiq's TreeModel, which only supports '<' and '<=' splits."
        )
        raise ValueError(msg)
    if len(used) > 1:
        msg = (
            "Tree mixes the split comparisons "
            f"{sorted(_OPERATOR_TO_DECISION_TYPE[op] for op in used)}; shapiq's TreeModel stores "
            "a single comparison per tree."
        )
        raise ValueError(msg)
    return _OPERATOR_TO_DECISION_TYPE[used.pop()]


def _leaf_values(
    accessor: TreeliteAccessor,
    leaf_mask: NDArray[np.bool_],
    class_index: int,
    vector_length: int,
) -> NDArray[np.float64]:
    """Read the leaf outputs of one tree as a dense per-node array.

    treelite stores either a scalar ``leaf_value`` per leaf (boosters, regression forests) or a
    per-class vote vector sliced out of a flat ``leaf_vector`` array (classification forests).
    Vector leaves are normalized to class probabilities and reduced to ``class_index``, matching
    what shapiq's own scikit-learn converter does.

    Args:
        accessor: The treelite tree accessor.
        leaf_mask: Boolean mask marking the leaf nodes of the tree.
        class_index: The class whose value is extracted from vector-valued leaves.
        vector_length: The length of a leaf vector; ``1`` for scalar leaves.

    Returns:
        A per-node array of leaf outputs; internal nodes are ``0.0``.
    """
    n_nodes = leaf_mask.size
    values = np.zeros(n_nodes, dtype=np.float64)

    if vector_length > 1:
        flat = _field(accessor, "leaf_vector", dtype=np.float64)
        begin = _field(accessor, "leaf_vector_begin", dtype=np.int64)
        end = _field(accessor, "leaf_vector_end", dtype=np.int64)
        if flat is None or begin is None or end is None:
            return values
        for node in np.flatnonzero(leaf_mask):
            vector = flat[begin[node] : end[node]]
            total = vector.sum()
            if total > 0 and vector.size > class_index:
                values[node] = vector[class_index] / total
        return values

    leaf_value = _field(accessor, "leaf_value", dtype=np.float64)
    if leaf_value is None:
        return values
    return np.where(leaf_mask, leaf_value, 0.0)


def _node_sample_weight(accessor: TreeliteAccessor, n_nodes: int) -> NDArray[np.float64]:
    """Read the per-node sample weights of one tree.

    treelite populates ``sum_hess`` for gradient boosters and ``data_count`` for models that only
    track row counts (LightGBM, scikit-learn); never necessarily both. Whichever is present with
    the right length wins, and a model that tracks neither falls back to uniform weights.

    Args:
        accessor: The treelite tree accessor.
        n_nodes: The number of nodes in the tree.

    Returns:
        A per-node array of sample weights.
    """
    for name in ("sum_hess", "data_count"):
        weights = _field(accessor, name, dtype=np.float64)
        if weights is not None and weights.size == n_nodes:
            return weights
    return np.ones(n_nodes, dtype=np.float64)


def _convert_treelite_tree(
    accessor: TreeliteAccessor,
    *,
    class_index: int,
    vector_length: int,
    scaling: float,
    base_score: float,
    output_type: str,
) -> TreeModel:
    """Convert a single treelite tree into shapiq's internal ``TreeModel`` format.

    Args:
        accessor: The treelite tree accessor for the tree to convert.
        class_index: The class to extract from vector-valued leaves.
        vector_length: The length of a leaf vector; ``1`` for scalar leaves.
        scaling: A multiplicative factor applied to all leaf values (``1 / n_trees`` for
            ensembles whose output is averaged rather than summed).
        base_score: The share of the ensemble intercept folded into this tree's leaves.
        output_type: The value of ``original_output_type`` to record on the tree.

    Returns:
        The tree in shapiq's internal ``TreeModel`` format.

    Raises:
        ValueError: If the tree contains a categorical split, which shapiq cannot represent.
    """
    children_left = _field(accessor, "cleft", dtype=np.int64)
    children_right = _field(accessor, "cright", dtype=np.int64)
    split_index = _field(accessor, "split_index", dtype=np.int64)
    thresholds = _field(accessor, "threshold", dtype=np.float64)
    node_types = _field(accessor, "node_type", dtype=np.int64)

    if node_types is not None and np.any(node_types == _CATEGORICAL_TEST_NODE):
        msg = (
            "The model contains categorical splits, which shapiq's TreeModel cannot represent. "
            "Re-train the model with numerical (e.g. one-hot encoded) features."
        )
        raise ValueError(msg)

    n_nodes = children_left.size
    leaf_mask = children_left == -1

    # shapiq routes missing-value samples via ``children_missing``, following whichever child the
    # source model's ``default_left`` flag points at.
    default_left = _field(accessor, "default_left", dtype=np.int64)
    children_missing = (
        np.where(default_left.astype(bool), children_left, children_right)
        if default_left is not None
        else children_left
    )

    features = split_index
    features[leaf_mask] = -2  # shapiq's leaf sentinel
    thresholds[leaf_mask] = np.nan

    values = _leaf_values(accessor, leaf_mask, class_index, vector_length) * scaling
    # The ensemble intercept is spread evenly over the trees so that the leaf values of the
    # converted ensemble sum to the model's raw output, matching shapiq's XGBoost converter.
    values[leaf_mask] += base_score

    # A split-less tree (a bare root leaf) contributes a constant. Boosters emit these once the
    # loss stops improving. ``TreeModel`` derives ``max_feature_id`` via ``max()`` over the
    # non-leaf features and would raise on the resulting empty set, so those fields are supplied
    # explicitly here.
    degenerate_fields: dict[str, Any] = (
        {"n_features_in_tree": 0, "max_feature_id": 0, "feature_ids": set()}
        if leaf_mask.all()
        else {}
    )

    return TreeModel(
        children_left=children_left,
        children_right=children_right,
        children_missing=children_missing,
        features=features,
        thresholds=thresholds,
        values=values,
        node_sample_weight=_node_sample_weight(accessor, n_nodes),
        empty_prediction=None,  # pyright: ignore[reportArgumentType] computed by TreeModel
        original_output_type=output_type,
        decision_type=_decision_type(node_types, _field(accessor, "cmp", dtype=np.int64)),
        **degenerate_fields,
    )


def convert_treelite_model(
    tl_model: TreeliteModel, class_label: int | None = None
) -> list[TreeModel]:
    """Convert an already-parsed treelite model to the unified internal tree format used by shapiq.

    The conversion reconciles four layout differences between treelite and shapiq:

    * **Leaf sentinels.** treelite marks leaves with ``cleft == -1``; shapiq expects
      ``features == -2`` and ``thresholds == NaN`` at leaves.
    * **Scalar vs. vector leaves.** Classification forests store a per-class vote vector per leaf,
      which is normalized to probabilities and reduced to the selected class. Boosters store a
      scalar leaf value, which is kept in raw (margin) space.
    * **Averaged output.** Forests set ``average_tree_output``; their leaf values are scaled by
      ``1 / n_trees`` so that the converted ensemble sums rather than averages.
    * **Intercept.** treelite keeps the ensemble intercept in ``base_scores``; shapiq has no slot
      for it, so it is spread evenly over the leaves of the selected trees.

    For multiclass boosters, treelite tags every tree with the class it predicts (``class_id``);
    only the trees of the selected class are returned, so the converted ensemble reproduces that
    class's raw margin.

    Args:
        tl_model: A parsed ``treelite.Model``.
        class_label: The class to extract for multiclass models. ``None`` selects class ``1``,
            matching shapiq's default for classifiers. Ignored for single-output models.

    Returns:
        A list of ``TreeModel`` instances, one per tree of the selected class.

    Raises:
        NotImplementedError: If the model has multiple regression targets.
        ValueError: If ``class_label`` is out of range, or a tree uses a split shapiq cannot
            represent.
    """
    header = tl_model.get_header_accessor()

    num_target = _field(header, "num_target", dtype=np.int64)
    if num_target is not None and int(num_target[0]) > 1:
        msg = (
            f"Multi-target models are not supported ({int(num_target[0])} targets found); "
            "shapiq explains one output at a time."
        )
        raise NotImplementedError(msg)

    num_class = _field(header, "num_class", dtype=np.int64)
    n_classes = int(num_class.max()) if num_class is not None else 1

    leaf_vector_shape = _field(header, "leaf_vector_shape", dtype=np.int64)
    vector_length = int(leaf_vector_shape[-1]) if leaf_vector_shape is not None else 1

    class_index = _resolve_class_index(max(n_classes, vector_length), class_label)

    # Multiclass boosters interleave one tree per class per boosting round and tag each tree with
    # its ``class_id``. Models with vector-valued leaves leave ``class_id`` at ``-1`` because
    # every tree predicts all classes at once; there, the class is selected inside the leaf.
    class_id = _field(header, "class_id", dtype=np.int64)
    if class_id is not None and np.any(class_id >= 0) and n_classes > 1:
        tree_indices = np.flatnonzero(class_id == class_index).tolist()
    else:
        tree_indices = list(range(tl_model.num_tree))
    n_trees = len(tree_indices)

    average = _field(header, "average_tree_output", dtype=np.int64)
    averaged = average is not None and bool(average[0])
    scaling = 1.0 / n_trees if averaged and n_trees > 0 else 1.0

    base_scores = _field(header, "base_scores", dtype=np.float64)
    base_score = 0.0
    if base_scores is not None and n_trees > 0:
        index = class_index if base_scores.size > class_index else 0
        base_score = float(base_scores[index]) / n_trees

    output_type = "probability" if vector_length > 1 else "raw"

    return [
        _convert_treelite_tree(
            tl_model.get_tree_accessor(tree_index),
            class_index=class_index,
            vector_length=vector_length,
            scaling=scaling,
            base_score=base_score,
            output_type=output_type,
        )
        for tree_index in tree_indices
    ]


def convert_sklearn_boosting_model(
    model: SklearnBoostingModel, class_label: int | None = None
) -> list[TreeModel]:
    """Convert a scikit-learn gradient boosting model to the unified internal tree format.

    Handles :class:`~sklearn.ensemble.GradientBoostingRegressor`,
    :class:`~sklearn.ensemble.GradientBoostingClassifier`,
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor`, and
    :class:`~sklearn.ensemble.HistGradientBoostingClassifier` by parsing them with treelite first.
    None of these expose their trees in a layout shapiq can read directly: the histogram-based
    variants use a bespoke ``TreePredictor`` node array, and the classic variants keep their
    learning rate and initial prediction outside the trees.

    The converted ensemble reproduces the model's **raw** output -- ``predict`` for the
    regressors and ``decision_function`` (log-odds for binary, per-class margin for multiclass)
    for the classifiers -- in line with shapiq's other gradient boosting converters.

    Args:
        model: The fitted scikit-learn gradient boosting model to convert.
        class_label: For multiclass classifiers, the class index to extract trees for.
            ``None`` selects class ``1``. Ignored for regressors and binary classifiers, which
            produce a single output.

    Returns:
        A list of ``TreeModel`` instances, one per boosting round for the selected class.
    """
    treelite = _import_treelite()
    return convert_treelite_model(treelite.sklearn.import_model(model), class_label=class_label)


register(GradientBoostingRegressor, convert_sklearn_boosting_model)
register(GradientBoostingClassifier, convert_sklearn_boosting_model)
register(HistGradientBoostingRegressor, convert_sklearn_boosting_model)
register(HistGradientBoostingClassifier, convert_sklearn_boosting_model)
