"""CatBoost converters to the unified tree layout."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from catboost.core import CatBoost  # noqa: TC002 - registration needs the class

from shapiq.trees._conversion import to_tree_model
from shapiq.trees._model import TreeModel, constant_tree, trusted_tree_model

_CATEGORICAL_MESSAGE = (
    "the model uses categorical splits, which the unified tree layout "
    "does not represent; train with numeric features"
)


def _model_json(model: CatBoost) -> dict[str, Any]:
    """Serialize a fitted model through CatBoost's JSON export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.json"
        model.save_model(str(path), format="json")
        return json.loads(path.read_text())


def _oblivious_to_tree(tree: dict[str, Any], scale: float, n_classes: int) -> TreeModel:
    """Unroll one oblivious tree into the binary node-array layout.

    CatBoost trees are symmetric: every node of depth level ``k`` splits on
    ``splits[k]``, and a leaf's index sets bit ``k`` exactly when
    ``x[feature_k] > border_k`` (verified against native predictions).
    The unified layout routes left on ``x <= threshold``, which is the
    exact complement — borders carry over unshifted. Multiclass leaf
    values arrive grouped per leaf and become vector-valued leaves.
    """
    splits = tree["splits"]
    depth = len(splits)
    n_leaves = 1 << depth
    n_internal = n_leaves - 1
    n_nodes = n_internal + n_leaves
    left = np.full(n_nodes, -1, dtype=np.int64)
    right = np.full(n_nodes, -1, dtype=np.int64)
    features = np.full(n_nodes, -2, dtype=np.int64)
    thresholds = np.full(n_nodes, np.nan, dtype=np.float64)
    for level, split in enumerate(splits):
        if split.get("split_type") != "FloatFeature":
            raise ValueError(_CATEGORICAL_MESSAGE)
        level_base = (1 << level) - 1
        child_base = (1 << (level + 1)) - 1
        for path in range(1 << level):
            node = level_base + path
            features[node] = int(split["float_feature_index"])
            thresholds[node] = float(split["border"])
            left[node] = child_base + path  # x <= border: leaf bit stays 0
            right[node] = child_base + path + (1 << level)  # x > border: bit set
    leaf_values = scale * np.asarray(tree["leaf_values"], dtype=np.float64)
    values: np.ndarray
    if n_classes > 1:
        values = np.zeros((n_nodes, n_classes))
        values[n_internal:] = leaf_values.reshape(n_leaves, n_classes)
    else:
        values = np.zeros(n_nodes)
        values[n_internal:] = leaf_values
    return trusted_tree_model(
        children_left=left,
        children_right=right,
        features=features,
        thresholds=thresholds,
        values=values,
    )


def _from_model(model: CatBoost) -> tuple[TreeModel, ...]:
    """Convert one model to its raw-score output as a sum of trees.

    The JSON export carries ``scale_and_bias``: the scale multiplies every
    leaf value and the bias becomes a lone-leaf constant tree, so the
    ensemble sum is the raw formula value (a margin vector for multiclass).
    Missing-value routing is not represented: explained points must not
    contain NaN.
    """
    dump = _model_json(model)
    scale, bias = dump["scale_and_bias"]
    bias_array = np.asarray(bias if isinstance(bias, list) else [bias], dtype=np.float64)
    n_classes = int(bias_array.shape[0])
    trees = [
        _oblivious_to_tree(tree, float(scale), n_classes)
        for tree in dump.get("oblivious_trees", [])
    ]
    if not trees:
        msg = (
            "the CatBoost JSON export holds no oblivious trees; non-symmetric "
            "grow policies are not supported by the converter yet"
        )
        raise ValueError(msg)
    trees.append(constant_tree(bias_array if n_classes > 1 else float(bias_array[0])))
    return tuple(trees)


@to_tree_model.register
def _catboost_to_model(model: CatBoost) -> tuple[TreeModel, ...]:
    return _from_model(model)
