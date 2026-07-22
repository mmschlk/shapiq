"""LightGBM converters to the unified tree layout."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

import numpy as np
from lightgbm.basic import Booster  # noqa: TC002 - registration needs the class
from lightgbm.sklearn import LGBMModel  # noqa: TC002 - registration needs the class

from shapiq.trees._model import TreeModel, trusted_tree_model
from shapiq.trees.converters._base import to_tree_model

if TYPE_CHECKING:
    from collections.abc import Callable

_CATEGORICAL_MESSAGE = (
    "the booster uses categorical splits, which the unified tree layout "
    "does not represent; train with numeric features"
)


def _load_kernel() -> Callable[..., tuple] | None:
    """Return the compiled text-dump parser, if the extension was built."""
    try:
        module = importlib.import_module("shapiq.trees._conversion_cext")
    except ImportError:  # pragma: no cover - pure-python installs
        return None
    return cast("Callable[..., tuple]", module.parse_lightgbm_text)


_kernel_parse = _load_kernel()

# one tree in node-array form: children, features, thresholds, values
type _TreeArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _arrays_from_kernel(booster: Booster) -> tuple[list[_TreeArrays], int]:
    """Parse the fast text dump through the compiled kernel."""
    kernel = _kernel_parse
    if kernel is None:  # guarded by the caller; direct calls get a real error
        msg = "the compiled conversion kernel is not available in this install"
        raise RuntimeError(msg)
    counts_raw, left_raw, right_raw, features_raw, thresholds_raw, values_raw, per_iteration = (
        kernel(booster.model_to_string().encode("utf-8"))
    )
    counts = np.frombuffer(counts_raw, dtype=np.int64)
    offsets = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    left = np.frombuffer(left_raw, dtype=np.int64)
    right = np.frombuffer(right_raw, dtype=np.int64)
    features = np.frombuffer(features_raw, dtype=np.int64)
    thresholds = np.frombuffer(thresholds_raw, dtype=np.float64)
    values = np.frombuffer(values_raw, dtype=np.float64)
    per_tree = [
        (
            left[offsets[i] : offsets[i + 1]],
            right[offsets[i] : offsets[i + 1]],
            features[offsets[i] : offsets[i + 1]],
            thresholds[offsets[i] : offsets[i + 1]],
            values[offsets[i] : offsets[i + 1]],
        )
        for i in range(counts.shape[0])
    ]
    return per_tree, int(per_iteration)


def _flatten(structure: dict) -> _TreeArrays:
    """Flatten one nested tree dict into node arrays via preorder walk."""
    lefts: list[int] = []
    rights: list[int] = []
    features: list[int] = []
    thresholds: list[float] = []
    values: list[float] = []

    def walk(node: dict) -> int:
        index = len(lefts)
        for column in (lefts, rights, features, thresholds, values):
            column.append(0)
        if "leaf_value" in node:
            lefts[index], rights[index], features[index] = -1, -1, -2
            thresholds[index] = np.nan
            values[index] = float(node["leaf_value"])
            return index
        if node.get("decision_type") != "<=":
            raise ValueError(_CATEGORICAL_MESSAGE)
        features[index] = int(node["split_feature"])
        thresholds[index] = float(node["threshold"])
        values[index] = 0.0
        lefts[index] = walk(node["left_child"])
        rights[index] = walk(node["right_child"])
        return index

    walk(structure)
    return (
        np.asarray(lefts, dtype=np.int64),
        np.asarray(rights, dtype=np.int64),
        np.asarray(features, dtype=np.int64),
        np.asarray(thresholds, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )


def _arrays_from_dump(booster: Booster) -> tuple[list[_TreeArrays], int]:
    """Walk the dict dump in pure Python; the fallback and the oracle."""
    dump = booster.dump_model()
    per_tree = [_flatten(info["tree_structure"]) for info in dump["tree_info"]]
    return per_tree, int(dump["num_tree_per_iteration"])


def _tree_from_arrays(arrays: _TreeArrays, class_id: int, n_classes: int) -> TreeModel:
    """Apply the unified-layout policy to one tree's node arrays.

    LightGBM routes left on ``x <= threshold``, matching the unified layout
    directly. Multiclass rounds become vector-valued leaves carrying their
    class's contribution.
    """
    left, right, features, thresholds, leaf_values = arrays
    values: np.ndarray
    if n_classes > 1:
        values = np.zeros((left.shape[0], n_classes))
        values[:, class_id] = leaf_values
    else:
        values = leaf_values
    return trusted_tree_model(
        children_left=left,
        children_right=right,
        features=features,
        thresholds=thresholds,
        values=values,
    )


def _from_booster(booster: Booster) -> tuple[TreeModel, ...]:
    """Convert one booster to its raw-score output as a sum of trees.

    The compiled kernel parses the fast text dump; without it (or when it
    rejects the stream) the dict dump is walked in Python. Missing-value
    routing is not represented: explained points must not contain NaN.
    """
    parsed: tuple[list[_TreeArrays], int] | None = None
    if _kernel_parse is not None:
        try:
            parsed = _arrays_from_kernel(booster)
        except ValueError:
            parsed = None
    if parsed is None:
        parsed = _arrays_from_dump(booster)
    per_tree, n_classes = parsed
    return tuple(
        _tree_from_arrays(arrays, index % n_classes, n_classes)
        for index, arrays in enumerate(per_tree)
    )


@to_tree_model.register
def _lightgbm_booster_to_model(model: Booster) -> tuple[TreeModel, ...]:
    return _from_booster(model)


@to_tree_model.register
def _lightgbm_sklearn_to_model(model: LGBMModel) -> tuple[TreeModel, ...]:
    return _from_booster(model.booster_)
