"""XGBoost converters to the unified tree layout."""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING, cast

import numpy as np
from xgboost.core import Booster  # noqa: TC002 - registration needs the class
from xgboost.sklearn import XGBModel  # noqa: TC002 - registration needs the class

from shapiq.trees._conversion import to_tree_model
from shapiq.trees._model import TreeModel, constant_tree, trusted_tree_model

if TYPE_CHECKING:
    from collections.abc import Callable

# objectives that store base_score in probability space; the margin the
# trees sum to needs the logit of the stored value
_LOGIT_OBJECTIVES = frozenset({"binary:logistic", "reg:logistic"})

_CATEGORICAL_MESSAGE = (
    "the booster uses categorical splits, which the unified tree layout "
    "does not represent; train with numeric features"
)


def _load_kernel() -> Callable[..., tuple] | None:
    """Return the compiled UBJSON parser, if the extension was built."""
    try:
        module = importlib.import_module("shapiq.trees._conversion_cext")
    except ImportError:  # pragma: no cover - pure-python installs
        return None
    return cast("Callable[..., tuple]", module.parse_xgboost_ubjson)


_kernel_parse = _load_kernel()

# one tree's raw structure: children, features, and split conditions
type _TreeArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _arrays_from_kernel(booster: Booster) -> tuple[list[_TreeArrays], list[int]]:
    """Parse the fast UBJSON dump through the compiled kernel."""
    kernel = _kernel_parse
    if kernel is None:  # guarded by the caller; direct calls get a real error
        msg = "the compiled conversion kernel is not available in this install"
        raise RuntimeError(msg)
    counts_raw, left_raw, right_raw, features_raw, conditions_raw, info_raw, _ = kernel(
        booster.save_raw()
    )
    counts = np.frombuffer(counts_raw, dtype=np.int64)
    offsets = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    left = np.frombuffer(left_raw, dtype=np.int64)
    right = np.frombuffer(right_raw, dtype=np.int64)
    features = np.frombuffer(features_raw, dtype=np.int64)
    conditions = np.frombuffer(conditions_raw, dtype=np.float64)
    per_tree = [
        (
            left[offsets[i] : offsets[i + 1]],
            right[offsets[i] : offsets[i + 1]],
            features[offsets[i] : offsets[i + 1]],
            conditions[offsets[i] : offsets[i + 1]],
        )
        for i in range(counts.shape[0])
    ]
    tree_classes = [int(c) for c in np.frombuffer(info_raw, dtype=np.int64)]
    return per_tree, tree_classes


def _arrays_from_json(booster: Booster) -> tuple[list[_TreeArrays], list[int]]:
    """Parse the JSON dump in pure Python; the fallback and the oracle."""
    model = json.loads(booster.save_raw(raw_format="json"))
    gbm = model["learner"]["gradient_booster"]["model"]
    per_tree: list[_TreeArrays] = []
    for dump in gbm["trees"]:
        if dump.get("categories_nodes"):
            raise ValueError(_CATEGORICAL_MESSAGE)
        per_tree.append(
            (
                np.asarray(dump["left_children"], dtype=np.int64),
                np.asarray(dump["right_children"], dtype=np.int64),
                np.asarray(dump["split_indices"], dtype=np.int64),
                np.asarray(dump["split_conditions"], dtype=np.float64),
            )
        )
    tree_classes = [int(class_id) for class_id in gbm["tree_info"]]
    return per_tree, tree_classes


def _margin_base_scores(booster: Booster) -> np.ndarray:
    """Return the booster's base score(s) in margin space from its config."""
    config = json.loads(booster.save_config())["learner"]
    raw = str(config["learner_model_param"]["base_score"]).strip("[]")
    values = np.asarray([float(token) for token in raw.split(",") if token], dtype=np.float64)
    objective = str(config["learner_train_param"]["objective"])
    if objective in _LOGIT_OBJECTIVES:
        clipped = np.clip(values, 1e-15, 1.0 - 1e-15)
        values = np.log(clipped / (1.0 - clipped))
    return values


def _tree_from_arrays(arrays: _TreeArrays, class_id: int, n_classes: int) -> TreeModel:
    """Apply the unified-layout policy to one tree's raw structure.

    XGBoost routes left on ``x < threshold`` while the unified layout uses
    ``x <= threshold``, so thresholds shift one ulp down — ``x <= nextafter(t)``
    holds exactly when ``x < t``. Leaf values live in ``split_conditions`` at
    leaf positions. Multiclass rounds become vector-valued leaves carrying
    their class's contribution.
    """
    left, right, features, conditions = arrays
    leaves = left == -1
    leaf_values = np.where(leaves, conditions, 0.0)
    values: np.ndarray
    if n_classes > 1:
        values = np.zeros((left.shape[0], n_classes))
        values[:, class_id] = leaf_values
    else:
        values = leaf_values
    return trusted_tree_model(
        children_left=left,
        children_right=right,
        features=np.where(leaves, -2, features),
        thresholds=np.where(leaves, np.nan, np.nextafter(conditions, -np.inf)),
        values=values,
    )


def _from_booster(booster: Booster) -> tuple[TreeModel, ...]:
    """Convert one booster to its margin output as a sum of trees.

    The compiled kernel parses the fast UBJSON dump; without it (or when it
    rejects the stream) the JSON dump is parsed in Python. The stored
    ``base_score`` becomes a lone-leaf tree so the ensemble sum is the
    margin prediction. Missing-value routing (``default_left``) is not
    represented: explained points must not contain NaN.
    """
    parsed: tuple[list[_TreeArrays], list[int]] | None = None
    if _kernel_parse is not None:
        try:
            parsed = _arrays_from_kernel(booster)
        except ValueError:
            parsed = None
    if parsed is None:
        parsed = _arrays_from_json(booster)
    per_tree, tree_classes = parsed
    n_classes = max(tree_classes, default=0) + 1
    trees = [
        _tree_from_arrays(arrays, class_id, n_classes)
        for arrays, class_id in zip(per_tree, tree_classes, strict=True)
    ]
    base = _margin_base_scores(booster)
    if n_classes > 1:
        base_vector = base if base.shape == (n_classes,) else np.full(n_classes, base[0])
        trees.append(constant_tree(base_vector))
    else:
        trees.append(constant_tree(float(base[0])))
    return tuple(trees)


@to_tree_model.register
def _xgboost_booster_to_model(model: Booster) -> tuple[TreeModel, ...]:
    return _from_booster(model)


@to_tree_model.register
def _xgboost_sklearn_to_model(model: XGBModel) -> tuple[TreeModel, ...]:
    return _from_booster(model.get_booster())
