"""Conversion utilities for XGBoost and LightGBM models to the unified internal tree format."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

from xgboost import Booster, XGBClassifier, XGBRegressor

from .cext import (
    parse_xgboost_ubjson_treemodels,  # ty: ignore[unresolved-import]
)
from .common import register

if TYPE_CHECKING:
    from shapiq.tree.base import TreeModel


# Objectives that store `base_score` in probability space. TreeSHAP-IQ models
# the margin (raw) output, so the parser needs `logit(base_score)` instead of
# the stored value for these objectives.
_LOGIT_OBJECTIVES = frozenset({"binary:logistic", "reg:logistic"})


def _xgboost_margin_base_score(cfg: dict, class_label: int) -> float:
    """Return the XGBoost base score in margin (raw) space.

    Reads ``base_score`` and ``objective`` from the booster config and applies
    the link function for objectives that store ``base_score`` in probability
    space.  A negative ``class_label`` means "unspecified" and resolves to class ``1``
    for multiclass models and ``0`` otherwise — the same default the C parser applies
    when filtering trees, so the base score always matches the selected class.
    """
    learner = cfg["learner"]
    objective = learner["learner_train_param"]["objective"]
    raw_field = learner["learner_model_param"]["base_score"].strip("[]")
    tokens = [tok for tok in raw_field.split(",") if tok]
    if class_label < 0:
        num_class = int(learner["learner_model_param"].get("num_class", "0") or 0)
        class_label = 1 if num_class > 1 else 0
    base = float(tokens[min(class_label, len(tokens) - 1)])

    if objective in _LOGIT_OBJECTIVES:
        eps = 1e-15
        p = min(max(base, eps), 1.0 - eps)
        return math.log(p / (1.0 - p))
    return base


def convert_xgboost_model(
    model: XGBRegressor | XGBClassifier | Booster, class_label: int | None = None
) -> list[TreeModel]:
    """Convert an XGBoost model to the unified internal tree format used by shapiq.

    For multiclass models, only the trees for ``class_label`` are returned (round-robin
    index ``i % num_class == class_label``); ``class_label=None`` defaults to class ``1``,
    consistent with the other converters. For binary/regression models all trees are
    returned unchanged.

    Args:
        model: The XGBoost regressor or classifier to convert.
        class_label: For multiclass classifiers, the class index to extract trees for.
            Defaults to ``None``, which selects class ``1`` for multiclass models and is
            ignored for regression / binary models.

    Returns:
        A list of ``TreeModel`` instances, one per boosting round for the selected class.
    """
    booster = model if isinstance(model, Booster) else model.get_booster()
    cfg = json.loads(booster.save_config())
    if class_label is None:
        class_label = -1  # sentinel: the parser defaults to class 1 for multiclass models
    margin_base_score = _xgboost_margin_base_score(cfg, class_label)
    trees = parse_xgboost_ubjson_treemodels(
        booster.save_raw(),
        class_label,
        margin_base_score,
    )
    for tree in trees:
        # XGBoost casts prediction inputs to float32 before comparing against its float32
        # thresholds; the explainers must route the same way (see TreeModel.cast_input)
        tree.input_precision = "float32"
    return trees


register(XGBRegressor, convert_xgboost_model)
register(XGBClassifier, convert_xgboost_model)
register(Booster, convert_xgboost_model)
