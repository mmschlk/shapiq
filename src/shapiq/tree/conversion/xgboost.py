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


def _xgboost_margin_base_score(booster: Booster, class_label: int | None) -> float:
    """Return the XGBoost base score in margin (raw) space.

    Reads ``base_score`` and ``objective`` from the booster config and applies
    the link function for objectives that store ``base_score`` in probability
    space.
    """
    cfg = json.loads(booster.save_config())
    learner = cfg["learner"]
    objective = learner["learner_train_param"]["objective"]
    raw_field = learner["learner_model_param"]["base_score"].strip("[]")
    tokens = [tok for tok in raw_field.split(",") if tok]
    idx = 0 if class_label is None or class_label < 0 else min(class_label, len(tokens) - 1)
    base = float(tokens[idx])

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
    index ``i % num_class == class_label``). For binary/regression models all trees are
    returned unchanged.

    Args:
        model: The XGBoost regressor or classifier to convert.
        class_label: For multiclass classifiers, the class index to extract trees for.
            Pass ``None`` to return all trees (regression / binary).

    Returns:
        A list of ``TreeModel`` instances, one per boosting round for the selected class.
    """
    booster = model if isinstance(model, Booster) else model.get_booster()
    margin_base_score = _xgboost_margin_base_score(booster, class_label)
    return parse_xgboost_ubjson_treemodels(
        booster.save_raw(),
        -1 if class_label is None else class_label,
        margin_base_score,
    )


register(XGBRegressor, convert_xgboost_model)
register(XGBClassifier, convert_xgboost_model)
register(Booster, convert_xgboost_model)
