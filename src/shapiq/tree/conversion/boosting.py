"""Conversion utilities for XGBoost and LightGBM models to the unified internal tree format."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cext import (
    parse_lightgbm_string_treemodels,  # ty: ignore[unresolved-import]
    parse_xgboost_ubjson_treemodels,  # ty: ignore[unresolved-import]
)
from .common import register

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier, LGBMRegressor
    from lightgbm.basic import Booster as LightGBMBooster
    from xgboost import XGBClassifier, XGBRegressor

    from shapiq.tree.base import TreeModel

    type LightGBMModel = LGBMRegressor | LGBMClassifier | LightGBMBooster


def convert_xgboost_model(
    model: XGBRegressor | XGBClassifier, class_label: int | None = None
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
    byte_array = model.get_booster().save_raw()
    return parse_xgboost_ubjson_treemodels(byte_array, -1 if class_label is None else class_label)


def _lightgbm_model_to_bytes(model: LightGBMModel) -> bytes:
    """Serialize a LightGBM model to a UTF-8-encoded byte string of its text representation.

    Args:
        model: A LightGBM model or booster object.

    Returns:
        The UTF-8-encoded text representation of the model.

    Raises:
        TypeError: If the model does not expose a ``model_to_string()`` method.
    """
    if hasattr(model, "get_booster"):
        booster = model.get_booster()  # ty: ignore[call-non-callable]
        if hasattr(booster, "model_to_string"):
            return booster.model_to_string().encode("utf-8")  # ty: ignore[call-non-callable]

    if hasattr(model, "booster_") and hasattr(model.booster_, "model_to_string"):
        return model.booster_.model_to_string().encode("utf-8")  # ty: ignore[call-non-callable]

    if hasattr(model, "model_to_string"):
        return model.model_to_string().encode("utf-8")

    msg = "Expected a LightGBM model/booster exposing model_to_string()"
    raise TypeError(msg)


def convert_lightgbm_model(model: LightGBMModel, class_label: int | None = None) -> list[TreeModel]:
    """Convert a LightGBM model to the unified internal tree format used by shapiq.

    For multiclass models, only the trees for ``class_label`` are returned (round-robin
    index ``i % num_tree_per_iteration == class_label``). For binary/regression models
    all trees are returned unchanged.

    Args:
        model: The LightGBM model to convert (``LGBMRegressor``, ``LGBMClassifier``, or
            native ``Booster``).
        class_label: For multiclass classifiers, the class index to extract trees for.
            Pass ``None`` to return all trees (regression / binary).

    Returns:
        A list of ``TreeModel`` instances, one per boosting round for the selected class.
    """
    byte_array = _lightgbm_model_to_bytes(model)
    return parse_lightgbm_string_treemodels(byte_array, -1 if class_label is None else class_label)


try:
    from xgboost import XGBClassifier, XGBRegressor

    register(XGBRegressor, convert_xgboost_model)
    register(XGBClassifier, convert_xgboost_model)
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    from lightgbm.basic import Booster as LightGBMBooster

    register(LGBMRegressor, convert_lightgbm_model)
    register(LGBMClassifier, convert_lightgbm_model)
    register(LightGBMBooster, convert_lightgbm_model)
except ImportError:
    pass
