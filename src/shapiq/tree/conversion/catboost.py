"""Conversion utilities for CatBoost models to the unified internal tree format."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

from .cext import parse_catboost_json_treemodels  # ty: ignore[unresolved-import]
from .common import register

if TYPE_CHECKING:
    from shapiq.tree.base import TreeModel

    type CatBoostModel = CatBoost | CatBoostRegressor | CatBoostClassifier


def _catboost_model_to_json(model: CatBoostModel) -> dict[str, Any]:
    """Serialize a fitted CatBoost model through its JSON model export."""
    return json.loads(_catboost_model_to_json_bytes(model).decode("utf-8"))


def _catboost_model_to_json_bytes(model: CatBoostModel) -> bytes:
    """Serialize a fitted CatBoost model to UTF-8 JSON bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.json"
        model.save_model(str(path), format="json")
        if not path.is_file():
            msg = "CatBoost JSON export failed: model.save_model() did not create a JSON file."
            raise RuntimeError(msg)
        return path.read_bytes()


def parse_catboost_json_model(
    model_json: dict[str, Any],
    class_label: int | None = None,
) -> list[TreeModel]:
    """Convert a CatBoost JSON model dictionary to the internal tree format.

    Args:
        model_json: CatBoost JSON model as a dictionary.
        class_label: For multiclass classifiers, the class index to extract. ``None`` is passed
            to the C++ parser as ``-1`` and defaults to class ``1`` for multiclass CatBoost
            models. It is ignored for regression/binary trees.

    Returns:
        A list of ``TreeModel`` instances, one per CatBoost tree.
    """
    byte_array = json.dumps(model_json, separators=(",", ":")).encode("utf-8")
    return parse_catboost_json_treemodels(byte_array, -1 if class_label is None else class_label)


def convert_catboost_model(
    model: CatBoostModel,
    class_label: int | None = None,
) -> list[TreeModel]:
    """Convert a CatBoost model to the unified internal tree format used by shapiq.

    The converter uses CatBoost's JSON export and currently supports numeric
    ``FloatFeature`` splits. For multiclass CatBoost models, pass ``class_label`` to
    select the raw margin for one class. If ``class_label`` is ``None``, the C++ parser
    defaults to class ``1`` for multiclass models.
    """
    byte_array = _catboost_model_to_json_bytes(model)
    return parse_catboost_json_treemodels(byte_array, -1 if class_label is None else class_label)


register(CatBoost, convert_catboost_model)
register(CatBoostRegressor, convert_catboost_model)
register(CatBoostClassifier, convert_catboost_model)
