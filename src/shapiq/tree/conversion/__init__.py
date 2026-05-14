"""Fast conversion of tree-based models to the unified internal tree format used by shapiq.

The public API of this sub-package is a single function, :func:`convert_tree_model`, which
dispatches to the appropriate converter based on the type of the model passed in.  Scikit-learn
converters are loaded eagerly; XGBoost, LightGBM, and CatBoost converters are loaded lazily the
first time an instance of the respective booster class is encountered.
"""

from __future__ import annotations

import importlib

from . import common

importlib.import_module(".sklearn", package=__package__)


@common.conversion_generator.delayed_register(
    ("xgboost.core.Booster", "xgboost.sklearn.XGBRegressor", "xgboost.sklearn.XGBClassifier")
)
def _(_: type) -> None:
    """Lazily import the XGBoost converter module when an XGBoost Booster is first encountered."""
    importlib.import_module(".xgboost", package=__package__)


@common.conversion_generator.delayed_register(
    ("lightgbm.basic.Booster", "lightgbm.sklearn.LGBMRegressor", "lightgbm.sklearn.LGBMClassifier")
)
def _(_: type) -> None:
    """Lazily import the LightGBM module when a LightGBM Booster is first encountered."""
    importlib.import_module(".lightgbm", package=__package__)


@common.conversion_generator.delayed_register(
    (
        "catboost.core.CatBoost",
        "catboost.core.CatBoostRegressor",
        "catboost.core.CatBoostClassifier",
    )
)
def _(_: type) -> None:
    """Lazily import the CatBoost module when a CatBoost model is first encountered."""
    importlib.import_module(".catboost", package=__package__)


convert_tree_model = common.convert_tree_model


__all__ = ["convert_tree_model"]
