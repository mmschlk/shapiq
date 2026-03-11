"""Fast conversion of tree-based models to the unified internal tree format used by shapiq.

The public API of this sub-package is a single function, :func:`convert_tree_model`, which
dispatches to the appropriate converter based on the type of the model passed in.  Scikit-learn
converters are loaded eagerly; XGBoost and LightGBM converters are loaded lazily the first time
an instance of the respective booster class is encountered.
"""

from __future__ import annotations

import importlib

from . import common


def _load_boosting_module() -> None:
    """Import boosting converters for side effects (converter registration)."""
    importlib.import_module(".boosting", package=__package__)


@common.conversion_generator.delayed_register("xgboost.Booster")
def _(_: type) -> None:
    """Lazily import the boosting module when an XGBoost Booster is first encountered."""
    _load_boosting_module()


@common.conversion_generator.delayed_register("xgboost.sklearn.XGBRegressor")
def _(_: type) -> None:
    """Lazily import the boosting module when an XGBRegressor is first encountered."""
    _load_boosting_module()


@common.conversion_generator.delayed_register("xgboost.sklearn.XGBClassifier")
def _(_: type) -> None:
    """Lazily import the boosting module when an XGBClassifier is first encountered."""
    _load_boosting_module()


@common.conversion_generator.delayed_register("lightgbm.basic.Booster")
def _(_: type) -> None:
    """Lazily import the boosting module when a LightGBM Booster is first encountered."""
    _load_boosting_module()


@common.conversion_generator.delayed_register("lightgbm.sklearn.LGBMRegressor")
def _(_: type) -> None:
    """Lazily import the boosting module when an LGBMRegressor is first encountered."""
    _load_boosting_module()


@common.conversion_generator.delayed_register("lightgbm.sklearn.LGBMClassifier")
def _(_: type) -> None:
    """Lazily import the boosting module when an LGBMClassifier is first encountered."""
    _load_boosting_module()


# Eagerly import sklearn converters since we have sklearn as a hard dependency
importlib.import_module(".sklearn", package=__package__)

convert_tree_model = common.convert_tree_model


__all__ = ["convert_tree_model"]
