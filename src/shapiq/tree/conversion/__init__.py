"""Fast conversion of tree-based models to the unified internal tree format used by shapiq.

The public API of this sub-package is a single function, :func:`convert_tree_model`, which
dispatches to the appropriate converter based on the type of the model passed in.  Scikit-learn
converters are loaded eagerly; XGBoost and LightGBM converters are loaded lazily the first time
an instance of the respective booster class is encountered.
"""

from __future__ import annotations

from . import common
from . import sklearn as sklearn


@common.conversion_generator.delayed_register("xgboost.Booster")
def _(_: type) -> None:
    """Lazily import the boosting module when an XGBoost Booster is first encountered."""
    from . import boosting as boosting  # noqa: F401


@common.conversion_generator.delayed_register("xgboost.sklearn.XGBRegressor")
def _(_: type) -> None:
    """Lazily import the boosting module when an XGBRegressor is first encountered."""
    from . import boosting as boosting  # noqa: F401


@common.conversion_generator.delayed_register("xgboost.sklearn.XGBClassifier")
def _(_: type) -> None:
    """Lazily import the boosting module when an XGBClassifier is first encountered."""
    from . import boosting as boosting  # noqa: F401


@common.conversion_generator.delayed_register("lightgbm.basic.Booster")
def _(_: type) -> None:
    """Lazily import the boosting module when a LightGBM Booster is first encountered."""
    from . import boosting as boosting  # noqa: F401


@common.conversion_generator.delayed_register("lightgbm.sklearn.LGBMRegressor")
def _(_: type) -> None:
    """Lazily import the boosting module when an LGBMRegressor is first encountered."""
    from . import boosting as boosting  # noqa: F401


@common.conversion_generator.delayed_register("lightgbm.sklearn.LGBMClassifier")
def _(_: type) -> None:
    """Lazily import the boosting module when an LGBMClassifier is first encountered."""
    from . import boosting as boosting  # noqa: F401


convert_tree_model = common.convert_tree_model

__all__ = ["convert_tree_model"]