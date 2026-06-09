"""Proxy model definitions shared by the proxy approximators.

This module collects everything about *which* model is used as a proxy, independent of how a
given approximator reads interactions out of it:

* the :class:`ProxyModel` / :class:`ProxyModelWithHPO` protocols,
* the hyperparameter-search wrappers (:class:`WrapperGridSearchCV`,
  :class:`WrapperRandomizedSearchCV`, :class:`SMACProxyModel`),
* the :data:`ProxyLiteral` string identifiers and :func:`_select_base_proxy_via_string`.

Both :class:`~shapiq.approximator.proxy.proxyshap.ProxySHAP` and
:class:`~shapiq.approximator.proxy.proxyspex.ProxySPEX` build on these.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable
from warnings import warn

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace


ProxyLiteral = Literal["xgboost", "lightgbm", "tree", "linear"]


def get_smac_config(proxy_name: str) -> ConfigurationSpace:
    """Build the SMAC :class:`ConfigurationSpace` for a supported proxy model.

    Args:
        proxy_name: ``"xgb"`` or ``"lightgbm"``.

    Returns:
        A :class:`ConfigSpace.ConfigurationSpace` covering the proxy's
        tunable hyperparameters.

    Raises:
        ValueError: If ``proxy_name`` is not a supported proxy.
    """
    try:
        from ConfigSpace import (
            ConfigurationSpace,
            UniformFloatHyperparameter,
            UniformIntegerHyperparameter,
        )
    except ImportError as e:
        msg = "ConfigSpace is not installed. SMAC-based proxy models require ConfigSpace to define the hyperparameter search space. Install it with: pip install 'smac'."
        raise ImportError(msg) from e

    cs = ConfigurationSpace(name=f"SMAC_{proxy_name}_ConfigSpace")
    if proxy_name == "xgb":
        # Integers
        cs.add(
            [
                UniformIntegerHyperparameter(
                    "n_estimators", lower=100, upper=2000, default_value=800
                ),
                UniformIntegerHyperparameter("max_depth", lower=2, upper=8, default_value=4),
                UniformIntegerHyperparameter(
                    "min_child_weight", lower=1, upper=20, default_value=5
                ),
            ]
        )

        # Floats
        cs.add(
            [
                UniformFloatHyperparameter("subsample", lower=0.4, upper=1.0, default_value=1),
                UniformFloatHyperparameter(
                    "colsample_bytree", lower=0.4, upper=1.0, default_value=1
                ),
                # learning_rate is almost always better searched on a log scale
                UniformFloatHyperparameter(
                    "learning_rate", lower=1e-3, upper=0.3, default_value=0.05, log=True
                ),
                # L2 regularization
                UniformFloatHyperparameter(
                    "reg_lambda", lower=1e-3, upper=50.0, default_value=1.0, log=True
                ),
                # L1 regularization
                UniformFloatHyperparameter(
                    "reg_alpha", lower=1e-3, upper=50.0, default_value=1.0, log=True
                ),
            ]
        )
    elif proxy_name == "lightgbm":
        cs.add(
            [
                UniformIntegerHyperparameter("max_depth", lower=2, upper=6, default_value=3),
                UniformIntegerHyperparameter(
                    "n_estimators", lower=100, upper=2000, default_value=800
                ),
                UniformIntegerHyperparameter(
                    "min_child_samples", lower=2, upper=20, default_value=20
                ),
            ]
        )

        cs.add(
            [
                UniformFloatHyperparameter(
                    "learning_rate", lower=1e-2, upper=1e-1, default_value=0.1, log=True
                ),
            ]
        )
    else:
        msg = f"Model {proxy_name} not recognized for SMAC configuration."
        raise ValueError(msg)

    return cs


@runtime_checkable
class ProxyModel(Protocol):
    """Protocol for the proxy model used in ProxySHAP. The model must implement the scikit-learn regressor interface."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the proxy model to the coalition matrix ``X`` and game values ``y``."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict game values for the coalition matrix ``X``."""
        ...


@runtime_checkable
class ProxyModelWithHPO(ProxyModel, Protocol):
    """Protocol for a proxy model that is the result of a hyperparameter optimization process, such as GridSearchCV.

    The wrapper exposes two estimators with distinct roles:

    * ``estimator`` -- the *unfitted* base model. Its type selects the proxy route (linear vs.
      tree, see :func:`shapiq.approximator.proxy.proxyshap.proxy_approximate`) and is available
      before fitting.
    * ``best_estimator_`` -- the *fitted* model chosen by the search, used for interaction
      extraction and residual adjustment. Only available after :meth:`fit`.
    """

    estimator: ProxyModel
    best_estimator_: ProxyModel


class WrapperGridSearchCV(ProxyModelWithHPO):
    """Wrapper for GridSearchCV to make it compatible with the ProxyModelWithHPO protocol."""

    def __init__(self, estimator: ProxyModel, param_grid: dict, cv: int = 5) -> None:
        """Initialize the GridSearchCV wrapper with the given estimator, parameter grid, and cross-validation strategy."""
        self.estimator = estimator  # unfitted base estimator; its type drives route dispatch
        self.search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GridSearchCV estimator to the given data."""
        self.search.fit(X, y)
        self.best_estimator_ = self.search.best_estimator_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best estimator found by GridSearchCV."""
        return self.best_estimator_.predict(X)


class WrapperRandomizedSearchCV(ProxyModelWithHPO):
    """Wrapper for RandomizedSearchCV to make it compatible with the ProxyModelWithHPO protocol."""

    def __init__(
        self, estimator: ProxyModel, param_distributions: dict, n_iter: int = 10, cv: int = 5
    ) -> None:
        """Initialize the RandomizedSearchCV wrapper with the given estimator, parameter distributions, number of iterations, and cross-validation strategy."""
        self.estimator = estimator  # unfitted base estimator; its type drives route dispatch
        self.search = RandomizedSearchCV(
            estimator=estimator, param_distributions=param_distributions, n_iter=n_iter, cv=cv
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the RandomizedSearchCV estimator to the given data."""
        self.search.fit(X, y)
        self.best_estimator_ = self.search.best_estimator_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best estimator found by RandomizedSearchCV."""
        return self.best_estimator_.predict(X)


def _smac_config_name(estimator: ProxyModel) -> str:
    """Map a SMAC base estimator to its :func:`get_smac_config` search-space name."""
    name = type(estimator).__name__
    if name == "XGBRegressor":
        return "xgb"
    if name == "LGBMRegressor":
        return "lightgbm"
    msg = f"SMAC HPO is only supported for XGBRegressor and LGBMRegressor, got {name}."
    raise ValueError(msg)


class SMACProxyModel(ProxyModelWithHPO):
    """Wrapper for a SMAC proxy model to make it compatible with the ProxyModelWithHPO protocol."""

    def __init__(self, estimator: ProxyModel, random_state: int) -> None:
        """Initialize the SMAC proxy model wrapper.

        Args:
            estimator: An unfitted ``XGBRegressor`` or ``LGBMRegressor`` instance. Its type
                selects both the SMAC search space and the model rebuilt with the tuned config,
                and drives the proxy route dispatch.
            random_state: Seed shared with the SMAC scenario for deterministic optimization.
        """
        try:
            from smac import Scenario
        except ImportError as e:
            msg = "SMAC is not installed. SMAC-based proxy models require the 'smac' package. Install it with: pip install 'smac'."
            raise ImportError(msg) from e

        self.estimator = estimator  # unfitted base estimator; its type drives route dispatch
        self.conf_space = get_smac_config(_smac_config_name(estimator))
        self.scenario = Scenario(
            self.conf_space,
            deterministic=True,
            n_trials=200,
            output_directory=Path(tempfile.mkdtemp()),
            seed=random_state,
        )

    def _fit_xgboost(self, X: np.ndarray, y: np.ndarray) -> ProxyModel:
        from xgboost import XGBRegressor

        try:
            from smac import HyperparameterOptimizationFacade
        except ImportError as e:
            msg = "SMAC is not installed. SMAC-based proxy models require the 'smac' package. Install it with: pip install 'smac'."
            raise ImportError(msg) from e

        def smac_objective(config: Configuration) -> float:
            # Initialize the XGBoost regressor with the given hyperparameters from SMAC and fit it to the data.
            # We set n_jobs=1 to avoid nested parallelism with SMAC.
            params = dict(config)
            model = XGBRegressor(
                n_jobs=1,
                random_state=self.scenario.seed,
                **params,
            )
            model.fit(X, y)
            # Use cross-validation to evaluate the model's performance; SMAC minimizes the objective, so we return the negative MSE.
            cv = KFold(n_splits=5, shuffle=True, random_state=self.scenario.seed)
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring="neg_mean_squared_error",
                n_jobs=1,  # avoid nested parallelism with CV/SMAC
            )
            return -np.mean(scores)  # SMAC minimizes, so negate the score

        self.smac = HyperparameterOptimizationFacade(
            scenario=self.scenario, target_function=smac_objective
        )
        best_found_config = self.smac.optimize()
        params_best = dict(best_found_config)  # ty: ignore[no-matching-overload]
        best_model = XGBRegressor(
            random_state=self.scenario.seed,
            **params_best,
        )
        best_model.fit(X, y)
        return best_model

    def _fit_lightgbm(self, X: np.ndarray, y: np.ndarray) -> ProxyModel:
        from lightgbm import LGBMRegressor

        try:
            from smac import HyperparameterOptimizationFacade
        except ImportError as e:
            msg = "SMAC is not installed. SMAC-based proxy models require the 'smac' package. Install it with: pip install 'smac'."
            raise ImportError(msg) from e

        def smac_objective(config: Configuration) -> float:
            # Initialize the LightGBM regressor with the given hyperparameters from SMAC and fit it to the data.
            # We set n_jobs=1 to avoid nested parallelism with SMAC.
            params = dict(config)
            model = LGBMRegressor(
                n_jobs=1,
                random_state=self.scenario.seed,
                **params,
            )
            model.fit(X, y)
            # Use cross-validation to evaluate the model's performance; SMAC minimizes the objective, so we return the negative MSE.
            cv = KFold(n_splits=5, shuffle=True, random_state=self.scenario.seed)
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring="neg_mean_squared_error",
                n_jobs=1,  # avoid nested parallelism with CV/SMAC
            )
            return -np.mean(scores)  # SMAC minimizes, so negate the score

        self.smac = HyperparameterOptimizationFacade(
            scenario=self.scenario, target_function=smac_objective
        )
        best_found_config = self.smac.optimize()
        params_best = dict(best_found_config)  # ty: ignore[no-matching-overload]
        best_model = LGBMRegressor(
            random_state=self.scenario.seed,
            **params_best,
        )
        best_model.fit(X, y)
        return best_model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the SMAC estimator to the given data."""
        name = type(self.estimator).__name__
        if name == "XGBRegressor":
            self.best_estimator_ = self._fit_xgboost(X, y)
        elif name == "LGBMRegressor":
            self.best_estimator_ = self._fit_lightgbm(X, y)
        else:
            msg = f"Unsupported SMAC proxy estimator: {name}"
            raise ValueError(msg)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best estimator found by SMAC."""
        return self.best_estimator_.predict(X)


def _select_base_proxy_via_string(proxy_str: str, random_state: int | None) -> ProxyModel:
    """Select a proxy model based on a string identifier.

    Recognized identifiers are the members of :data:`ProxyLiteral`; any other string falls
    back to a :class:`~sklearn.tree.DecisionTreeRegressor` with a warning.
    """
    # XGBoost route: We try XGBoost, then LightGBM and fall back to a Decision Tree if both are unavailable.
    if proxy_str == "xgboost":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(random_state=random_state)
        except ImportError:
            msg = "XGBoost is not installed. Install it with: pip install 'shapiq[proxy]' or choose a different proxy_model."
            warn(msg, stacklevel=2)
        if proxy_str == "lightgbm":
            try:
                from lightgbm import LGBMRegressor

                return LGBMRegressor(random_state=random_state)
            except ImportError:
                msg = "LightGBM is not installed. Install it with: pip install 'shapiq[proxy]' or choose a different proxy_model."
                warn(msg, stacklevel=2)

    # LightGBM route: We try LightGBM and fall back to a Decision Tree if it's unavailable.
    if proxy_str == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(random_state=random_state)
        except ImportError:
            msg = "LightGBM is not installed. Install it with: pip install 'shapiq[proxy]' or choose a different proxy_model."
            warn(msg, stacklevel=2)

        if proxy_str == "xgboost":
            try:
                from xgboost import XGBRegressor

                return XGBRegressor(random_state=random_state)
            except ImportError:
                msg = "XGBoost is not installed. Install it with: pip install 'shapiq[proxy]' or choose a different proxy_model."
                warn(msg, stacklevel=2)

    # Tree route: We try a Decision Tree regressor, which is the fallback if no other tree model is available.
    if proxy_str != "linear":
        msg = f"Proxy model '{proxy_str}' is not available. Falling back to DecisionTreeRegressor."
        warn(msg, stacklevel=2)
        return DecisionTreeRegressor(random_state=random_state)
    # Linear route: We use a simple linear regression model, which is the only option for a linear proxy.
    return LinearRegression()
