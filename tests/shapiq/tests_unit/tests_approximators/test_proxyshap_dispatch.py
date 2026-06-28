"""Tests for ProxySHAP's type-based route dispatch.

These tests pin down the *typing* behavior of :func:`_extract_proxy_interactions`: which extraction
route a given proxy model resolves to is decided purely from the type of its (HPO-unwrapped)
base estimator. They cover

* linear vs. tree route resolution per estimator type (incl. linear subclasses via MRO),
* that HPO wrappers dispatch on their *base* estimator while the wrapper itself is handed to
  the route (so its search runs and ``best_estimator_`` is read out),
* the :class:`ProxyModelWithHPO` runtime-protocol semantics the routes rely on, and
* that an unsupported proxy type surfaces a clear ``NotImplementedError``.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from shapiq.approximator.proxy import ProxySHAP
from shapiq.approximator.proxy._models import (
    ProxyModelWithHPO,
    WrapperGridSearchCV,
    WrapperRandomizedSearchCV,
)
from shapiq.approximator.proxy._routes import (
    _base_estimator,
    _extract_linear,
    _extract_proxy_interactions,
    _extract_tree,
)
from tests.shapiq.markers import skip_if_no_catboost, skip_if_no_lightgbm, skip_if_no_xgboost


def _resolve_route(proxy_model: object) -> object:
    """Resolve the extraction route ``proxy_model`` would hit once fit and unwrapped.

    ``fit_proxy`` fits the (possibly HPO-wrapping) model and returns its base estimator, on whose
    type ``_extract_proxy_interactions`` then dispatches. The fitted base shares its type with the
    unfitted base, so unwrapping via :func:`_base_estimator` resolves the same route.
    """
    base = _base_estimator(proxy_model)
    return _extract_proxy_interactions.dispatch(type(base))


def _game(coalitions: np.ndarray) -> np.ndarray:
    """Small deterministic game: a bias, a pairwise interaction and a main effect."""
    return np.array([1.0 + 2.0 * c[0] * c[1] + c[2] for c in coalitions], dtype=float)


# ---------------------------------------------------------------------------------------------
# Route resolution by estimator type
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", [LinearRegression(), Ridge(), Lasso(), ElasticNet()])
def test_linear_models_route_to_linear(model):
    """All ``sklearn`` linear models resolve to the linear route via the ``LinearModel`` MRO."""
    assert _resolve_route(model) is _extract_linear


@pytest.mark.parametrize("model", [DecisionTreeRegressor(), RandomForestRegressor()])
def test_sklearn_trees_route_to_tree(model):
    """Scikit-learn decision trees and forests resolve to the tree route."""
    assert _resolve_route(model) is _extract_tree


@skip_if_no_xgboost
def test_xgboost_routes_to_tree():
    """An ``XGBRegressor`` resolves to the tree route (registered by qualified name)."""
    from xgboost import XGBRegressor

    assert _resolve_route(XGBRegressor()) is _extract_tree


@skip_if_no_lightgbm
def test_lightgbm_routes_to_tree():
    """An ``LGBMRegressor`` resolves to the tree route (registered by qualified name)."""
    from lightgbm import LGBMRegressor

    assert _resolve_route(LGBMRegressor()) is _extract_tree


@skip_if_no_catboost
def test_catboost_routes_to_tree():
    """A ``CatBoostRegressor`` resolves to the tree route (registered by qualified name)."""
    from catboost import CatBoostRegressor

    assert _resolve_route(CatBoostRegressor(verbose=False)) is _extract_tree


def test_unsupported_model_resolves_to_neither_route():
    """A model that is neither linear nor a registered tree falls through to the default."""
    route = _resolve_route(KNeighborsRegressor())
    assert route is not _extract_linear
    assert route is not _extract_tree


# ---------------------------------------------------------------------------------------------
# HPO wrappers dispatch on their base estimator
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wrapper", "expected_route"),
    [
        (WrapperGridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=2), _extract_linear),
        (
            WrapperGridSearchCV(DecisionTreeRegressor(), {"max_depth": [2, 3]}, cv=2),
            _extract_tree,
        ),
        (
            WrapperRandomizedSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, n_iter=2, cv=2),
            _extract_linear,
        ),
        (
            WrapperRandomizedSearchCV(
                DecisionTreeRegressor(), {"max_depth": [2, 3]}, n_iter=2, cv=2
            ),
            _extract_tree,
        ),
    ],
)
def test_hpo_wrapper_dispatches_on_base_estimator(wrapper, expected_route):
    """An HPO wrapper resolves to the route of its *base* estimator, not the wrapper type."""
    assert _resolve_route(wrapper) is expected_route


def test_base_estimator_unwraps_wrapper_to_base_instance():
    """``_base_estimator`` returns the wrapper's unfitted base estimator instance."""
    base = Ridge()
    wrapper = WrapperGridSearchCV(base, {"alpha": [0.1]}, cv=2)
    assert _base_estimator(wrapper) is base


def test_base_estimator_passes_raw_model_through():
    """For a raw proxy model (no ``.estimator``) the base estimator is the model itself."""
    model = DecisionTreeRegressor()
    assert _base_estimator(model) is model


# ---------------------------------------------------------------------------------------------
# ProxyModelWithHPO runtime-protocol semantics relied on by the routes
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", [Ridge(), LinearRegression(), DecisionTreeRegressor()])
def test_raw_models_do_not_satisfy_hpo_protocol(model):
    """Raw estimators expose neither ``estimator`` nor ``best_estimator_`` -> not HPO models."""
    assert not isinstance(model, ProxyModelWithHPO)


def test_wrapper_satisfies_hpo_protocol_and_gains_best_estimator_on_fit():
    """The wrappers are nominal ``ProxyModelWithHPO`` subclasses; ``best_estimator_`` is set on fit."""
    wrapper = WrapperGridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=2)
    assert isinstance(wrapper, ProxyModelWithHPO)  # nominal subclass
    assert not hasattr(wrapper, "best_estimator_")  # only the search result, set on fit
    rng = np.random.default_rng(0)
    wrapper.fit(rng.random((20, 3)), rng.random(20))
    assert hasattr(wrapper, "best_estimator_")


def test_bare_sklearn_gridsearch_matches_hpo_protocol_structurally_after_fit():
    """A plain ``GridSearchCV`` (not our wrapper) matches ``ProxyModelWithHPO`` only once fitted.

    This is the structural-typing path the routes (and ProxySPEX's default proxy) rely on to
    unwrap ``best_estimator_`` for HPO objects that do not nominally subclass the protocol.
    """
    from sklearn.model_selection import GridSearchCV

    search = GridSearchCV(DecisionTreeRegressor(), {"max_depth": [2, 3]}, cv=2)
    assert not isinstance(search, ProxyModelWithHPO)  # best_estimator_ missing pre-fit
    rng = np.random.default_rng(0)
    search.fit(rng.random((20, 3)), rng.random(20))
    assert isinstance(search, ProxyModelWithHPO)


# ---------------------------------------------------------------------------------------------
# End-to-end behavior through ProxySHAP.approximate
# ---------------------------------------------------------------------------------------------


def test_unsupported_proxy_raises_in_approximate():
    """A proxy with no registered route raises a clear ``NotImplementedError`` when used."""
    approximator = ProxySHAP(
        n=4, max_order=2, proxy_model=KNeighborsRegressor(), adjustment="none", random_state=0
    )
    with pytest.raises(NotImplementedError, match="No proxy approximation route registered"):
        approximator.approximate(16, _game)


@pytest.mark.parametrize(
    ("raw_model", "wrapped_model"),
    [
        (Ridge(alpha=1.0), WrapperGridSearchCV(Ridge(alpha=1.0), {"alpha": [1.0]}, cv=2)),
        (
            DecisionTreeRegressor(max_depth=3, random_state=0),
            WrapperGridSearchCV(
                DecisionTreeRegressor(max_depth=3, random_state=0), {"max_depth": [3]}, cv=2
            ),
        ),
    ],
)
def test_hpo_wrapper_matches_equivalent_raw_proxy(raw_model, wrapped_model):
    """A degenerate (single-point) search must reproduce the equivalent raw proxy exactly.

    This confirms the HPO path actually fits the wrapper and reads ``best_estimator_`` (rather
    than silently fitting/reading the unfitted base), for both the linear and the tree route.
    """
    budget = 16  # full coalition budget for n=4 -> deterministic given the shared random_state
    raw = ProxySHAP(n=4, max_order=2, proxy_model=raw_model, adjustment="none", random_state=0)
    hpo = ProxySHAP(n=4, max_order=2, proxy_model=wrapped_model, adjustment="none", random_state=0)

    iv_raw = raw.approximate(budget, _game)
    iv_hpo = hpo.approximate(budget, _game)

    assert iv_hpo.index == iv_raw.index
    assert np.allclose(iv_hpo.get_n_order_values(1), iv_raw.get_n_order_values(1))
    assert np.allclose(iv_hpo.get_n_order_values(2), iv_raw.get_n_order_values(2))
