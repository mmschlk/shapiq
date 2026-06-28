"""Proxy fit/predict/extract routing for :class:`~shapiq.approximator.proxy.proxyshap.ProxySHAP`.

This module holds the model-type-dependent machinery the ProxySHAP approximator dispatches on,
kept separate from the approximator class itself:

* :func:`_proxy_features`: the single place the linear proxy's polynomial expansion lives; tree
  proxies use the raw coalitions.
* :func:`fit_proxy` / :func:`predict_proxy`: fit and predict a proxy, applying the feature
  transform its base estimator type selects.
* :func:`_extract_proxy_interactions`: read interaction values out of an *already-fitted* proxy
  (linear coefficient read-out vs. exact tree read-out), dispatching on the fitted model's type.
* :class:`ResidualGame`: the game wrapping a proxy's residuals for the adjustment approximator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from lazy_dispatch.singledispatch import lazydispatch
from shapiq.approximator.proxy._models import ProxyModel, ProxyModelWithHPO
from shapiq.game import Game
from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.interaction_values import InteractionValues
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer

if TYPE_CHECKING:
    from sklearn.linear_model._base import LinearModel

    from shapiq.typing import CoalitionMatrix, GameValues

ValidProxySHAPIndices = Literal["k-SII", "FSII", "FBII", "SII", "SV", "BV"]


def _base_estimator(proxy_model: ProxyModel | ProxyModelWithHPO) -> ProxyModel:
    """Return a proxy's *base* estimator: an HPO wrapper's ``.estimator``, else the model itself.

    Used by :func:`fit_proxy` to pick the feature transform from the base estimator's type before
    the (possibly wrapping) model is fit.
    """
    return getattr(proxy_model, "estimator", proxy_model)


def _polynomial_features(max_order: int) -> PolynomialFeatures:
    """Build the interaction-only polynomial expansion used by the linear proxy route."""
    return PolynomialFeatures(degree=max_order, interaction_only=True, include_bias=False)


@lazydispatch
def _proxy_features(
    model: ProxyModel,  # noqa: ARG001  # dispatch key only
    coalitions_matrix: CoalitionMatrix,
    *,
    max_order: int,  # noqa: ARG001  # used by the linear route
) -> CoalitionMatrix:
    """Return the feature matrix a proxy of this type is fit on / predicts from.

    Default route: no feature transform, used e.g. by tree proxies.
    """
    return coalitions_matrix


@_proxy_features.register("sklearn.linear_model._base.LinearModel")
def _linear_proxy_features(
    model: LinearModel,  # noqa: ARG001  # dispatch key only
    coalitions_matrix: CoalitionMatrix,
    *,
    max_order: int,
) -> CoalitionMatrix:
    """Linear route: the interaction-only polynomial expansion of the coalitions."""
    return _polynomial_features(max_order).fit_transform(coalitions_matrix)


def fit_proxy(
    estimator: ProxyModel | ProxyModelWithHPO,
    coalitions_matrix: CoalitionMatrix,
    coalition_values: GameValues,
    *,
    max_order: int,
) -> ProxyModel:
    """Fit a proxy (running HPO if it is a wrapper) and return its fitted base estimator.

    The proxy's *base* estimator type selects the feature transform (see :func:`_proxy_features`).
    HPO wrappers are fit and unwrapped to ``best_estimator_``.
    """
    base = _base_estimator(estimator)
    features = _proxy_features(base, coalitions_matrix, max_order=max_order)
    estimator.fit(features, coalition_values)
    return estimator.best_estimator_ if isinstance(estimator, ProxyModelWithHPO) else estimator


def predict_proxy(
    fitted: ProxyModel,
    coalitions_matrix: CoalitionMatrix,
    *,
    max_order: int,
) -> GameValues:
    """Predict proxy values, re-applying the fitted model's feature transform.

    Again, we require _proxy_features to cover the potential feature transform of the fitted proxy,e.g. linear models' polynomial expansion.
    """
    return fitted.predict(_proxy_features(fitted, coalitions_matrix, max_order=max_order))


@lazydispatch
def _extract_proxy_interactions(
    fitted: object,
    *,
    baseline_value: float,
    max_order: int,
    approximation_index: str,
    target_index: ValidProxySHAPIndices,
    budget: int,
    n_players: int,
) -> InteractionValues:
    """Read interaction values out of an already-fitted proxy, dispatching on its type.

    This is the default fallback: it runs only for fitted models that have no registered route
    (linear models via :func:`_extract_linear`, tree models via :func:`_extract_tree`) and raises
    :class:`NotImplementedError` to surface the unsupported proxy early.

    Args:
        fitted: The fitted (HPO-unwrapped) proxy model; its type selects the route.
        baseline_value: Value of the empty coalition.
        max_order: Maximum interaction order to extract.
        approximation_index: Index the proxy is read out in (the computation index).
        target_index: Index the result is finalized to (the user-facing index).
        budget: Number of sampled coalitions (drives the ``estimated`` flag).
        n_players: Number of players (features).

    Returns:
        The extracted interaction values in ``target_index``.

    Raises:
        NotImplementedError: Always, since this fallback only runs for unregistered types.
    """
    msg = (
        f"No proxy approximation route registered for fitted proxy type {type(fitted).__name__}. "
        "This likely means that the proxy model is of an unsupported type, and the error will be raised downstream in the tree-conversion layer. If you believe this estimator should be supported, please open an issue or submit a pull request with the implementation of the appropriate route in shapiq.approximator.proxy._routes."
    )
    raise NotImplementedError(msg)


@_extract_proxy_interactions.register("sklearn.linear_model._base.LinearModel")
def _extract_linear(
    fitted: LinearModel,
    *,
    baseline_value: float,
    max_order: int,
    approximation_index: str,
    target_index: ValidProxySHAPIndices,
    budget: int,
    n_players: int,
) -> InteractionValues:
    """Route for linear-in-features proxies: read interactions from coefficients.

    ``fitted`` is the linear model already fit (by :func:`fit_proxy`) on the interaction-only
    polynomial expansion of the coalitions. The same (deterministic) expansion is rebuilt here --
    on a dummy of ``n_players`` columns, since its columns depend only on the feature count -- to
    map coefficients back to interaction tuples. ``max_order == 1`` is handled uniformly: a degree-1
    interaction-only expansion is exactly the singleton features.
    """
    poly = _polynomial_features(max_order).fit(np.zeros((1, n_players)))
    linear_interactions = extract_linear_interactions(
        coefficients=fitted.coef_,
        poly=poly,
    )
    proxy_interactions = InteractionValues(
        linear_interactions,
        index=approximation_index,
        n_players=n_players,
        min_order=0,
        max_order=max_order,
        baseline_value=float(baseline_value),
        estimated=budget < 2**n_players,
        estimation_budget=int(budget),
    )
    return MoebiusConverter(moebius_coefficients=proxy_interactions).compute(
        index=target_index, order=max_order
    )


@_extract_proxy_interactions.register(
    (
        "sklearn.tree._classes.DecisionTreeRegressor",
        "sklearn.ensemble._forest.RandomForestRegressor",
        "xgboost.sklearn.XGBRegressor",
        "lightgbm.sklearn.LGBMRegressor",
        "catboost.core.CatBoostRegressor",
    )
)
def _extract_tree(
    fitted: ProxyModel,
    *,
    baseline_value: float,
    max_order: int,
    approximation_index: str,
    target_index: ValidProxySHAPIndices,
    budget: int,
    n_players: int,
) -> InteractionValues:
    """Route for tree proxies: read interactions from the fitted tree via exact tree readout.

    ``fitted`` is the tree model already fit (by :func:`fit_proxy`) on the raw coalitions.
    """
    explainer = InterventionalTreeExplainer(
        fitted,
        data=np.zeros((1, n_players)),  # reference data for boolean tree
        index=approximation_index,
        max_order=max_order,
        bool_tree=True,
    )
    proxy_values = explainer.explain_function(np.ones((1, n_players)))
    return InteractionValues(
        values=proxy_values.interactions,
        index=approximation_index,
        max_order=max_order,
        n_players=n_players,
        min_order=0,
        estimated=budget < 2**n_players,
        estimation_budget=budget,
        baseline_value=float(baseline_value),
        target_index=target_index,
    )


def extract_linear_interactions(
    coefficients: np.ndarray, poly: PolynomialFeatures
) -> dict[tuple[int, ...], float]:
    """Map coefficients of a linear-in-features model back to interaction tuples.

    Args:
        coefficients: Fitted model coefficients, ordered to match ``poly``'s
            column layout.
        poly: Fitted :class:`sklearn.preprocessing.PolynomialFeatures` instance
            used to expand the coalition matrix.

    Returns:
        Mapping from interaction tuple (sorted feature indices) to its
        coefficient.
    """
    interaction_to_col = {}
    for col, p in enumerate(poly.powers_):
        interactions = np.flatnonzero(p)
        interactions.sort()
        interactions = interactions.tolist()
        idx = tuple(interactions)  # features used in this interaction
        interaction_to_col[idx] = col

    # Now build your coefficient dict safely
    return {idx: float(coefficients[col]) for idx, col in interaction_to_col.items()}


class ResidualGame(Game):
    """Residual game class for adjusting the proxy model's predictions.

    The residual values are precomputed on the coalitions that :class:`ProxySHAP` sampled
    and returned for every query. This is correct only because :class:`ProxySHAP` forces a
    fixed ``random_state`` (see its constructor) so that the adjustment approximator's
    coalition sampler reproduces the *identical* coalitions in the identical order; the
    ``i``-th row of any queried coalition matrix then matches the ``i``-th precomputed
    residual.
    """

    def __init__(self, n_players: int, game_values: np.ndarray) -> None:
        """Initialize the residual game with the given values for each coalition."""
        super().__init__(n_players=n_players, normalize=False)
        self.vals = game_values

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:  # noqa: ARG002
        """Return the values of the given coalitions in the residual game.

        Args:
            coalitions: A binary matrix of shape (n_samples, n_features) where each row represents a coalition and each column represents a feature. A value of 1 indicates that the feature is included in the coalition, while a value of 0 indicates that it is not.
            Note: The coalitions are expected to be ordered in the same way as the values in self.vals, i.e., the i-th row of coalitions corresponds to the i-th entry in self.vals. ProxySHAP guarantees this by fixing the random_state shared with the adjustment approximator.

        Returns:
            A vector of shape (n_samples,) where each entry is the value of the corresponding coalition in the residual game.
        """
        return self.vals
