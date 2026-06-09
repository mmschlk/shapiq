"""ProxySHAP approximator class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from lazy_dispatch.singledispatch import lazydispatch
from shapiq.approximator.base import Approximator
from shapiq.approximator.montecarlo.shapiq import SHAPIQ
from shapiq.approximator.montecarlo.svarmiq import SVARMIQ
from shapiq.approximator.proxy._models import (
    ProxyLiteral,
    ProxyModel,
    ProxyModelWithHPO,
    _select_base_proxy_via_string,
)
from shapiq.approximator.regression.kernelshapiq import KernelSHAPIQ
from shapiq.game import Game
from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.interaction_values import InteractionValues
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer

if TYPE_CHECKING:
    from collections.abc import Callable

    from sklearn.linear_model._base import LinearModel

    from shapiq.typing import CoalitionMatrix, FloatVector, GameValues

ValidProxySHAPIndices = Literal["k-SII", "FSII", "FBII", "SII", "SV", "BV"]


def _dispatch_on_base_estimator(proxy_model: object, *_args: object, **_kwargs: object) -> object:
    """Dispatch key for :func:`proxy_approximate`: the proxy's *base* estimator.

    HPO wrappers expose their unfitted base via ``.estimator``; raw proxy models are their own
    base. ``lazydispatch`` keys on ``.__class__`` of the returned object, so returning the base
    *instance* makes the base estimator's type select the route while the original (possibly
    wrapping) model is still handed to the route for fitting.
    """
    return getattr(proxy_model, "estimator", proxy_model)


@lazydispatch(dispatch_on=_dispatch_on_base_estimator)
def proxy_approximate(
    estimator: object,
    *,
    coalitions_matrix: CoalitionMatrix,
    coalition_values: GameValues,
    baseline_value: float,
    max_order: int,
    approximation_index: ValidProxySHAPIndices,
    target_index: ValidProxySHAPIndices,
    adjustment_method: Approximator | None = None,
) -> InteractionValues:
    """Dispatch interaction extraction on the type of the proxy's base ``estimator``.

    This is the default fallback: it is invoked only for estimator types that have no
    registered route (linear models via :func:`_approximate_linear`, tree models via
    :func:`_approximate_tree`) and raises :class:`NotImplementedError` to surface the
    unsupported proxy early.

    Args:
        estimator: The proxy's (HPO-unwrapped) base estimator; its type selects the route.
        coalitions_matrix: Binary coalition matrix the proxy is fit on.
        coalition_values: Baseline-centered game values for ``coalitions_matrix``.
        baseline_value: Value of the empty coalition.
        max_order: Maximum interaction order to extract.
        approximation_index: Index the proxy is read out in (the computation index).
        target_index: Index the result is finalized to (the user-facing index).
        adjustment_method: Optional residual-adjustment approximator, or ``None``.

    Returns:
        The approximated interaction values.

    Raises:
        NotImplementedError: Always, since this fallback only runs for unregistered types.
    """
    msg = (
        f"No proxy approximation route registered for estimator type {type(estimator).__name__}. "
        "This likely means that the proxy model is of an unsupported type, and the error will be raised downstream in the tree-conversion layer. If you believe this estimator should be supported, please open an issue or submit a pull request with the implementation of the appropriate route in shapiq.approximator.proxyshap."
    )
    raise NotImplementedError(msg)


@proxy_approximate.register("sklearn.linear_model._base.LinearModel")
def _approximate_linear(
    estimator: LinearModel | ProxyModelWithHPO,
    *,
    coalitions_matrix: CoalitionMatrix,
    coalition_values: GameValues,
    baseline_value: float,
    max_order: int,
    approximation_index: ValidProxySHAPIndices,
    target_index: ValidProxySHAPIndices,
    adjustment_method: Approximator | None = None,
) -> InteractionValues:
    """Route for linear-in-features proxies: read interactions from coefficients.

    ``estimator`` may be a bare linear model or an HPO wrapper around one; the wrapper is fit
    on the (polynomially expanded) coalition features and its ``best_estimator_`` is read out.
    """
    budget = coalitions_matrix.shape[0]
    n_players = coalitions_matrix.shape[1]
    # 2. Fit the proxy (running HPO if it is a wrapper) and read its linear coefficients.
    linear_interactions: dict[tuple[int, ...], float]
    if max_order == 1:
        proxy_features = coalitions_matrix  # linear proxy fits the raw coalitions
        estimator.fit(proxy_features, coalition_values)
        fitted = (
            estimator.best_estimator_ if isinstance(estimator, ProxyModelWithHPO) else estimator
        )
        linear_interactions = {
            (i,): float(fitted.coef_[i])  # ty: ignore[unresolved-attribute]
            for i in range(n_players)
        }
    else:
        poly = PolynomialFeatures(degree=max_order, interaction_only=True, include_bias=False)
        proxy_features = poly.fit_transform(coalitions_matrix)  # interaction-only expansion
        estimator.fit(proxy_features, coalition_values)
        fitted = (
            estimator.best_estimator_ if isinstance(estimator, ProxyModelWithHPO) else estimator
        )
        linear_interactions = extract_linear_interactions(
            coefficients=fitted.coef_,  # ty: ignore[unresolved-attribute]
            poly=poly,
        )

    proxy_interactions = InteractionValues(
        linear_interactions,
        index=approximation_index,
        n_players=n_players,
        min_order=0,
        max_order=max_order,
        baseline_value=float(baseline_value),
        estimated=not budget >= 2**n_players,
        estimation_budget=int(budget),
    )
    proxy_interactions = MoebiusConverter(moebius_coefficients=proxy_interactions).compute(
        index=target_index, order=max_order
    )

    # 3. Optional adjustment of the proxy. The adjustment approximator re-samples the same
    if adjustment_method is not None:
        residual_values = coalition_values - fitted.predict(proxy_features)
        residual_values -= residual_values[0]  # Normalize residuals
        residual_game = ResidualGame(n_players=n_players, game_values=residual_values)
        proxy_interactions += adjustment_method.approximate(budget, residual_game)
    proxy_interactions.baseline_value = baseline_value
    proxy_interactions.interactions[()] = baseline_value  # Ensure empty coalition value is correct
    return proxy_interactions


@proxy_approximate.register(
    (
        "sklearn.tree._classes.DecisionTreeRegressor",
        "sklearn.ensemble._forest.RandomForestRegressor",
        "xgboost.sklearn.XGBRegressor",
        "lightgbm.sklearn.LGBMRegressor",
        "catboost.core.CatBoostRegressor",
    )
)
def _approximate_tree(
    estimator: ProxyModel | ProxyModelWithHPO,
    /,
    coalitions_matrix: CoalitionMatrix,
    coalition_values: GameValues,
    baseline_value: float,
    max_order: int,
    approximation_index: ValidProxySHAPIndices,
    target_index: ValidProxySHAPIndices,
    adjustment_method: Approximator | None = None,
) -> InteractionValues:
    """Route for tree proxies: fit the tree and read interactions via exact tree readout.

    ``estimator`` may be a bare tree model or an HPO wrapper around one; the wrapper is fit on
    the coalitions and its ``best_estimator_`` is read out.
    """
    budget = coalitions_matrix.shape[0]
    n_players = coalitions_matrix.shape[1]
    estimator.fit(coalitions_matrix, coalition_values)
    fitted = estimator.best_estimator_ if isinstance(estimator, ProxyModelWithHPO) else estimator
    # 2. Extract interactions from proxy tree
    explainer = InterventionalTreeExplainer(
        fitted,
        data=np.zeros((1, n_players)),  # reference data for boolean tree
        index=approximation_index,
        max_order=max_order,
        bool_tree=True,
    )
    proxy_values = explainer.explain_function(np.ones((1, n_players)))
    proxy_interactions = InteractionValues(
        values=proxy_values.interactions,
        index=approximation_index,
        max_order=max_order,
        n_players=n_players,
        min_order=0,
        estimated=budget >= 2**n_players,
        estimation_budget=budget,
        baseline_value=float(baseline_value),
        target_index=target_index,
    )

    # 3. Optional adjustment of the proxy. The adjustment approximator re-samples the same
    # coalitions (ProxySHAP fixes a shared random_state), so the residual values stay aligned.
    if adjustment_method is not None:
        residual_values = coalition_values - fitted.predict(coalitions_matrix)
        residual_values -= residual_values[0]  # Normalize residuals
        residual_game = ResidualGame(n_players=n_players, game_values=residual_values)
        proxy_interactions += adjustment_method.approximate(budget, residual_game)
    proxy_interactions.baseline_value = baseline_value
    proxy_interactions.interactions[()] = baseline_value  # Ensure empty coalition value is correct

    return proxy_interactions


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


class ProxySHAP(Approximator[ValidProxySHAPIndices]):
    """ProxySHAP is a proxy-based approximator that uses a regression model to approximate the value function and applies an adjustment method to better match the true value function.

    It extends RegressionMSR able to compute any-order cardinal-probabilistic indices and supports multiple adjustment methods, including MSR, SVARMIQ, and KernelSHAPIQ.

    The regression model is trained on a subset of the coalitions, and its predictions are adjusted using the selected method to better match the true value function.

    Example:
        >>> from shapiq_games.synthetic import DummyGame
        >>> from shapiq.approximator import ProxySHAP
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = ProxySHAP(n=5, max_order=2, index="k-SII", adjustment="svarm")
        >>> approximator.approximate(budget=100, game=game)
        InteractionValues(
            index=k-SII, max_order=2, estimated=True, estimation_budget=100
        )
    """

    def __init__(
        self,
        n: int,
        *,
        max_order: int = 2,
        index: ValidProxySHAPIndices = "k-SII",
        proxy_model: ProxyModel | ProxyModelWithHPO | ProxyLiteral = "xgboost",
        adjustment: str = "msr",
        sampling_weights: FloatVector | None = None,
        pairing_trick: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the ProxySHAP approximator.

        Args:
            n: Number of features (players).
            max_order: Maximum order of interactions to consider.
            index: Index of the instance to explain.
            proxy_model: Optional proxy model to use for approximating the value function. If None, a default XGBoost regressor will be used.
                We support HPO of tree-models, via sklearn's GridSearchCV, RandomizedSearchCV, and HalvingGridSearchCV. In this case, the ``.best_estimator_`` will be used as the proxy model for interaction extraction and adjustment.
            adjustment: Method for adjusting the proxy model's predictions to better match the true value function. Options are "none" (no adjustment), "msr","svarm" (statified MSR), "kernel" (KernelSHAPIQ).
            sampling_weights: Optional array of weights for the sampling procedure. The weights must be of shape (n + 1,) and are used to determine the probability of sampling a coalition. Defaults to None.
            pairing_trick: If True, the pairing trick is applied to the sampling procedure. Defaults to True.
            random_state: The random state of the estimator. Defaults to None, which is internally
                replaced by a fixed seed (0). ProxySHAP and its residual-adjustment approximator
                use *separate* coalition samplers, and the residual correction most beneficial when they use the same coalitions. A shared, fixed seed
                guarantees this alignment; with ``random_state=None`` the two samplers would diverge
                and the adjustment would be applied to mismatched coalitions. Pass an explicit
                integer to control the (still shared) seed; passing ``None`` keeps results
                deterministic across runs.
        """
        if random_state is None:
            # ProxySHAP and the adjustment approximator must sample the *same* coalitions for the
            # residual correction to align; a shared fixed seed enforces this (see docstring).
            random_state = 0
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=random_state,
            initialize_dict=False,
        )
        self._sampling_weights = sampling_weights
        self._pairing_trick = pairing_trick
        self.set_adjustment_method(adjustment)
        if isinstance(proxy_model, str):
            self.proxy_model: ProxyModel | ProxyModelWithHPO = _select_base_proxy_via_string(
                proxy_model, random_state
            )
        else:
            self.proxy_model: ProxyModel | ProxyModelWithHPO = proxy_model

    def set_adjustment_method(self, adjustment: str) -> None:
        """Select the method for adjusting the proxy model's predictions."""
        if adjustment not in {"none", "msr", "svarm", "kernel"}:
            msg = f"Invalid adjustment method: {adjustment}"
            raise ValueError(msg)
        self.adjustment = adjustment
        match adjustment:
            case "msr":
                self.adjustment_method = SHAPIQ(
                    n=self.n,
                    max_order=self.max_order,
                    index=self.index,
                    sampling_weights=self._sampling_weights,
                    pairing_trick=self._pairing_trick,
                    random_state=self._random_state,
                )
            case "svarm":
                self.adjustment_method = SVARMIQ(
                    n=self.n,
                    max_order=self.max_order,
                    index=self.index,
                    sampling_weights=self._sampling_weights,
                    pairing_trick=self._pairing_trick,
                    random_state=self._random_state,
                )
            case "kernel":
                if self.index not in KernelSHAPIQ.valid_indices:
                    msg = f"KernelSHAPIQ adjustment is only supported for indices {KernelSHAPIQ.valid_indices}, but got index {self.index}"
                    raise ValueError(msg)
                self.adjustment_method = KernelSHAPIQ(
                    n=self.n,
                    max_order=self.max_order,
                    index=self.index,
                    sampling_weights=self._sampling_weights,
                    pairing_trick=self._pairing_trick,
                    random_state=self._random_state,
                )
            case "none":
                self.adjustment_method = None

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        **kwargs: dict,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximate interaction values, dispatching on the proxy's base estimator type.

        The route is resolved by :func:`proxy_approximate`, which dispatches on the type of the
        proxy's *base* estimator (unwrapped from any HPO wrapper via its ``estimator`` attribute):
        linear models route to :func:`_approximate_linear`, registered tree models to
        :func:`_approximate_tree`.

        Args:
            budget: Number of coalition evaluations to draw.
            game: Coalition game (a :class:`shapiq.game.Game` or any callable
                accepting a binary coalition matrix and returning game values).
            **kwargs: Ignored; present for interface compatibility.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` for orders 0
            through ``self.max_order``.
        """
        # 1. Sample coalitions and fit proxy linear model. Keep track of binary coalition matrix for adjustment.
        self._sampler.sample(int(budget))
        coalitions_matrix = self._sampler.coalitions_matrix
        coalition_values = game(coalitions_matrix)
        baseline_value = coalition_values[0]
        coalition_values -= baseline_value

        return proxy_approximate(
            self.proxy_model,
            coalitions_matrix=coalitions_matrix,
            coalition_values=coalition_values,
            baseline_value=baseline_value,
            max_order=self.max_order,
            approximation_index=self.approximation_index,
            target_index=self.index,
            adjustment_method=self.adjustment_method,
        )
