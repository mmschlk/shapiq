"""ProxySHAP approximator class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq.approximator.base import Approximator
from shapiq.approximator.montecarlo.shapiq import SHAPIQ
from shapiq.approximator.montecarlo.svarmiq import SVARMIQ
from shapiq.approximator.proxy._models import (
    ProxyLiteral,
    ProxyModel,
    ProxyModelWithHPO,
    _select_base_proxy_via_string,
    _wrap_in_default_hpo,
)
from shapiq.approximator.proxy._routes import (
    ResidualGame,
    ValidProxySHAPIndices,
    _extract_proxy_interactions,
    fit_proxy,
    predict_proxy,
)
from shapiq.approximator.regression.kernelshapiq import KernelSHAPIQ

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from shapiq.game import Game
    from shapiq.interaction_values import InteractionValues
    from shapiq.typing import FloatVector


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
            index=k-SII, max_order=2, estimated=False, estimation_budget=100
        )
    """

    def __init__(
        self,
        n: int,
        *,
        max_order: int = 2,
        index: ValidProxySHAPIndices = "k-SII",
        proxy_model: ProxyModel | ProxyModelWithHPO | ProxyLiteral = "xgboost",
        hpo: bool = False,
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
            hpo: If ``True``, wrap a string-resolved gradient-boosting proxy (``"xgboost"`` /
                ``"lightgbm"``) in its default grid search (the HPO-informed proxy). Defaults to
                ``False`` (a bare estimator). Has no effect when ``proxy_model`` is a passed-in
                estimator/wrapper, or for the ``"tree"`` / ``"linear"`` tags.
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
        if isinstance(proxy_model, ProxyModel):
            self.proxy_model: ProxyModel | ProxyModelWithHPO = proxy_model
        else:
            resolved = _select_base_proxy_via_string(proxy_model, random_state)
            # ``hpo`` wraps a resolved boosting backend in its default grid search (the
            # HPO-informed proxy); a DecisionTree fallback is left unwrapped by the helper.
            self.proxy_model = _wrap_in_default_hpo(resolved) if hpo else resolved

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

        The proxy is fit by :func:`fit_proxy` (which selects the feature transform from the base
        estimator type and unwraps any HPO wrapper). Interactions are then read out of the *fitted*
        model by :func:`_extract_proxy_interactions`, which dispatches on its type: linear models
        route to :func:`_extract_linear`, registered tree models to :func:`_extract_tree`. The
        optional residual adjustment and baseline fix are applied here. The adjustment approximator
        re-samples the same coalitions (ProxySHAP fixes a shared ``random_state``), so the residuals
        stay aligned with the proxy's predictions on the features it was fit on.

        Args:
            budget: Number of coalition evaluations to draw.
            game: Coalition game (a :class:`shapiq.game.Game` or any callable
                accepting a binary coalition matrix and returning game values).
            **kwargs: Ignored; present for interface compatibility.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` for orders 0
            through ``self.max_order``.
        """
        # 1. Sample coalitions and evaluate the game. Keep the binary coalition matrix for adjustment.
        self._sampler.sample(int(budget))
        coalitions_matrix = self._sampler.coalitions_matrix
        coalition_values = game(coalitions_matrix)
        baseline_value = coalition_values[0]
        coalition_values -= baseline_value
        n_samples = coalitions_matrix.shape[0]
        n_players = coalitions_matrix.shape[1]

        # 2. Fit the proxy, then read interactions out of the fitted model (dispatch on its type).
        fitted = fit_proxy(
            self.proxy_model, coalitions_matrix, coalition_values, max_order=self.max_order
        )
        proxy_interactions = _extract_proxy_interactions(
            fitted,
            baseline_value=baseline_value,
            max_order=self.max_order,
            approximation_index=self.approximation_index,
            target_index=self.index,
            budget=n_samples,
            n_players=n_players,
        )

        # 3. Apply the optional residual adjustment and fix the empty-coalition/baseline value.
        if self.adjustment_method is not None:
            proxy_predictions = predict_proxy(fitted, coalitions_matrix, max_order=self.max_order)
            residual_values = coalition_values - proxy_predictions
            residual_values -= residual_values[0]  # Normalize residuals
            residual_game = ResidualGame(n_players=n_players, game_values=residual_values)
            proxy_interactions += self.adjustment_method.approximate(n_samples, residual_game)
        proxy_interactions.baseline_value = baseline_value
        proxy_interactions.interactions[()] = baseline_value  # Ensure empty coalition is correct
        return proxy_interactions
