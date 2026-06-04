"""ProxySHAP approximator class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from shapiq.approximator.base import Approximator
from shapiq.approximator.montecarlo.shapiq import SHAPIQ
from shapiq.approximator.montecarlo.svarmiq import SVARMIQ
from shapiq.approximator.regression.kernelshapiq import KernelSHAPIQ
from shapiq.game import Game
from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.interaction_values import InteractionValues
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
from shapiq.utils.modules import safe_isinstance

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.typing import CoalitionMatrix, FloatVector, GameValues

ValidProxySHAPIndices = Literal["k-SII", "FSII", "FBII", "SII", "SV", "BV"]


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
        proxy_model: object | None = None,
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
        if proxy_model is not None:
            self.proxy_model = proxy_model
        else:
            try:
                from xgboost import XGBRegressor
            except ImportError as e:
                msg = "XGBoost is required for the default proxy model. Install it with: pip install 'shapiq[proxy]' or provide a custom proxy_model that implements the scikit-learn regressor interface."
                raise ImportError(msg) from e
            self.proxy_model = XGBRegressor(random_state=random_state)

        self.set_adjustment_method(adjustment)

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

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        **kwargs: dict,
    ) -> InteractionValues:
        """Approximate interaction values, dispatching by proxy type.

        Routes to :meth:`approximate_linear` for an
        :class:`sklearn.linear_model.LinearRegression` proxy and to
        :meth:`approximate_tree` for everything else.

        Args:
            budget: Number of coalition evaluations to draw.
            game: Coalition game (a :class:`shapiq.game.Game` or any callable
                accepting a binary coalition matrix and returning game values).
            **kwargs: Forwarded to the dispatched method.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` for orders 0
            through ``self.max_order``.
        """
        if safe_isinstance(self.proxy_model, "sklearn.linear_model.LinearRegression"):
            return self.approximate_linear(budget, game, **kwargs)
        return self.approximate_tree(budget, game, **kwargs)

    def approximate_linear(
        self, budget: int, game: Game | Callable[[np.ndarray], np.ndarray], **_: dict
    ) -> InteractionValues:
        """Approximate interactions with a linear-in-features proxy.

        For ``max_order > 1`` the coalition matrix is expanded with
        :class:`sklearn.preprocessing.PolynomialFeatures` (interaction-only) so
        that fitted coefficients map directly to Möbius interactions; the
        result is then converted to ``self.approximation_index`` via
        :class:`~shapiq.game_theory.moebius_converter.MoebiusConverter`.
        Optional residual adjustment is applied to the proxy's residuals on the same coalitions; see :class:`ResidualGame` for how alignment is ensured.

        Args:
            budget: Number of coalition evaluations to draw.
            game: Coalition game.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` for the
            requested index and order.
        """
        # 1. Sample coalitions and fit proxy linear model. Keep track of binary coalition matrix for adjustment.
        self._sampler.sample(int(budget))
        coalitions_matrix = self._sampler.coalitions_matrix
        coalition_values = game(coalitions_matrix)
        baseline_value = coalition_values[0]
        coalition_values -= baseline_value

        # 2. Extract interactions from proxy model coefficients, converting to the correct index if necessary
        linear_interactions: dict[tuple[int, ...], float]
        if self.max_order == 1:
            proxy_features = coalitions_matrix  # linear proxy fits the raw coalitions
            self.proxy_model.fit(  # ty: ignore[unresolved-attribute]
                proxy_features, coalition_values
            )
            linear_interactions = {
                (i,): float(self.proxy_model.coef_[i])  # ty: ignore[unresolved-attribute]
                for i in range(self.n)
            }
        else:
            poly = PolynomialFeatures(
                degree=self.max_order, interaction_only=True, include_bias=False
            )
            proxy_features = poly.fit_transform(coalitions_matrix)  # interaction-only expansion
            self.proxy_model.fit(  # ty: ignore[unresolved-attribute]
                proxy_features, coalition_values
            )
            linear_interactions = extract_linear_interactions(
                coefficients=self.proxy_model.coef_,  # ty: ignore[unresolved-attribute]
                poly=poly,
            )

        proxy_interactions = InteractionValues(
            linear_interactions,
            index=self.approximation_index,
            n_players=self.n,
            min_order=self.min_order,
            max_order=self.max_order,
            baseline_value=float(baseline_value),
            estimated=not budget >= 2**self.n,
            estimation_budget=int(budget),
        )
        proxy_interactions = MoebiusConverter(moebius_coefficients=proxy_interactions).compute(
            index=self.index, order=self.max_order
        )

        # 3. Optional adjustment of the proxy. The adjustment approximator re-samples the same
        # coalitions (ProxySHAP fixes a shared random_state), so the residual values stay aligned.
        if self.adjustment != "none":
            residual_values = (
                coalition_values
                - self.proxy_model.predict(proxy_features)  # ty: ignore[unresolved-attribute]
            )
            residual_values -= residual_values[0]  # Normalize residuals
            residual_game = ResidualGame(n_players=self.n, game_values=residual_values)
            proxy_interactions += self.adjustment_method.approximate(budget, residual_game)
        proxy_interactions.baseline_value = baseline_value
        proxy_interactions.interactions[()] = (
            baseline_value  # Ensure empty coalition value is correct
        )
        return proxy_interactions

    def approximate_tree(
        self, budget: int, game: Game | Callable[[np.ndarray], np.ndarray], **_: dict
    ) -> InteractionValues:
        """Approximate interactions with a tree proxy and exact tree readout.

        Samples ``budget`` coalitions, evaluates the game, fits the tree proxy,
        then reads off interactions exactly via
        :class:`~shapiq.tree.interventional.explainer.InterventionalTreeExplainer`
        in boolean-tree mode. Optional residual adjustment is applied to the proxy's residuals on the same coalitions; see :class:`ResidualGame` for how alignment is ensured.

        Args:
            budget: Number of coalition evaluations to draw.
            game: Coalition game.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` for the
            requested index and order.
        """
        # 1. Sample coalitions and fit proxy tree
        self._sampler.sample(budget)
        coalitions_matrix = self._sampler.coalitions_matrix
        game_values = game(coalitions_matrix)
        baseline_value = game_values[0]  # Value of the empty coalition
        game_values -= baseline_value  # Normalize values

        # 2. Extract interactions from proxy tree
        self.proxy_model.fit(coalitions_matrix, game_values)  # ty: ignore[unresolved-attribute]

        if safe_isinstance(
            self.proxy_model,
            [
                "sklearn.model_selection._search.GridSearchCV",
                "sklearn.model_selection._search.RandomizedSearchCV",
                "sklearn.model_selection._search.HalvingGridSearchCV",
            ],
        ):
            self.proxy_model = self.proxy_model.best_estimator_  # ty: ignore[unresolved-attribute]

        explainer = InterventionalTreeExplainer(
            self.proxy_model,
            data=np.zeros((1, self.n)),  # reference data for boolean tree
            index=self.index,
            max_order=self.max_order,
            bool_tree=True,
        )
        proxy_values = explainer.explain_function(np.ones((1, self.n)))
        proxy_interactions = InteractionValues(
            values=proxy_values.interactions,
            index=self.index,
            max_order=self.max_order,
            n_players=self.n,
            min_order=0,
            estimated=budget >= 2**self.n,
            estimation_budget=budget,
            baseline_value=float(baseline_value),
        )

        # 3. Optional adjustment of the proxy. The adjustment approximator re-samples the same
        # coalitions (ProxySHAP fixes a shared random_state), so the residual values stay aligned.
        if self.adjustment != "none":
            residual_values = (
                game_values
                - self.proxy_model.predict(coalitions_matrix)  # ty: ignore[unresolved-attribute]
            )
            residual_values -= residual_values[0]  # Normalize residuals
            residual_game = ResidualGame(n_players=self.n, game_values=residual_values)
            proxy_interactions += self.adjustment_method.approximate(budget, residual_game)
        proxy_interactions.baseline_value = baseline_value
        proxy_interactions.interactions[()] = (
            baseline_value  # Ensure empty coalition value is correct
        )

        return proxy_interactions
