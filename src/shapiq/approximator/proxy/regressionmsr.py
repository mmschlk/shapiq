"""RegressionMSR is a proxy-based approximator that uses a regression model to approximate the value function and applies the MSR adjustment method."""

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING, Literal

import numpy as np

from shapiq.approximator.proxy._routes import _extract_proxy_interactions, fit_proxy, predict_proxy
from shapiq.interaction_values import InteractionValues

from .proxyshap import ProxySHAP

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.approximator.proxy._models import ProxyLiteral, ProxyModel, ProxyModelWithHPO
    from shapiq.game import Game


ValidRegressionMSRIndices = Literal["SV", "BV"]


def _semivalue_p(n: int, index: ValidRegressionMSRIndices) -> np.ndarray:
    """The probabilistic-value coefficients p_0, ..., p_{n-1} used by the closed-form MSR correction.

    SV (Shapley): ``p_k = k! (n-k-1)! / n!`` for ``k = 0, ..., n-1`` (shapiq's internal Shapley
    semivalue weight; matches ``regressionMSR/utils/p_generator.py``'s ``shapley_distribution``).
    BV (Banzhaf): ``p_k = 1 / 2**(n-1)``, constant in ``k`` (matches ``p_generator.py``'s
    ``banzhaf_distribution``). Computed via arbitrary-precision Python ints (:func:`math.factorial`
    / ``2 ** (n - 1)``) before the final division, so this stays numerically exact (no overflow)
    even for ``n`` well beyond the range where a naive float-factorial computation would overflow.

    Args:
        n: The number of players.
        index: The semivalue to compute coefficients for; ``"SV"`` or ``"BV"``.

    Returns:
        An array of shape ``(n,)`` holding ``p_0, ..., p_{n-1}``.

    Raises:
        ValueError: If ``index`` is not ``"SV"`` or ``"BV"``.
    """
    if index == "SV":
        return np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
    if index == "BV":
        return np.full(n, 1.0 / (2 ** (n - 1)))
    msg = f"No closed-form semivalue weight implemented for index={index!r}."
    raise ValueError(msg)


class RegressionMSR(ProxySHAP):
    """RegressionMSR is a proxy-based approximator that uses a regression model to approximate the value function and applies the MSR adjustment method.

    This is the "k=1, all-samples" variant of the MSR (Model-based Shapley Regression) estimator of
    :cite:t:`Witter.2025`: a single proxy model is trained on *all* of the sampled coalitions, its
    exact Shapley (or Banzhaf) values are read out via :class:`~shapiq.approximator.proxy.proxyshap.ProxySHAP`'s
    proxy extraction, and a closed-form Horvitz-Thompson correction for the proxy's residual is added
    on top, evaluated on the *same* coalitions the proxy was fit on (no held-out fold, no second
    sampling round). The estimator is exact at full budget (every coalition is sampled, so the
    residual is the proxy's exact in-sample error against the fully observed game) and is
    consistent -- but not proven unbiased -- at partial budgets, because the same coalitions are used
    both to fit the proxy and to estimate its residual; see :attr:`train_residual_ratio` and
    :attr:`correction_norm`, which diagnose how much the correction contributes. The paper's ``k >=
    2`` cross-fitting variant, which holds out a fold from the proxy fit to recover unbiasedness, is
    not implemented here.

    The regression model is trained on a subset of the coalitions, and its predictions are adjusted using the MSR method to better match the true value function.
    The method was proposed by Witter et al. (2025) :cite:t:`Witter.2025` and is designed to provide more accurate approximations of the Shapley values, especially in cases where the value function is complex and non-linear.

    Attributes:
        train_residual_ratio: Set after :meth:`approximate`. The ratio of the proxy's residual norm
            to the (baseline-normalized) game-value norm over the training coalitions, ``||v -
            f_hat||_2 / ||v - v(empty)||_2``; near ``0`` means the proxy interpolates its training
            coalitions and the MSR correction is nearly vanishing (see
            :cite:t:`Witter.2025`, ``regressionMSR/estimators/regMSR_all.py:96-101``).
        correction_norm: Set after :meth:`approximate`. The Euclidean norm of the MSR correction
            vector that was added on top of the proxy's own singleton attributions (``||correction||_2``;
            see :cite:t:`Witter.2025`, ``regressionMSR/estimators/regMSR_all.py:141``).

    Example:
        >>> from shapiq_games.synthetic import DummyGame
        >>> from shapiq.approximator import RegressionMSR
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = RegressionMSR(n=5, index="SV")
        >>> approximator.approximate(budget=100, game=game)
        InteractionValues(
            index=SV, max_order=1, min_order=0, estimated=False, estimation_budget=32,
            n_players=5, baseline_value=0.0
        )

    """

    def __init__(
        self,
        n: int,
        index: ValidRegressionMSRIndices,
        *,
        proxy_model: ProxyModel | ProxyModelWithHPO | ProxyLiteral = "xgboost",
        sampling_weights: np.ndarray | None = None,
        pairing_trick: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the RegressionMSR approximator.

        Args:
            n: The number of players in the game.
            index: The index to be approximated. Must be a valid index for the chosen adjustment method.
            proxy_model: The model used as the proxy. Either an estimator/HPO wrapper or a
                string tag (``"xgboost"`` (default), ``"lightgbm"``, ``"tree"``, ``"linear"``);
                see :class:`~shapiq.approximator.proxy.proxyshap.ProxySHAP` for details.
            sampling_weights: The sampling weights for the coalitions, of shape ``(n + 1,)``. If
                None, the bowl-shaped default ``1 / (s * (n - s))`` is used (with the empty and
                grand coalition sizes forced to be sampled), i.e. the same
                :meth:`~shapiq.approximator.base.Approximator._init_sampling_weights` default every
                other approximator uses -- not uniform weights.
            pairing_trick: Whether to use the pairing trick for sampling coalitions. Default is True.
            random_state: The random state for reproducibility. Default is None.

        """
        super().__init__(
            n=n,
            max_order=1,
            index=index,
            proxy_model=proxy_model,
            adjustment="msr",
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=random_state,
        )

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        **kwargs: dict,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximate SV/BV via the proxy readout plus a closed-form MSR correction.

        This overrides :meth:`~shapiq.approximator.proxy.proxyshap.ProxySHAP.approximate`. It
        replicates its steps 1-2 (sample/evaluate, fit/extract) exactly, but replaces step 3 --
        which in the base class re-samples an *independent* :class:`~shapiq.approximator.sampling.CoalitionSampler`
        and runs a generic Monte-Carlo estimator (:class:`~shapiq.approximator.montecarlo.shapiq.SHAPIQ`)
        on the residual game -- with the direct closed-form Horvitz-Thompson correction, computed
        from the coalitions already drawn in step 1 (no second sampling round, no second
        estimator). This is what makes the "msr" adjustment fast: the residual correction never
        pays for a second coalition sampler or a second Monte-Carlo pass.

        The formula, for each player ``i``, is::

            correction[i] = sum_{S in samples, i in S}     residual(S) * p_{|S|-1} * D(S)
                           - sum_{S in samples, i not in S} residual(S) * p_{|S|}   * D(S)

        where ``residual(S) = (true value - proxy prediction)`` on ``S`` (baseline-normalized),
        ``p_k`` is the semivalue coefficient for the target index (SV or BV, see
        :func:`_semivalue_p`), and ``D(S) = self._sampler.sampling_adjustment_weights`` is shapiq's
        own Horvitz-Thompson inverse-probability weight (``1 / pi_S``) for the *same*
        sampler/coalitions used in step 1 -- i.e. exactly the "msr" adjustment's own ``D(S)``, not a
        different sampling scheme.

        The result is assembled as a fresh :class:`~shapiq.interaction_values.InteractionValues`
        (never mutating the proxy's own result): a singleton missing from the proxy's extracted
        values (e.g. a tree proxy that never split on a feature at a tiny budget) is treated as
        ``0`` before the correction is added, exactly like the generic path would treat an interaction
        the adjustment approximator never touched.

        Only valid for the exact configuration :class:`RegressionMSR` always uses (``max_order=1``,
        ``adjustment="msr"``, ``index`` in ``{"SV", "BV"}``); this class permits nothing else, so no
        runtime guard is needed here beyond what :func:`_semivalue_p` already raises for an unknown
        index.

        Args:
            budget: Number of coalition evaluations to draw.
            game: Coalition game (a :class:`shapiq.game.Game` or any callable
                accepting a binary coalition matrix and returning game values).
            **kwargs: Ignored; present for interface compatibility.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` of order 1 (the singletons plus
            the empty coalition).
        """
        # 1. Sample coalitions and evaluate the game (identical to ProxySHAP.approximate()).
        self._sampler.sample(int(budget))
        coalitions_matrix = self._sampler.coalitions_matrix
        coalition_values = game(coalitions_matrix)
        baseline_value = coalition_values[0]
        coalition_values = coalition_values - baseline_value
        n_samples, n_players = coalitions_matrix.shape

        # 2. Fit the proxy, then read interactions out of the fitted model (identical).
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

        # 3. Closed-form MSR correction (replaces the re-sampled Monte-Carlo detour) and its
        # diagnostics (B5), computed on the base-class-normalized residual (residual - residual[0],
        # matching ProxySHAP.approximate()'s "Normalize residuals" step) so that both the correction
        # and the diagnostics describe the same quantity.
        proxy_predictions = predict_proxy(fitted, coalitions_matrix, max_order=self.max_order)
        residual_values = coalition_values - proxy_predictions
        residual_values = residual_values - residual_values[0]  # normalize, matches base class

        y_norm = float(np.linalg.norm(coalition_values))
        residual_norm = float(np.linalg.norm(residual_values))
        self.train_residual_ratio = residual_norm / y_norm if y_norm > 0 else float("nan")

        p = _semivalue_p(n_players, self.index)
        sizes = coalitions_matrix.sum(axis=1).astype(int)
        d_weights = self._sampler.sampling_adjustment_weights
        p_below = np.concatenate(([0.0], p))[sizes]  # p_{|S|-1}
        p_above = np.concatenate((p, [0.0]))[sizes]  # p_{|S|}

        correction = np.zeros(n_players)
        for i in range(n_players):
            in_s = coalitions_matrix[:, i] == 1
            correction[i] = np.sum(
                residual_values[in_s] * p_below[in_s] * d_weights[in_s]
            ) - np.sum(residual_values[~in_s] * p_above[~in_s] * d_weights[~in_s])
        self.correction_norm = float(np.linalg.norm(correction))

        # 4. Assemble a FRESH InteractionValues (never mutate proxy_interactions): a singleton
        # missing from the proxy's extracted values counts as 0 before the correction is added.
        interactions = proxy_interactions.interactions.copy()
        for i in range(n_players):
            interactions[(i,)] = interactions.get((i,), 0.0) + correction[i]
        interactions[()] = baseline_value

        # estimated/estimation_budget are computed directly from n_samples (the number of
        # distinct coalitions this call actually evaluated), rather than read off
        # proxy_interactions: for index="BV" with proxy_model="linear", the linear route's
        # MoebiusConverter(...).compute(index="BV", ...) step (_routes.py's _extract_linear)
        # does not carry these fields through -- proxy_interactions.estimated is always True and
        # proxy_interactions.estimation_budget is always None for that one (index, proxy_model)
        # combination, even at full budget (confirmed via run_logs/pr_b/check_linear_metadata.py,
        # job 284305). This is a pre-existing bug in the linear extraction route (same family as
        # the pre-existing BV+linear issue in item 6 of the brief), unrelated to the MSR
        # correction; computing these fields ourselves from data we already have avoids
        # depending on it.
        result = InteractionValues(
            values=interactions,
            index=proxy_interactions.index,
            max_order=proxy_interactions.max_order,
            n_players=n_players,
            min_order=proxy_interactions.min_order,
            estimated=n_samples < 2**n_players,
            estimation_budget=n_samples,
            baseline_value=baseline_value,
        )
        return result
