"""RegressionMSR is a proxy-based approximator that uses a regression model to approximate the value function and applies the MSR adjustment method."""

from __future__ import annotations

from math import exp, factorial, lgamma, log
from typing import TYPE_CHECKING, Literal

import numpy as np

from shapiq.approximator.proxy._routes import (
    _base_estimator,
    _extract_proxy_interactions,
    fit_proxy,
    predict_proxy,
)
from shapiq.interaction_values import InteractionValues
from shapiq.utils.modules import safe_isinstance

from .proxyshap import ProxySHAP

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.approximator.proxy._models import ProxyLiteral, ProxyModel, ProxyModelWithHPO
    from shapiq.game import Game


ValidRegressionMSRIndices = Literal["SV", "BV"]


def _semivalue_p(n: int, index: ValidRegressionMSRIndices) -> np.ndarray:
    """The probabilistic-value coefficients p_0, ..., p_{n-1} used by the closed-form MSR correction.

    SV (Shapley): ``p_k = k! (n-k-1)! / n!`` for ``k = 0, ..., n-1`` (shapiq's internal Shapley
    semivalue weight; matches :cite:t:`Witter.2025`'s Shapley probabilistic-value coefficients).
    BV (Banzhaf): ``p_k = 1 / 2**(n-1)``, constant in ``k`` (matches the same reference's Banzhaf
    coefficients). Computed via arbitrary-precision Python ints (:func:`math.factorial`
    / ``2 ** (n - 1)``) before the final division, so this stays numerically exact (no overflow)
    even for ``n`` well beyond the range where a naive float-factorial computation would overflow.

    Args:
        n: The number of players.
        index: The semivalue to compute coefficients for; ``"SV"`` or ``"BV"``.

    Returns:
        An array of shape ``(n,)`` holding ``p_0, ..., p_{n-1}``.

    Raises:
        ValueError: If ``index`` is not ``"SV"`` or ``"BV"``. :meth:`RegressionMSR.approximate`
            never triggers this branch itself: it checks ``self.index`` and delegates to
            :meth:`ProxySHAP.approximate() <shapiq.approximator.proxy.proxyshap.ProxySHAP.approximate>`
            *before* calling this function for any index other than ``"SV"``/``"BV"``. This branch
            is exercised directly (e.g. ``test_semivalue_p_raises_for_unsupported_index``) so it
            stays covered without depending on that guard never changing.
    """
    if index == "SV":
        return np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
    if index == "BV":
        return np.full(n, 1.0 / (2 ** (n - 1)))
    msg = f"No closed-form semivalue weight implemented for index={index!r}."
    raise ValueError(msg)


def _proxy_selects_linear_kernel(
    proxy_model: ProxyModel | ProxyModelWithHPO | ProxyLiteral,
) -> bool:
    """Whether ``proxy_model`` selects (or resolves to) a linear-in-features proxy.

    Mirrors :cite:t:`Witter.2025`'s reference condition, ``reg_model_class == 'linear'`` (the
    reference's ``UniversalMSR.__init__``), which only ever sees a string tag, since the
    reference's own ``regression_adj``/``reg_model_class`` parameter is always a string.
    shapiq's ``proxy_model`` additionally allows passing a resolved estimator directly (optionally
    wrapped in an HPO search), bypassing the ``"linear"`` string tag entirely
    (:meth:`~shapiq.approximator.proxy.proxyshap.ProxySHAP.__init__`'s
    ``isinstance(proxy_model, ProxyModel)`` branch) -- such an estimator is still routed through
    the exact same linear coefficient read-out as the ``"linear"`` tag
    (:func:`~shapiq.approximator.proxy._routes._extract_linear`, registered on
    ``"sklearn.linear_model._base.LinearModel"``), so it is treated identically here: an HPO
    wrapper is unwrapped to its *unfitted* base estimator first
    (:func:`~shapiq.approximator.proxy._routes._base_estimator`, the same unwrap
    :func:`~shapiq.approximator.proxy._routes.fit_proxy` itself uses to pick a feature transform),
    then checked against that same class family.

    Args:
        proxy_model: The raw, not-yet-resolved ``proxy_model`` argument as passed to
            :meth:`RegressionMSR.__init__` (i.e. *before*
            :meth:`~shapiq.approximator.proxy.proxyshap.ProxySHAP.__init__` resolves a string tag
            into an estimator).

    Returns:
        ``True`` if the linear-proxy kernel applies (see :func:`_paper_sampling_weights`),
        ``False`` if the generic kernel applies instead.
    """
    if isinstance(proxy_model, str):
        return proxy_model == "linear"
    return safe_isinstance(_base_estimator(proxy_model), "sklearn.linear_model._base.LinearModel")


def _paper_sampling_weights(
    n: int,
    index: ValidRegressionMSRIndices,
    proxy_model: ProxyModel | ProxyModelWithHPO | ProxyLiteral = "xgboost",
) -> np.ndarray:
    r"""The default ``sampling_weights`` of :class:`RegressionMSR`: the paper's own sampling kernel.

    Reproduces :cite:t:`Witter.2025`'s ``UniversalMSR`` sampling density ``sample_dist`` exactly --
    a per-INDIVIDUAL-COALITION density, normalized to sum to 1, then converted to shapiq's
    per-SIZE probability-mass convention the same way the reference itself does before drawing a
    coalition size: ``mass[s] = D(s) * C(n, s)``, renormalized. shapiq's ``sampling_weights[s]`` is
    this per-SIZE probability mass, not the reference's per-coalition density -- see the
    ``sampling_weights`` parameter's docstring below for the conversion, needed if porting in a
    *different* density than this default.

    The reference defines **two** kernels for ``D(s)``, selected by ``reg_model_class`` (shapiq's
    ``proxy_model``, see :func:`_proxy_selects_linear_kernel`):

    * **Generic** (any ``proxy_model`` other than a linear one -- the default, e.g.
      ``"xgboost"``/``"lightgbm"``/``"tree"``): the ``UniversalMSR``/``TreeMSRAll`` kernel
      ``D(s) = sqrt(p_{s-1}^2 * s * [s>0] + p_s^2 * (n-s) * [s<n])``, where ``p_0, ..., p_{n-1}``
      are :func:`_semivalue_p`'s semivalue coefficients for the target ``index``.
    * **Linear** (``proxy_model`` is (or resolves to) a linear-in-features proxy): for ``index=
      "SV"``, the Leverage-SHAP density ``D(s) = (p_s + p_{s-1}) * s * (n - s)`` for
      ``s = 1, ..., n - 1`` and ``D(0) = D(n) = 0`` -- *not* the generic formula above, which the
      reference only uses for non-linear proxies. For ``index="BV"``, the reference instead uses a
      constant density (``ones_like``); this happens to be mathematically identical (not merely
      proportional) to the generic formula for Banzhaf, since Banzhaf's ``p_k`` is constant in
      ``k`` (verified in ``test_bv_linear_default_equals_generic_and_uniform``), so this function
      does not special-case ``index="BV"`` at all -- the generic branch already reproduces it
      exactly.

    Computed entirely in log-space (:func:`math.lgamma`, :func:`numpy.logaddexp`) rather than by
    squaring/summing/dividing raw floats, so it stays finite and correctly normalized for ``n`` up
    to (at least) the low thousands: a direct float computation of ``p_k ** 2`` (or, for the linear
    kernel, ``p_k`` itself) underflows to exactly ``0.0`` for essentially every ``k`` once ``n`` is
    a few hundred or more -- e.g. Banzhaf's constant ``p_k = 1 / 2 ** (n - 1)`` squared underflows
    below the smallest subnormal ``float64`` once ``n > ~538`` -- which would silently zero out the
    entire distribution and divide-by-zero on renormalization.

    Args:
        n: The number of players.
        index: ``"SV"`` or ``"BV"`` (the only indices :func:`_semivalue_p` supports).
        proxy_model: The raw, not-yet-resolved ``proxy_model`` argument (see
            :func:`_proxy_selects_linear_kernel`); selects between the linear and generic kernels
            above. Defaults to ``"xgboost"`` (a non-linear proxy, i.e. the generic kernel), the
            same default :class:`RegressionMSR` itself uses for ``proxy_model``.

    Returns:
        An array of shape ``(n + 1,)``, indexed by coalition size ``0, ..., n``: non-negative and
        summing to 1 (up to floating-point rounding).

    Raises:
        ValueError: If ``index`` is not ``"SV"`` or ``"BV"``. :class:`RegressionMSR` never
            triggers this: :meth:`RegressionMSR.__init__` only calls this function when
            ``index in ("SV", "BV")``.
    """
    if index == "SV":
        log_p = np.array([lgamma(k + 1) + lgamma(n - k) - lgamma(n + 1) for k in range(n)])
    elif index == "BV":
        log_p = np.full(n, -(n - 1) * log(2))
    else:
        msg = f"No closed-form paper sampling kernel implemented for index={index!r}."
        raise ValueError(msg)

    if index == "SV" and _proxy_selects_linear_kernel(proxy_model):
        # Leverage-SHAP density: D(s) = (p_s + p_{s-1}) * s * (n - s) for s = 1, ..., n - 1;
        # D(0) = D(n) = 0. log(p_s + p_{s-1}) via np.logaddexp (not log(exp(.)+exp(.))) so the sum
        # never overflows/underflows before being combined -- same underflow rationale as the
        # generic branch's manual logsumexp below, just for a plain sum instead of a sum of
        # squares.
        log_density = np.full(n + 1, -np.inf)
        for size in range(1, n):
            log_density[size] = (
                np.logaddexp(log_p[size], log_p[size - 1]) + log(size) + log(n - size)
            )
    else:
        # Generic density (also the reference's own kernel for proxy_model=<linear> +
        # index="BV" -- see this function's docstring). log(prob[size]) =
        # log( p_{size-1}^2 * size + p_size^2 * (n - size) ), via a manual (>=1, <=2-term)
        # logsumexp so the (up to) two summands never overflow/underflow independently before
        # being combined.
        log_prob = np.empty(n + 1)
        for size in range(n + 1):
            terms = []
            if size > 0:
                terms.append(2 * log_p[size - 1] + log(size))
            if size < n:
                terms.append(2 * log_p[size] + log(n - size))
            m = max(terms)
            log_prob[size] = m + log(sum(exp(t - m) for t in terms))
        log_density = 0.5 * log_prob  # sqrt(.) in log-space

    log_comb = np.array([lgamma(n + 1) - lgamma(s + 1) - lgamma(n - s + 1) for s in range(n + 1)])
    log_mass = log_density + log_comb  # D(s) * C(n, s), in log-space
    mass = np.exp(log_mass - log_mass.max())  # numerically stable softmax-style normalization
    return mass / mass.sum()


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
        train_residual_ratio: ``None`` until a fast-path :meth:`approximate` call (``index in
            ("SV", "BV")``) sets it; stays ``None`` if ``approximate`` instead falls back to
            :meth:`ProxySHAP.approximate() <shapiq.approximator.proxy.proxyshap.ProxySHAP.approximate>`
            for any other index. Once set: the ratio of the proxy's residual norm to the
            game-value norm over the training coalitions, ``||v - f_hat||_2 / ||v||_2``, using the
            *raw* game values ``v`` (i.e. ``game(coalitions)``, before shapiq's own
            baseline-shift-to-zero-at-the-empty-coalition convention is applied for fitting), which
            matches :cite:t:`Witter.2025`'s own definition exactly: ``residual = y - reg_pred``,
            ``train_residual_ratio = ||residual||_2 / ||y||_2`` on the raw model outputs ``y``.
            Near ``0`` means the proxy interpolates its training coalitions and the MSR correction
            is nearly vanishing. Using shapiq's baseline-shifted ``v - v(empty)`` instead of the
            raw ``v`` would diverge from the reference by up to ~6x whenever ``v(empty) != 0``
            (real games almost always have ``v(empty) != 0``).
        correction_norm: ``None`` until a fast-path :meth:`approximate` call sets it (see
            ``train_residual_ratio`` above for when that is). Once set: the Euclidean norm of the
            MSR correction vector that was added on top of the proxy's own singleton attributions
            (``||correction||_2``; see :cite:t:`Witter.2025`).

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
                ``None`` and ``index in ("SV", "BV")``, the default is :cite:t:`Witter.2025`'s own
                sampling kernel (:func:`_paper_sampling_weights`) -- *not* the bowl-shaped
                ``1 / (s * (n - s))`` default every other :class:`~shapiq.approximator.base.Approximator`
                subclass uses (:meth:`~shapiq.approximator.base.Approximator._init_sampling_weights`).
                For any other ``index`` (the fast path does not cover it; see
                :meth:`approximate`), ``None`` falls back to that same bowl-shaped default instead,
                since no closed-form paper kernel is defined outside ``{"SV", "BV"}``.

                The reference actually defines **two** kernels, selected by whether ``proxy_model``
                is (or resolves to) a linear-in-features proxy (see
                :func:`_proxy_selects_linear_kernel`) -- in shapiq's per-SIZE probability-mass
                convention (``mass[s]`` below):

                * Non-linear ``proxy_model`` (the default, e.g. ``"xgboost"``): ``D(s) =
                  sqrt(p_{s-1}^2 * s * [s>0] + p_s^2 * (n - s) * [s<n])``.
                * Linear ``proxy_model`` (e.g. ``proxy_model="linear"``) with ``index="SV"``: the
                  Leverage-SHAP density ``D(s) = (p_s + p_{s-1}) * s * (n - s)`` for
                  ``s = 1, ..., n - 1`` and ``D(0) = D(n) = 0``. With ``index="BV"`` the reference
                  uses a constant density instead, which is mathematically identical to the
                  non-linear formula above for Banzhaf (constant ``p_k``), so this case is not
                  distinguished separately.

                Either way: ``D(s)`` is normalized to sum to 1, then ``mass[s] = D(s) * C(n, s)``,
                renormalized to sum to 1 -- where ``p_0, ..., p_{n-1}`` are the semivalue
                coefficients for ``index`` (see :func:`_semivalue_p`). This exactly reproduces
                :cite:t:`Witter.2025`'s reference ``sample_dist`` (for the ``proxy_model`` actually
                passed), converted the way the reference itself converts it before sampling a size.
                Empirically, the non-linear kernel is markedly more accurate than the previous
                bowl-shaped default at large budgets -- about 25% lower pooled error across an
                8-dataset benchmark grid (pooled ratio-to-reference 0.972 vs. 1.32).

                ``sampling_weights[s]`` is a per-SIZE probability mass (``P(a drawn coalition has
                size s)``), the convention every other :class:`~shapiq.approximator.base.Approximator`
                subclass uses -- not a per-individual-coalition density. :cite:t:`Witter.2025`'s
                reference instead specifies a per-coalition density ``D(s)``; porting a *different*
                density here (rather than using this class's built-in default) requires first
                converting via ``sampling_weights = D * binom(n, np.arange(n + 1))``, renormalized
                to sum to 1, or the sampler silently starves middle coalition sizes. Passing an
                explicit array here always overrides this class's default, for either code path
                above.
            pairing_trick: Whether to use the pairing trick for sampling coalitions. Default is True.
            random_state: The random state for reproducibility. Default is None.

        """
        if sampling_weights is None and index in ("SV", "BV"):
            # The paper's own kernel, not the generic bowl-shaped Approximator default -- see
            # _paper_sampling_weights and this parameter's docstring above. Only defined for the
            # two indices the closed-form fast path covers; any other index falls through to
            # `sampling_weights=None`, letting Approximator.__init__ apply its usual bowl-shaped
            # default exactly as before. `proxy_model` is passed through as-is (not yet resolved
            # into an estimator -- that happens below, in super().__init__()) so
            # _paper_sampling_weights can select the linear-proxy kernel when it applies; see
            # _proxy_selects_linear_kernel.
            sampling_weights = _paper_sampling_weights(n, index, proxy_model)
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
        # Declared here (not just assigned inside `approximate`) so both attributes always exist,
        # even before `approximate` is ever called or when it falls back to
        # `ProxySHAP.approximate()` for an index outside the closed-form fast path's
        # `{"SV", "BV"}` -- see the class docstring's `Attributes:` section.
        self.train_residual_ratio: float | None = None
        self.correction_norm: float | None = None

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        **kwargs: dict,
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

        The closed-form correction only has a semivalue weight formula (:func:`_semivalue_p`) for
        ``index in {"SV", "BV"}``. For any other index -- e.g. ``RegressionMSR`` constructed
        directly with ``index="SII"`` (the base :class:`~shapiq.approximator.proxy.proxyshap.ProxySHAP`
        validation this class inherits does not narrow ``valid_indices``, so construction with such
        an index succeeds), or with ``self.index`` mutated after construction -- this method checks
        ``self.index`` *before* sampling or fitting anything and delegates the entire call to
        :meth:`ProxySHAP.approximate() <shapiq.approximator.proxy.proxyshap.ProxySHAP.approximate>`,
        the generic re-sampled Monte-Carlo path, so no coalitions are sampled and no proxy is fit for
        a configuration the fast path cannot handle. This also means an index the *parent* itself
        rejects (e.g. ``"BII"``, which is not implemented for ``adjustment="msr"`` upstream) now
        raises the same error the parent raises, rather than a different, closed-form-specific
        ``ValueError`` from :func:`_semivalue_p`.

        Args:
            budget: Number of coalition evaluations to draw.
            game: Coalition game (a :class:`shapiq.game.Game` or any callable
                accepting a binary coalition matrix and returning game values).
            **kwargs: Passed through to :meth:`ProxySHAP.approximate` when delegating for an
                unsupported index; ignored otherwise (present for interface compatibility).

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` of order 1 (the singletons plus
            the empty coalition).
        """
        # 0. The closed-form correction below only covers index in {"SV", "BV"} (see
        # _semivalue_p). Check this *before* sampling or fitting anything, so an unsupported
        # index never wastes a sampling/fitting pass before falling back.
        if self.index not in ("SV", "BV"):
            return super().approximate(budget, game, **kwargs)

        # Reset both diagnostics to None *before* any computation below, not just at the end once
        # both are freshly computed: if this is a second `approximate()` call on the same instance
        # and something raises partway through (between setting train_residual_ratio and
        # correction_norm further down), this guarantees the pair is never left as one fresh value
        # next to one stale value from a prior successful call -- a caller catching the exception
        # always sees either both None or both freshly consistent, never a stale/fresh mix.
        self.train_residual_ratio = None
        self.correction_norm = None

        # 1. Sample coalitions and evaluate the game (identical to ProxySHAP.approximate()).
        self._sampler.sample(int(budget))
        coalitions_matrix = self._sampler.coalitions_matrix
        coalition_values_raw = game(coalitions_matrix)
        baseline_value = coalition_values_raw[0]
        coalition_values = coalition_values_raw - baseline_value
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
        # diagnostics (B5).
        proxy_predictions = predict_proxy(fitted, coalitions_matrix, max_order=self.max_order)
        # raw_residual = v(S) - f_hat(S) is invariant to shapiq's baseline shift (v and f_hat both
        # shift by the same baseline_value, which cancels in the subtraction), so it equals
        # :cite:t:`Witter.2025`'s reference `residual = y - reg_pred` exactly, using the RAW
        # (unshifted) game values.
        raw_residual = coalition_values - proxy_predictions

        # --- diagnostics: train_residual_ratio matches the reference's raw formula exactly
        # (||y - f_hat||_2 / ||y||_2 on the RAW, baseline-inclusive game values y -- not shapiq's
        # own baseline-shifted convention, which diverges from the reference by up to ~6x whenever
        # v(empty) != 0; see :attr:`train_residual_ratio`'s docstring).
        raw_y_norm = float(np.linalg.norm(coalition_values_raw))
        residual_norm = float(np.linalg.norm(raw_residual))
        self.train_residual_ratio = residual_norm / raw_y_norm if raw_y_norm > 0 else float("nan")

        # --- correction: uses shapiq's own base-class residual convention, recentered so the
        # sampled empty coalition has zero residual, exactly matching
        # ProxySHAP.approximate()'s "Normalize residuals" step. This recentering is NOT part of
        # the reference's train_residual_ratio formula (hence computed separately, above, from the
        # non-recentered raw_residual) but IS required here: it is what makes the closed-form
        # correction reproduce the generic Monte-Carlo residual-game path bit-for-bit (verified in
        # test_fast_path_matches_parent_path).
        residual_values = raw_residual - raw_residual[0]

        # self.index is typed at the base class's broader ValidProxySHAPIndices, but the guard
        # above (self.index not in ("SV", "BV") -> delegate) narrows it to Literal["SV", "BV"] for
        # the rest of this method, which ty accepts as a ValidRegressionMSRIndices argument here
        # without a suppression comment (confirmed empirically: a `ty: ignore[invalid-argument-type]`
        # on this line is flagged as an unused-ignore-comment warning by `ty check`).
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
        # combination, even at full budget. This is a pre-existing bug in the linear extraction
        # route, unrelated to the MSR correction; computing these fields ourselves from data we
        # already have avoids depending on it.
        return InteractionValues(
            values=interactions,
            index=proxy_interactions.index,
            max_order=proxy_interactions.max_order,
            n_players=n_players,
            min_order=proxy_interactions.min_order,
            estimated=n_samples < 2**n_players,
            estimation_budget=n_samples,
            baseline_value=baseline_value,
        )
