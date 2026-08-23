"""Tests for the :class:`~shapiq.approximator.proxy.regressionmsr.RegressionMSR` approximator.

Focused on the closed-form MSR fast path (:meth:`RegressionMSR.approximate`), which replaces the
generic :class:`~shapiq.approximator.proxy.proxyshap.ProxySHAP` base class's re-sampled Monte
Carlo residual correction with a direct Horvitz-Thompson correction computed from the coalitions
already sampled to fit the proxy, and on the class's default ``sampling_weights`` (the sampling
kernel of Witter et al., 2025).
"""

from __future__ import annotations

import importlib.util
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import shapiq.approximator.proxy.regressionmsr as regressionmsr_mod
from shapiq.approximator import RegressionMSR
from shapiq.approximator.proxy._routes import _extract_proxy_interactions, fit_proxy, predict_proxy
from shapiq.approximator.proxy.proxyshap import ProxySHAP
from shapiq.approximator.proxy.regressionmsr import _paper_sampling_weights, _semivalue_p
from shapiq.game_theory.exact import ExactComputer
from shapiq_games.synthetic import SOUM, DummyGame

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues

_CATBOOST_AVAILABLE = importlib.util.find_spec("catboost") is not None
if _CATBOOST_AVAILABLE:
    from catboost import CatBoostRegressor


def _budgets_for(n: int) -> list[int]:
    """``{n+2, an odd budget, 2n, 8n, 2**n}``, deduplicated and capped at ``2**n``."""
    raw = [n + 2, 2 * n + 1, 2 * n, 8 * n, 2**n]
    return sorted({b for b in raw if b <= 2**n})


def _pairwise_game(n: int, seed: int):
    """A game with main effects, a few pairwise interactions, and a smooth nonlinearity.

    Used throughout this file to verify the fast path against the generic ``ProxySHAP`` path.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    n_pairs = min(5, max(1, n // 2))
    pairs = [tuple(rng.choice(n, size=2, replace=False)) for _ in range(n_pairs)]
    pair_w = rng.normal(size=len(pairs))

    def game(coalitions: np.ndarray) -> np.ndarray:
        coalitions = np.atleast_2d(coalitions).astype(float)
        val = coalitions @ w
        for (i, j), pw in zip(pairs, pair_w, strict=False):
            val = val + pw * coalitions[:, i] * coalitions[:, j]
        return val + 0.3 * np.sin(coalitions.sum(axis=1))

    return game


def _offset_game(n: int, offset: float = 3.7):
    """A purely linear game with a nonzero intercept, i.e. ``v(empty) == offset != 0``.

    Every other game in this file has ``v(empty) == 0`` by construction, which hides a bug that
    hardcodes or otherwise loses the baseline value, and hides a ``train_residual_ratio``
    definitional mismatch: shapiq's own baseline-shifted convention and the reference's raw
    formula only diverge when ``v(empty) != 0`` (see :attr:`RegressionMSR.train_residual_ratio`'s
    docstring).
    """
    rng = np.random.default_rng(0)
    w = rng.normal(size=n)

    def game(coalitions: np.ndarray) -> np.ndarray:
        coalitions = np.atleast_2d(coalitions).astype(float)
        return offset + coalitions @ w

    return game


def _irrelevant_feature_game(n: int, seed: int, irrelevant: int = 0):
    """A purely linear game where ``irrelevant`` has exactly zero effect on the value."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    w[irrelevant] = 0.0

    def game(coalitions: np.ndarray) -> np.ndarray:
        coalitions = np.atleast_2d(coalitions).astype(float)
        return coalitions @ w

    return game


def _make_tree_game(n: int, seed: int):
    """Tree-friendly synthetic regression game: a fitted ``DecisionTreeRegressor`` used directly
    as ``v(S) = model.predict(S)`` (generalized from shapiq's own
    ``reg_data_coalitions``/``dt_reg_model_coalitions`` test fixtures).
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(200, n)).astype(float)
    weights = rng.standard_normal(n)
    y = x @ weights + 0.1 * rng.standard_normal(200)
    model = DecisionTreeRegressor(random_state=seed, max_depth=min(6, n))
    model.fit(x, y)

    def game(coalitions: np.ndarray) -> np.ndarray:
        coalitions = np.atleast_2d(coalitions).astype(float)
        return model.predict(coalitions)

    return game


def _phi(iv: InteractionValues, n: int) -> np.ndarray:
    return np.array([iv[(i,)] for i in range(n)])


def _relative_squared_error(phi_hat: np.ndarray, phi_exact: np.ndarray) -> float:
    denom = max(float(np.sum(phi_exact**2)), 1e-12)
    return float(np.sum((phi_hat - phi_exact) ** 2) / denom)


_PROXY_TAGS = ["xgboost", "tree", *(["catboost"] if _CATBOOST_AVAILABLE else [])]


def _resolve_proxy(tag: str, seed: int):
    """Resolve a proxy tag to what gets passed as ``proxy_model=``.

    ``"catboost"`` is not a valid :data:`~shapiq.approximator.proxy._models.ProxyLiteral` string
    tag (only ``"xgboost"``, ``"lightgbm"``, ``"tree"``, ``"linear"`` are), so it must be passed
    as an actual (unfitted) ``CatBoostRegressor`` instance instead.
    """
    if tag == "catboost":
        return CatBoostRegressor(verbose=False, random_state=seed)
    return tag


def _equality_cases() -> list[tuple[int, str, int, str]]:
    cases = []
    for n in (5, 8, 10):
        for index in ("SV", "BV"):
            for budget in _budgets_for(n):
                cases.extend((n, index, budget, proxy_tag) for proxy_tag in _PROXY_TAGS)
    return cases


@pytest.mark.parametrize(("n", "index", "budget", "proxy_tag"), _equality_cases())
def test_fast_path_matches_parent_path(n, index, budget, proxy_tag):
    """The closed-form fast path must match the generic ``ProxySHAP(adjustment="msr")`` path.

    Both approximators share a fixed, identical ``random_state``, so they sample the identical
    coalitions in the identical order and fit identically-seeded proxies; the only difference is
    which code computes the residual correction (closed-form vs. a re-sampled Monte Carlo pass).
    The two must therefore agree to numerical precision, not just to the required 1e-10.

    ``RegressionMSR``'s default ``sampling_weights`` (the paper kernel, for ``index in
    {"SV", "BV"}``) now differs from ``ProxySHAP``'s own unchanged bowl-shaped default, so
    ``sampling_weights`` is passed explicitly to both constructors here -- otherwise ``fast`` and
    ``generic`` would sample different coalitions and this comparison would no longer be
    apples-to-apples.
    """
    seed = 0
    game = _pairwise_game(n, seed)
    weights = _paper_sampling_weights(n, index)

    fast = RegressionMSR(
        n=n,
        index=index,
        proxy_model=_resolve_proxy(proxy_tag, seed),
        sampling_weights=weights,
        random_state=seed,
    ).approximate(budget, game)
    generic = ProxySHAP(
        n=n,
        max_order=1,
        index=index,
        proxy_model=_resolve_proxy(proxy_tag, seed),
        adjustment="msr",
        sampling_weights=weights,
        random_state=seed,
    ).approximate(budget, game)

    assert np.allclose(_phi(fast, n), _phi(generic, n), atol=1e-10)
    assert np.isclose(fast.baseline_value, generic.baseline_value, atol=1e-10)
    assert fast.estimated == generic.estimated
    assert fast.estimation_budget == generic.estimation_budget


def test_missing_singleton_treated_as_zero():
    """A singleton the proxy never reports must be treated as 0, not dropped or raised on.

    xgboost can drop a feature it never split on from the extracted interactions entirely at a
    tiny budget. This reproduces one such case (n=5, budget=7, seed=3, xgboost, on a game where
    feature 0 has zero effect: the proxy's extracted interactions there contain only keys ``()``,
    ``(1,)``, ``(2,)``, entirely missing ``(0,)``, ``(3,)``, ``(4,)``). The fast path must still
    return a finite value for every player, and must still match the generic path (which handles
    this via :meth:`InteractionValues.__add__`'s own missing-key-as-0 semantics).
    """
    n, budget, seed = 5, 7, 3
    game = _irrelevant_feature_game(n, seed, irrelevant=0)

    # confirm this configuration really does reproduce a missing singleton in the raw proxy
    # readout (guards the test itself against silently testing nothing if upstream changes).
    probe = RegressionMSR(n=n, index="SV", proxy_model="xgboost", random_state=seed)
    probe._sampler.sample(budget)
    coalitions_matrix = probe._sampler.coalitions_matrix
    coalition_values = game(coalitions_matrix)
    coalition_values = coalition_values - coalition_values[0]
    fitted = fit_proxy(probe.proxy_model, coalitions_matrix, coalition_values, max_order=1)
    proxy_interactions = _extract_proxy_interactions(
        fitted,
        baseline_value=0.0,
        max_order=1,
        approximation_index=probe.approximation_index,
        target_index="SV",
        budget=coalitions_matrix.shape[0],
        n_players=n,
    )
    missing = [i for i in range(n) if (i,) not in proxy_interactions.interactions]
    assert missing, "setup no longer reproduces a missing singleton; update repro parameters"

    fast = RegressionMSR(n=n, index="SV", proxy_model="xgboost", random_state=seed).approximate(
        budget, game
    )
    for i in range(n):
        assert np.isfinite(fast[(i,)])

    # matching sampling_weights: RegressionMSR's default (the paper kernel) now differs from
    # ProxySHAP's own unchanged bowl-shaped default (see test_fast_path_matches_parent_path).
    generic = ProxySHAP(
        n=n,
        max_order=1,
        index="SV",
        proxy_model="xgboost",
        adjustment="msr",
        sampling_weights=_paper_sampling_weights(n, "SV"),
        random_state=seed,
    ).approximate(budget, game)
    assert np.allclose(_phi(fast, n), _phi(generic, n), atol=1e-10)


@pytest.mark.parametrize("index", ["SV", "BV"])
def test_full_budget_exact(index):
    """At full budget every coalition is sampled, so the residual is the proxy's exact in-sample
    error against the fully observed game and the correction recovers the exact SV/BV.

    Uses ``proxy_model="linear"``: xgboost/tree proxies are only exact to ~1e-7 at full budget
    (their own internal floating-point precision -- unrelated to the MSR correction itself, which
    matches the generic path to numerical precision in ``test_fast_path_matches_parent_path``), so
    they cannot meet a 1e-10 tolerance against an independently-computed :class:`ExactComputer`
    ground truth. The linear route uses plain linear algebra and is exact to ~1e-15.
    """
    n = 6
    game = _pairwise_game(n, seed=1)
    exact = ExactComputer(game=game, n_players=n)(index=index, order=1)
    phi_exact = np.array([exact[(i,)] for i in range(n)])

    approx = RegressionMSR(n=n, index=index, proxy_model="linear", random_state=1)
    result = approx.approximate(budget=2**n, game=game)

    assert result.estimated is False
    assert result.estimation_budget == 2**n
    assert np.allclose(_phi(result, n), phi_exact, atol=1e-10)


def test_estimation_budget_matches_distinct_coalitions_sampled():
    """``estimation_budget`` must equal the number of distinct coalitions actually evaluated.

    Not necessarily the requested ``budget``: the sampler may realize fewer distinct coalitions
    (e.g. once it has covered a large fraction of the space). Read the ground truth directly off
    the approximator's own sampler after the call.
    """
    n = 7
    budget = 3 * n
    game = _pairwise_game(n, seed=0)
    approx = RegressionMSR(n=n, index="SV", proxy_model="xgboost", random_state=0)
    result = approx.approximate(budget=budget, game=game)

    n_distinct = approx._sampler.coalitions_matrix.shape[0]
    assert result.estimation_budget == n_distinct
    assert result.estimated == (n_distinct < 2**n)
    assert result.estimation_budget <= budget


def test_bv_linear_estimation_budget_not_lost():
    """Regression test for a pre-existing bug in the linear extraction route.

    ``index="BV"`` with ``proxy_model="linear"`` loses ``estimation_budget`` (always ``None``)
    and ``estimated`` (always ``True``, even at full budget) when read off the proxy's own
    extracted :class:`InteractionValues`; ``index="SV"`` with the same proxy, and ``index="BV"``
    with ``xgboost``/``tree``, are all unaffected. The fast path avoids depending on this by
    computing both fields itself from the coalitions it actually sampled, so this must hold
    regardless.
    """
    n = 6
    game = _pairwise_game(n, seed=0)
    approx = RegressionMSR(n=n, index="BV", proxy_model="linear", random_state=0)
    result = approx.approximate(budget=2**n, game=game)
    assert result.estimation_budget == 2**n
    assert result.estimated is False


def test_partial_budget_msr_beats_none_on_average():
    """At a partial budget, the msr-adjusted estimate should on average be more accurate than
    the unadjusted proxy-only estimate, over several seeds, by a statistically meaningful margin.

    Uses the tree-game family, checked against the paired-difference standard error rather than
    a bare inequality: a bare-``<`` version over few seeds can have only a fraction-of-a-percent
    relative margin, which is not distinguishable from noise and could flip on an unrelated
    dependency bump without indicating a real regression. Requiring the mean gap to exceed 2
    standard errors of the paired per-seed differences is a standard way to separate a real effect
    from sampling noise. ``RegressionMSR`` hardcodes ``adjustment="msr"``, so the no-correction
    reference uses ``ProxySHAP(adjustment="none")`` directly with the same
    ``max_order=1``/``index="SV"`` configuration.
    ``approx_none`` is given the same explicit ``sampling_weights`` as ``approx_msr`` (the paper
    kernel, ``RegressionMSR``'s new default) so both draw the same coalitions per seed -- isolating
    the effect under test (msr correction vs. none) from the unrelated question of which sampling
    scheme is used.
    """
    n = 6
    budget = 4 * n
    n_seeds = 30
    weights = _paper_sampling_weights(n, "SV")
    rse_msr, rse_none = [], []
    last_result = None
    for seed in range(n_seeds):
        game = _make_tree_game(n, seed)
        exact = ExactComputer(game=game, n_players=n)(index="SV", order=1)
        phi_exact = _phi(exact, n)

        approx_msr = RegressionMSR(
            n=n, index="SV", proxy_model="xgboost", sampling_weights=weights, random_state=seed
        )
        iv_msr = approx_msr.approximate(budget=budget, game=game)
        last_result = iv_msr
        rse_msr.append(_relative_squared_error(_phi(iv_msr, n), phi_exact))

        approx_none = ProxySHAP(
            n=n,
            max_order=1,
            index="SV",
            proxy_model="xgboost",
            adjustment="none",
            sampling_weights=weights,
            random_state=seed,
        )
        iv_none = approx_none.approximate(budget=budget, game=game)
        rse_none.append(_relative_squared_error(_phi(iv_none, n), phi_exact))

    assert last_result.estimated is True
    assert last_result.estimation_budget <= budget

    diffs = np.array(rse_none) - np.array(rse_msr)
    mean_diff = float(np.mean(diffs))
    se_diff = float(np.std(diffs, ddof=1) / np.sqrt(n_seeds))
    assert mean_diff > 2 * se_diff, (
        f"mean gap {mean_diff} is not a statistically meaningful margin (2*SE={2 * se_diff})"
    )


def test_diagnostics_set_and_in_range():
    """``train_residual_ratio`` and ``correction_norm`` are set on the approximator after
    :meth:`~RegressionMSR.approximate`, are finite, and are non-negative (both are norms/norm
    ratios; see the class docstring's :attr:`~RegressionMSR.train_residual_ratio` /
    :attr:`~RegressionMSR.correction_norm` for their definitions and reference-code citation).
    """
    n = 6
    game = _pairwise_game(n, seed=0)
    approx = RegressionMSR(n=n, index="SV", proxy_model="xgboost", random_state=0)
    approx.approximate(budget=4 * n, game=game)

    assert np.isfinite(approx.train_residual_ratio)
    assert approx.train_residual_ratio >= 0
    assert np.isfinite(approx.correction_norm)
    assert approx.correction_norm >= 0
    # a partial-budget xgboost proxy on a nonlinear game has a genuinely nonzero residual, so
    # the correction should be doing real work, not vanishing to (numerical) zero.
    assert approx.correction_norm > 0


def test_diagnostics_none_before_approximate_and_after_fallback():
    """``train_residual_ratio``/``correction_norm`` must exist (not raise ``AttributeError``) and
    be ``None`` before :meth:`~RegressionMSR.approximate` is ever called, and must stay ``None``
    when that call instead falls back to the generic ``ProxySHAP.approximate()`` path (any index
    outside ``{"SV", "BV"}``), since the fallback never computes a closed-form correction to set
    them from (see the class docstring's ``Attributes:`` section).
    """
    n = 6
    fresh = RegressionMSR(n=n, index="SV", proxy_model="xgboost", random_state=0)
    assert fresh.train_residual_ratio is None
    assert fresh.correction_norm is None

    game = _pairwise_game(n, seed=0)
    fallback = RegressionMSR(n=n, index="SII", proxy_model="xgboost", random_state=0)
    fallback.approximate(budget=4 * n, game=game)
    assert fallback.train_residual_ratio is None
    assert fallback.correction_norm is None


def test_docstring_example_reproduces():
    """Reproduce the class docstring's ``Example:`` block exactly."""
    game = DummyGame(n=5, interaction=(1, 2))
    approximator = RegressionMSR(n=5, index="SV")
    result = approximator.approximate(budget=100, game=game)

    assert result.index == "SV"
    assert result.max_order == 1
    assert result.min_order == 0
    assert result.estimated is False
    assert result.estimation_budget == 32
    assert result.n_players == 5
    assert result.baseline_value == 0.0
    assert repr(result) == (
        "InteractionValues(\n"
        "    index=SV, max_order=1, min_order=0, estimated=False, estimation_budget=32,\n"
        "    n_players=5, baseline_value=0.0\n"
        ")"
    )


def test_semivalue_p_raises_for_unsupported_index():
    """``_semivalue_p`` only has a closed-form weight for ``"SV"``/``"BV"``; any other index must
    raise, not silently fall back to (e.g.) the SV weights.

    ``RegressionMSR.approximate()`` itself never reaches this branch (it delegates to
    ``ProxySHAP.approximate()`` for any index outside ``{"SV", "BV"}`` before calling
    ``_semivalue_p`` at all -- see ``test_unsupported_index_falls_back_to_parent`` below), so this
    branch is exercised directly here to keep it covered without depending on that guard.
    """
    with pytest.raises(ValueError, match="No closed-form semivalue weight"):
        _semivalue_p(5, "SII")


def test_paper_sampling_weights_raises_for_unsupported_index():
    """``_paper_sampling_weights`` only has a closed-form kernel for ``"SV"``/``"BV"``; any other
    index must raise, not silently fall back to (e.g.) the SV kernel.

    ``RegressionMSR.__init__`` never reaches this branch itself (it only calls
    ``_paper_sampling_weights`` when ``index in ("SV", "BV")``, per its own guard), so this branch
    is exercised directly here to keep it covered without depending on that guard never changing.
    """
    with pytest.raises(ValueError, match="No closed-form paper sampling kernel"):
        _paper_sampling_weights(5, "SII")


@pytest.mark.parametrize("index", ["SII", "k-SII", "STII"])
def test_unsupported_index_falls_back_to_parent(index):
    """For any index the closed-form correction does not handle (``{"SV", "BV"}`` only),
    ``approximate()`` must delegate to the generic ``ProxySHAP.approximate()`` path and produce the
    identical result -- not crash.

    Construction with e.g. ``index="SII"`` succeeds (the inherited ``valid_indices`` is not
    narrowed), so ``approximate()`` must check ``self.index`` and delegate to the generic path
    *before* sampling or fitting anything -- not waste a full sampling/fitting pass and then raise
    a closed-form-specific ``ValueError`` from ``_semivalue_p``. ``ProxySHAP`` itself handles these
    three indices fine at ``max_order=1``/``adjustment="msr"``, so this is a meaningful, not merely
    hypothetical, fallback target. ``index="BII"`` is a separate case (see
    ``test_bii_index_delegates_and_matches_parent_failure``): the *parent* itself does not support
    it for ``adjustment="msr"``.
    """
    n, seed, budget = 6, 0, 4 * 6
    game = _pairwise_game(n, seed)

    fast = RegressionMSR(n=n, index=index, proxy_model="xgboost", random_state=seed).approximate(
        budget, game
    )
    generic = ProxySHAP(
        n=n,
        max_order=1,
        index=index,
        proxy_model="xgboost",
        adjustment="msr",
        random_state=seed,
    ).approximate(budget, game)

    assert np.allclose(_phi(fast, n), _phi(generic, n), atol=1e-10)
    assert fast.index == generic.index
    assert fast.estimated == generic.estimated
    assert fast.estimation_budget == generic.estimation_budget


def test_bii_index_delegates_and_matches_parent_failure():
    r"""``index="BII"`` is not supported by ``ProxySHAP`` itself for ``adjustment="msr"`` when the
    proxy is a *linear* model (a pre-existing, upstream limitation, unrelated to
    ``RegressionMSR``): the generic
    ``ProxySHAP(adjustment="msr", index="BII", proxy_model="linear").approximate()`` itself raises
    ``ValueError: Invalid index. Index \`BII\` is not supported...``, raised from
    ``MoebiusConverter.compute``.

    This is specific to the linear proxy's extraction route (``_extract_linear`` in ``_routes.py``
    unconditionally runs its interaction read-out through ``MoebiusConverter(...).compute()``); the
    tree route (``_extract_tree``) does not call ``MoebiusConverter`` at all, so
    ``proxy_model="xgboost"``/``"tree"`` does *not* reproduce this failure. Hence
    ``proxy_model="linear"`` here.

    Delegating the *entire* call to the parent (rather than special-casing the exception) means
    ``RegressionMSR`` now raises exactly the same error the parent does for this index, instead of
    the different, closed-form-specific ``_semivalue_p`` ``ValueError`` a naive implementation
    might raise -- keeping the failure mode consistent across every "unsupported" index.
    """
    n, seed, budget = 6, 0, 24
    game = _pairwise_game(n, seed)
    fast = RegressionMSR(n=n, index="BII", proxy_model="linear", random_state=seed)
    generic = ProxySHAP(
        n=n, max_order=1, index="BII", proxy_model="linear", adjustment="msr", random_state=seed
    )

    with pytest.raises(ValueError, match=r"Index `BII` is not supported"):
        fast.approximate(budget, game)
    with pytest.raises(ValueError, match=r"Index `BII` is not supported"):
        generic.approximate(budget, game)


def test_fallback_forwards_kwargs_to_parent_approximate(monkeypatch):
    """The unsupported-index fallback (``return super().approximate(budget, game, **kwargs)``)
    must actually forward ``budget``, ``game``, and any extra ``kwargs`` unchanged to
    ``ProxySHAP.approximate`` -- not silently drop them.

    Not otherwise observable: ``ProxySHAP.approximate`` itself ignores every kwarg it is called
    with today, so a test that only checks the final *output* cannot distinguish
    ``super().approximate(budget, game, **kwargs)`` from ``super().approximate(budget, game)`` --
    a mutant dropping ``**kwargs`` from the delegation call would otherwise survive undetected.
    Spy on ``ProxySHAP.approximate`` directly (monkeypatched on the class, so the ``RegressionMSR``
    instance still reaches it, bound, via ``super()``) to capture what it was actually called with.
    """
    captured = {}
    sentinel = object()

    def spy(self, budget, game, **kwargs):
        captured["self"] = self
        captured["budget"] = budget
        captured["game"] = game
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(ProxySHAP, "approximate", spy)

    n, seed, budget = 6, 0, 24
    game = _pairwise_game(n, seed)
    approx = RegressionMSR(n=n, index="SII", proxy_model="xgboost", random_state=seed)
    result = approx.approximate(budget, game, foo="bar", baz=42)

    assert captured["self"] is approx
    assert captured["budget"] == budget
    assert captured["game"] is game
    assert captured["kwargs"] == {"foo": "bar", "baz": 42}
    assert result is sentinel


@pytest.mark.parametrize("index", ["SV", "BV"])
def test_baseline_value_matches_v_of_empty_coalition(index):
    """``result.baseline_value`` and ``result[()]`` must equal the game's own ``v(empty)``, not be
    silently hardcoded to ``0.0``. Every other game in this file has ``v(empty) == 0`` by
    construction, which lets such a bug hide behind every existing test; ``_offset_game`` has
    ``v(empty) == 3.7``.

    Both assertions matter and are NOT redundant: ``InteractionValues.__init__`` reconciles a
    disagreement between the passed ``values[()]`` dict entry and the passed ``baseline_value=``
    kwarg by overwriting the dict entry from ``baseline_value=`` -- but only for indices where
    ``is_empty_value_the_baseline(index)`` is ``True``, which holds for ``"SV"`` but not for
    ``"BV"``. A bug that corrupts only the local ``interactions[()]`` dict entry (while
    ``baseline_value=`` itself stays correct) would be silently self-healed and invisible via
    ``result[()]`` for SV; parametrizing over both of ``RegressionMSR``'s two supported indices is
    what makes this test meaningful.
    """
    n = 6
    game = _offset_game(n, offset=3.7)
    approx = RegressionMSR(n=n, index=index, proxy_model="xgboost", random_state=0)
    result = approx.approximate(budget=4 * n, game=game)

    assert result.baseline_value == pytest.approx(3.7, abs=1e-9)
    assert result[()] == pytest.approx(3.7, abs=1e-9)


def test_proxy_interactions_object_not_mutated_in_place(monkeypatch):
    """``approximate()`` must assemble a FRESH ``InteractionValues``, never mutating the proxy's
    own extracted result in place (explicit requirement in the class docstring).

    Not observable from the final output alone (``proxy_interactions`` is a purely local variable
    in the implementation), so capture the object at the seam -- monkeypatching the module-level
    ``_extract_proxy_interactions`` that ``regressionmsr.py`` calls -- and check it is
    byte-for-byte unchanged after ``approximate()`` returns.
    """
    captured = {}
    orig = regressionmsr_mod._extract_proxy_interactions

    def spy(*args, **kwargs):
        result = orig(*args, **kwargs)
        captured["proxy_interactions"] = result
        captured["snapshot"] = dict(result.interactions)
        return result

    monkeypatch.setattr(regressionmsr_mod, "_extract_proxy_interactions", spy)

    n = 6
    game = _pairwise_game(n, seed=0)
    RegressionMSR(n=n, index="SV", proxy_model="xgboost", random_state=0).approximate(
        budget=4 * n, game=game
    )

    assert captured, "the spy was never invoked; approximate() no longer calls this seam"
    assert captured["proxy_interactions"].interactions == captured["snapshot"]


def test_train_residual_ratio_matches_hand_computed_reference_formula():
    """``train_residual_ratio`` must match the reference's raw (baseline-inclusive) formula
    exactly: ``||v - f_hat||_2 / ||v||_2``, using ``v(S)``/``f_hat(S)`` *before* any baseline
    shift -- not shapiq's internal ``||v - v(empty)||_2`` convention, which diverges from the
    reference by up to ~6x whenever ``v(empty) != 0`` (see
    :attr:`RegressionMSR.train_residual_ratio`'s docstring).

    Uses a game with ``v(empty) != 0`` so the two formulas are actually distinguishable, and
    recomputes the expected ratio *by hand* from the reference's own definition (not by calling
    any of the approximator's correction/diagnostics code): the sampler's own realized coalitions
    and the game's raw values, refit via the same public ``fit_proxy``/``predict_proxy`` routes the
    implementation itself uses (not a reimplementation of the correction formula, so this isn't a
    self-comparison tautology).
    """
    n, seed = 6, 0
    game = _offset_game(n, offset=3.7)

    approx = RegressionMSR(n=n, index="SV", proxy_model="xgboost", random_state=seed)
    approx.approximate(budget=4 * n, game=game)

    z = approx._sampler.coalitions_matrix
    v_raw = game(z)  # raw, NOT baseline-shifted
    baseline = v_raw[0]
    v_shifted = v_raw - baseline
    fitted = fit_proxy(approx.proxy_model, z, v_shifted, max_order=1)
    f_hat_shifted = predict_proxy(fitted, z, max_order=1)
    f_hat_raw = f_hat_shifted + baseline
    residual = v_raw - f_hat_raw
    expected_ratio = np.linalg.norm(residual) / np.linalg.norm(v_raw)

    assert approx.train_residual_ratio == pytest.approx(expected_ratio, rel=1e-9)


def test_correction_norm_is_l2_not_l1():
    """``correction_norm`` must be the Euclidean (L2) norm of the correction vector, not the L1
    norm (the previous ``test_diagnostics_set_and_in_range``'s ``> 0`` check cannot distinguish a
    norm type).

    Recovers the per-player correction behaviorally as ``(msr singleton) - (proxy-only singleton,
    same seed/coalitions, adjustment="none")``, then compares its L2 norm to ``correction_norm``.

    ``approx_none`` is given the same explicit ``sampling_weights`` as ``approx`` (``approx``'s
    default is now the paper kernel, which ``ProxySHAP``'s own default does not use) so both draw
    the identical coalitions -- needed for "same seed/coalitions" above to actually hold.
    """
    n, seed = 6, 0
    game = _pairwise_game(n, seed)
    weights = _paper_sampling_weights(n, "SV")
    approx = RegressionMSR(
        n=n, index="SV", proxy_model="xgboost", sampling_weights=weights, random_state=seed
    )
    result = approx.approximate(budget=4 * n, game=game)

    approx_none = ProxySHAP(
        n=n,
        max_order=1,
        index="SV",
        proxy_model="xgboost",
        adjustment="none",
        sampling_weights=weights,
        random_state=seed,
    )
    result_none = approx_none.approximate(budget=4 * n, game=game)
    correction = _phi(result, n) - _phi(result_none, n)
    expected_l2 = float(np.linalg.norm(correction))

    assert approx.correction_norm == pytest.approx(expected_l2, rel=1e-6)


def _reference_paper_sampling_weights(n: int, index: str) -> np.ndarray:
    """Independent (non-log-space) reference for the paper sampling kernel.

    Recomputes the reference sampling kernel's ``sample_dist`` (Witter et al., 2025)
    (``D(s) = sqrt(p_{s-1}^2 * s * [s>0] + p_s^2 * (n-s) * [s<n])``) and the per-size mass
    conversion (``mass[s] = D(s) * C(n, s)``, renormalized) directly with plain
    ``float``/``math.factorial``/``math.comb`` arithmetic -- deliberately *not* sharing
    :func:`~shapiq.approximator.proxy.regressionmsr._paper_sampling_weights`'s log-space
    implementation, so this is an independent check of that function's output, not a
    self-comparison tautology. Only intended for the moderate ``n`` values this test file uses
    (up to 60): unlike the log-space implementation under test, this does not defend against
    underflow at large ``n`` (e.g. Banzhaf's ``p_k = 1 / 2 ** (n - 1)`` squared underflows past
    ``n > ~538``).
    """
    if index == "SV":
        p = np.array(
            [math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) for k in range(n)]
        )
    elif index == "BV":
        p = np.full(n, 1.0 / (2 ** (n - 1)))
    else:
        msg = f"unsupported index {index!r}"
        raise ValueError(msg)

    density = np.empty(n + 1)
    for s in range(n + 1):
        term = 0.0
        if s > 0:
            term += p[s - 1] ** 2 * s
        if s < n:
            term += p[s] ** 2 * (n - s)
        density[s] = math.sqrt(term)

    comb = np.array([math.comb(n, s) for s in range(n + 1)], dtype=float)
    mass = density * comb
    return mass / mass.sum()


@pytest.mark.parametrize("index", ["SV", "BV"])
@pytest.mark.parametrize("n", [3, 5, 10, 60])
def test_default_sampling_weights_match_independent_reference(n, index):
    """:func:`_paper_sampling_weights` (and hence ``RegressionMSR``'s default ``sampling_weights``
    for ``index in {"SV", "BV"}``) matches :func:`_reference_paper_sampling_weights`, an
    independently written, non-log-space reference implementation of the same formula, and is a
    valid probability distribution over coalition sizes ``0, ..., n`` (non-negative, sums to 1).
    """
    weights = _paper_sampling_weights(n, index)
    expected = _reference_paper_sampling_weights(n, index)

    assert weights.shape == (n + 1,)
    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(weights, expected, rtol=1e-9, atol=1e-12)

    # And the constructed approximator actually uses this as its default (no sampling_weights=
    # passed), for both the fast RegressionMSR path and its ProxySHAP parent.
    fast = RegressionMSR(n=n, index=index, random_state=0)
    np.testing.assert_allclose(fast._sampler._sampling_weights, weights, rtol=1e-9)


def _reference_leverage_shap_density_sv_linear(n: int) -> np.ndarray:
    """Independent (non-log-space) reference for the SV kernel used when ``proxy_model`` is linear.

    Recomputes :cite:t:`Witter.2025`'s reference override for ``reg_model_class == 'linear'`` with
    Shapley weighting (the reference's ``UniversalMSR.__init__``): ``D(s) = (p_s +
    p_{s-1}) * s * (n - s)`` for ``s = 1, ..., n - 1``, and ``D(0) = D(n) = 0`` -- deliberately
    *not* sharing :func:`~shapiq.approximator.proxy.regressionmsr._paper_sampling_weights`'s
    log-space implementation (nor :func:`~shapiq.approximator.proxy.regressionmsr._semivalue_p`),
    so this is an independent check of that function's linear-kernel branch, not a self-comparison
    tautology.
    """
    p = np.array(
        [math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) for k in range(n)]
    )
    density = np.zeros(n + 1)
    for s in range(1, n):
        density[s] = (p[s] + p[s - 1]) * s * (n - s)
    comb = np.array([math.comb(n, s) for s in range(n + 1)], dtype=float)
    mass = density * comb
    return mass / mass.sum()


@pytest.mark.parametrize("n", [3, 5, 10, 60])
def test_sv_linear_default_matches_leverage_shap_reference_and_differs_from_generic(n):
    """For ``index="SV"`` with a linear ``proxy_model``, :func:`_paper_sampling_weights` must
    switch to the reference's Leverage-SHAP kernel (the reference's ``reg_model_class == 'linear'``
    override for Shapley weighting), not silently keep using the generic (tree/xgboost) kernel --
    the M1 finding this fix addresses. Checked against
    :func:`_reference_leverage_shap_density_sv_linear`, an independently written reference of that
    specific kernel (not the generic one already covered by
    ``test_default_sampling_weights_match_independent_reference``), and confirmed to actually
    differ from the generic kernel so this branch is observable rather than accidentally reducing
    to the same numbers.
    """
    linear_weights = _paper_sampling_weights(n, "SV", "linear")
    expected = _reference_leverage_shap_density_sv_linear(n)

    assert linear_weights.shape == (n + 1,)
    assert np.all(linear_weights >= 0)
    assert linear_weights.sum() == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(linear_weights, expected, rtol=1e-9, atol=1e-12)

    generic_weights = _paper_sampling_weights(n, "SV", "xgboost")
    assert not np.allclose(linear_weights, generic_weights)

    # An actual (unresolved) linear estimator instance -- not just the "linear" string tag --
    # must select the same kernel: RegressionMSR.__init__ passes proxy_model through unresolved,
    # see _proxy_selects_linear_kernel.
    instance_weights = _paper_sampling_weights(n, "SV", LinearRegression())
    np.testing.assert_allclose(instance_weights, linear_weights, rtol=1e-12)

    # And the constructed approximator actually uses the linear kernel as its default (no
    # sampling_weights= passed) when proxy_model="linear".
    fast = RegressionMSR(n=n, index="SV", proxy_model="linear", random_state=0)
    np.testing.assert_allclose(fast._sampler._sampling_weights, linear_weights, rtol=1e-9)


@pytest.mark.parametrize("n", [3, 5, 10, 60])
def test_bv_linear_default_equals_generic_and_uniform(n):
    """For ``index="BV"``, the reference's ``proxy_model="linear"`` override (a constant density,
    the reference's ``ones_like``) is mathematically identical -- not just proportional -- to the
    generic (non-linear) kernel, since Banzhaf's semivalue coefficient ``p_k`` is constant in
    ``k``. :func:`_paper_sampling_weights` therefore does not special-case ``index="BV"`` at all;
    this test verifies that identity directly rather than assuming it (per the M1 fix's
    instructions), and additionally checks both against a third, independent characterization: a
    constant per-coalition density's per-size mass is exactly the ``Binomial(n, 1/2)`` pmf
    (``C(n, s) / 2**n``, normalized), since ``mass[s] = D(s) * C(n, s)`` with constant ``D(s)`` is
    proportional to ``C(n, s)`` alone.
    """
    linear_weights = _paper_sampling_weights(n, "BV", "linear")
    generic_weights = _paper_sampling_weights(n, "BV", "xgboost")
    uniform_mass = np.array([math.comb(n, s) for s in range(n + 1)], dtype=float)
    uniform_mass = uniform_mass / uniform_mass.sum()

    np.testing.assert_allclose(linear_weights, generic_weights, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(linear_weights, uniform_mass, rtol=1e-9, atol=1e-12)


def test_non_linear_proxies_use_generic_kernel():
    """``_paper_sampling_weights``'s default ``proxy_model="xgboost"`` (matching
    ``RegressionMSR``'s own default) and other non-linear proxies all select the generic kernel,
    unaffected by the linear-kernel branch added for M1 -- i.e. the pre-existing behavior
    ``test_default_sampling_weights_match_independent_reference`` already covers, exercised here
    explicitly across a few non-linear proxy representations (string tags and a raw estimator
    instance) to guard against the linear check accidentally over-triggering.
    """
    n = 10
    baseline = _paper_sampling_weights(n, "SV")  # proxy_model defaults to "xgboost"
    for proxy_model in ("xgboost", "tree", "lightgbm", DecisionTreeRegressor()):
        np.testing.assert_allclose(
            _paper_sampling_weights(n, "SV", proxy_model), baseline, rtol=1e-12
        )


@pytest.mark.parametrize("n", [3, 5, 10, 60])
def test_default_sampling_weights_finite_at_large_n(n):
    """The log-space implementation stays finite and normalized well past the ``n`` at which a
    naive (non-log-space) computation of the semivalue coefficients squared would underflow to
    exactly ``0.0`` (``n > ~538`` for Banzhaf, per :func:`_paper_sampling_weights`'s docstring).
    Also exercised directly at ``n=500`` and ``n=1000`` below, which
    :func:`_reference_paper_sampling_weights` cannot handle at all.
    """
    for index in ("SV", "BV"):
        weights = _paper_sampling_weights(n, index)
        assert np.all(np.isfinite(weights))
        assert np.all(weights >= 0)
        assert weights.sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("n", [500, 1000])
@pytest.mark.parametrize("index", ["SV", "BV"])
def test_default_sampling_weights_finite_for_large_n(n, index):
    """At ``n=500``/``n=1000``, a naive (non-log-space) computation of ``p_k ** 2`` underflows to
    exactly ``0.0`` for essentially every ``k`` (see :func:`_paper_sampling_weights`'s docstring),
    which would zero out the whole distribution and divide-by-zero on renormalization. The actual
    (log-space) implementation must still produce a finite, non-negative, normalized distribution.
    """
    weights = _paper_sampling_weights(n, index)
    assert weights.shape == (n + 1,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_explicit_sampling_weights_override_default():
    """An explicit ``sampling_weights=`` argument must still be used as-is, not silently replaced
    by the new paper-kernel default -- the default only applies when ``sampling_weights is None``
    (see ``RegressionMSR.__init__``'s ``if sampling_weights is None and index in ("SV", "BV")``
    guard).
    """
    n = 7
    rng = np.random.default_rng(0)
    custom = rng.uniform(0.1, 1.0, size=n + 1)
    custom = custom / custom.sum()

    default_weights = _paper_sampling_weights(n, "SV")
    # sanity: the custom weights are not (accidentally) equal to the default, so this test would
    # actually fail if the override were silently ignored.
    assert not np.allclose(custom, default_weights)

    approx = RegressionMSR(n=n, index="SV", sampling_weights=custom, random_state=0)
    np.testing.assert_allclose(approx._sampler._sampling_weights, custom, rtol=1e-12)


def test_explicit_sampling_weights_override_unnormalized():
    """An explicit ``sampling_weights=`` that isn't already normalized is accepted and normalized
    by :class:`~shapiq.approximator.sampling.CoalitionSampler` itself (not replaced by the
    default), exactly like it was before this default changed.
    """
    n = 4
    unnormalized = np.array([1.0, 2.0, 3.0, 2.0, 1.0])  # sums to 9, not 1
    approx = RegressionMSR(n=n, index="SV", sampling_weights=unnormalized, random_state=0)
    np.testing.assert_allclose(
        approx._sampler._sampling_weights, unnormalized / unnormalized.sum(), rtol=1e-12
    )


def test_default_sampling_weights_not_worse_than_old_default_on_soum():
    """The new paper-kernel default should, on average, be at least as accurate as the old
    bowl-shaped default it replaces -- not just on the tree-game family
    ``test_partial_budget_msr_beats_none_on_average`` already covers, but on an unrelated
    synthetic game family (:class:`~shapiq_games.synthetic.SOUM`, a sum of unanimity games).

    The "old default" is recovered behaviorally: constructing a ``ProxySHAP`` with
    ``sampling_weights=None`` still gets the old, unchanged bowl-shaped default (only
    ``RegressionMSR``'s default changed), so that weight array is read off and passed explicitly
    to a second ``RegressionMSR`` instance to reproduce pre-change sampling behavior exactly.

    Checked via the same paired-difference standard-error margin as
    ``test_partial_budget_msr_beats_none_on_average`` (see that test's docstring for why a bare
    inequality is not used), but as a "not worse" bound (``mean_diff > -2 * SE``) rather than a
    "must beat" bound: the paper kernel is known to beat the old default by ~25% pooled across a
    wider benchmark grid, but this is only 30 seeds on one game family/size/budget, so demanding
    statistically-significant *improvement* here risks flaking on an off sample even though the
    true effect is positive.
    """
    n = 12
    budget = 8 * n
    n_seeds = 30

    old_default_weights = ProxySHAP(
        n=n, max_order=1, index="SV", proxy_model="xgboost", random_state=0
    )._sampler._sampling_weights.copy()
    new_default_weights = _paper_sampling_weights(n, "SV")
    # sanity: these are genuinely different sampling schemes, so the comparison below is
    # meaningful (not accidentally comparing identical sampling behavior against itself).
    assert not np.allclose(old_default_weights, new_default_weights)

    rse_new, rse_old = [], []
    for seed in range(n_seeds):
        game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=seed)
        exact = ExactComputer(game=game, n_players=n)(index="SV", order=1)
        phi_exact = _phi(exact, n)

        approx_new = RegressionMSR(n=n, index="SV", proxy_model="xgboost", random_state=seed)
        iv_new = approx_new.approximate(budget=budget, game=game)
        rse_new.append(_relative_squared_error(_phi(iv_new, n), phi_exact))

        approx_old = RegressionMSR(
            n=n,
            index="SV",
            proxy_model="xgboost",
            sampling_weights=old_default_weights,
            random_state=seed,
        )
        iv_old = approx_old.approximate(budget=budget, game=game)
        rse_old.append(_relative_squared_error(_phi(iv_old, n), phi_exact))

    diffs = np.array(rse_old) - np.array(rse_new)  # positive => new default is more accurate
    mean_diff = float(np.mean(diffs))
    se_diff = float(np.std(diffs, ddof=1) / np.sqrt(n_seeds))
    assert mean_diff > -2 * se_diff, (
        f"new default is statistically significantly worse than the old default: "
        f"mean gap {mean_diff} (2*SE={2 * se_diff})"
    )
