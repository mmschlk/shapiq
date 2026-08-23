"""Tests for the :class:`~shapiq.approximator.proxy.regressionmsr.RegressionMSR` approximator.

Focused on the closed-form MSR fast path (:meth:`RegressionMSR.approximate`), which replaces the
generic :class:`~shapiq.approximator.proxy.proxyshap.ProxySHAP` base class's re-sampled Monte
Carlo residual correction with a direct Horvitz-Thompson correction computed from the coalitions
already sampled to fit the proxy. See ``run_logs/pr_b/BRIEF.md`` and
``run_logs/pr_b/IMPL_REPORT.md`` for the design rationale and empirical verification this test
file's tolerances/scenarios are based on.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from shapiq.approximator import RegressionMSR
from shapiq.approximator.proxy._routes import _extract_proxy_interactions, fit_proxy
from shapiq.approximator.proxy.proxyshap import ProxySHAP
from shapiq.game_theory.exact import ExactComputer
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import DummyGame

_CATBOOST_AVAILABLE = importlib.util.find_spec("catboost") is not None
if _CATBOOST_AVAILABLE:
    from catboost import CatBoostRegressor


def _budgets_for(n: int) -> list[int]:
    """``{n+2, an odd budget, 2n, 8n, 2**n}``, deduplicated and capped at ``2**n``."""
    raw = [n + 2, 2 * n + 1, 2 * n, 8 * n, 2**n]
    return sorted({b for b in raw if b <= 2**n})


def _pairwise_game(n: int, seed: int):
    """A game with main effects, a few pairwise interactions, and a smooth nonlinearity.

    Adapted from ``run_logs/msr_verify/phase3_run_variant.py``'s grid game (the same recipe used
    to verify the fast path against the generic ``ProxySHAP`` path during the MSR audit).
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
    as ``v(S) = model.predict(S)``. Adapted from ``run_logs/msr_verify/phase2b_accuracy.py``'s
    ``make_tree_game`` (itself generalized from shapiq's own
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
    """
    seed = 0
    game = _pairwise_game(n, seed)

    fast = RegressionMSR(
        n=n, index=index, proxy_model=_resolve_proxy(proxy_tag, seed), random_state=seed
    ).approximate(budget, game)
    generic = ProxySHAP(
        n=n,
        max_order=1,
        index=index,
        proxy_model=_resolve_proxy(proxy_tag, seed),
        adjustment="msr",
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
    feature 0 has zero effect; confirmed via ``run_logs/pr_b/check_missing_singleton.py``, job
    284300 -- the proxy's extracted interactions there contain only keys ``()``, ``(1,)``,
    ``(2,)``, entirely missing ``(0,)``, ``(3,)``, ``(4,)``). The fast path must still return a
    finite value for every player, and must still match the generic path (which handles this via
    :meth:`InteractionValues.__add__`'s own missing-key-as-0 semantics).
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

    generic = ProxySHAP(
        n=n, max_order=1, index="SV", proxy_model="xgboost", adjustment="msr", random_state=seed
    ).approximate(budget, game)
    assert np.allclose(_phi(fast, n), _phi(generic, n), atol=1e-10)


@pytest.mark.parametrize("index", ["SV", "BV"])
def test_full_budget_exact(index):
    """At full budget every coalition is sampled, so the residual is the proxy's exact in-sample
    error against the fully observed game and the correction recovers the exact SV/BV.

    Uses ``proxy_model="linear"``: xgboost/tree proxies are only exact to ~1e-7 at full budget
    (their own internal floating-point precision, confirmed via
    ``run_logs/pr_b/check_full_budget_exact.py``, job 284310, max abs diff 4.976e-07 -- unrelated
    to the MSR correction itself, which matches the generic path to ~1e-19 in
    ``test_fast_path_matches_parent_path``), so they cannot meet a 1e-10 tolerance against an
    independently-computed :class:`ExactComputer` ground truth. The linear route uses plain
    linear algebra and is exact to ~1e-15 (job 284311).
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

    n_distinct = approx._sampler.coalitions_matrix.shape[0]  # noqa: SLF001
    assert result.estimation_budget == n_distinct
    assert result.estimated == (n_distinct < 2**n)
    assert result.estimation_budget <= budget


def test_bv_linear_estimation_budget_not_lost():
    """Regression test for a pre-existing bug in the linear extraction route.

    ``index="BV"`` with ``proxy_model="linear"`` loses ``estimation_budget`` (always ``None``)
    and ``estimated`` (always ``True``, even at full budget) when read off the proxy's own
    extracted :class:`InteractionValues` -- confirmed via
    ``run_logs/pr_b/check_linear_metadata.py``, job 284305: this combination is the only one of
    the 18 (index, proxy) x budget combinations tested there with this behavior; ``index="SV"``
    with the same proxy, and ``index="BV"`` with ``xgboost``/``tree``, are all unaffected. The
    fast path avoids depending on this by computing both fields itself from the coalitions it
    actually sampled (see ``regressionmsr.py``'s step 4), so this must hold regardless.
    """
    n = 6
    game = _pairwise_game(n, seed=0)
    approx = RegressionMSR(n=n, index="BV", proxy_model="linear", random_state=0)
    result = approx.approximate(budget=2**n, game=game)
    assert result.estimation_budget == 2**n
    assert result.estimated is False


def test_partial_budget_msr_beats_none_on_average():
    """At a partial budget, the msr-adjusted estimate should on average be more accurate than
    the unadjusted proxy-only estimate, over several seeds.

    Uses the tree-game family and configuration verified in
    ``run_logs/pr_b/check_partial_budget.py`` (job 284308: mean RSE msr=0.08786 < none=0.08796
    over these exact 10 seeds), matching the "none" baseline convention established by
    ``run_logs/msr_verify/phase2b_accuracy.py`` (``RegressionMSR`` hardcodes
    ``adjustment="msr"``, so the no-correction reference uses ``ProxySHAP(adjustment="none")``
    directly with the same ``max_order=1``/``index="SV"`` configuration).
    """
    n = 6
    budget = 4 * n
    rse_msr, rse_none = [], []
    last_result = None
    for seed in range(10):
        game = _make_tree_game(n, seed)
        exact = ExactComputer(game=game, n_players=n)(index="SV", order=1)
        phi_exact = _phi(exact, n)

        approx_msr = RegressionMSR(n=n, index="SV", proxy_model="xgboost", random_state=seed)
        iv_msr = approx_msr.approximate(budget=budget, game=game)
        last_result = iv_msr
        rse_msr.append(_relative_squared_error(_phi(iv_msr, n), phi_exact))

        approx_none = ProxySHAP(
            n=n, max_order=1, index="SV", proxy_model="xgboost", adjustment="none",
            random_state=seed,
        )
        iv_none = approx_none.approximate(budget=budget, game=game)
        rse_none.append(_relative_squared_error(_phi(iv_none, n), phi_exact))

    assert last_result.estimated is True
    assert last_result.estimation_budget <= budget
    assert np.mean(rse_msr) < np.mean(rse_none)


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


def test_docstring_example_reproduces():
    """Reproduce the class docstring's ``Example:`` block exactly (values verified via
    ``run_logs/pr_b/check_docstring.py``, job 284299).
    """
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
