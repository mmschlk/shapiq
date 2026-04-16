"""Cross-check tests: verify shapiq implementations by making independent
ground-truth sources agree on the same game.

Ground-truth sources used here:

1. ``ExactComputer(game)(index, order)`` — brute-force over 2^n coalitions.
2. ``SOUM.exact_values(index, order)`` — closed-form via Moebius conversion.
3. ``SOUM.moebius_coefficients`` — the SOUM's own analytical Moebius transform.
4. Consistent approximators at ``budget = 2**n`` — regression family + SHAPIQ /
   SVARMIQ on indices where they are known to be exact.
5. ``TreeSHAPIQXAI.exact_values`` — TreeExplainer on a tree model, versus
   ``ExactComputer`` running on the same tree game's ``value_function``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from shapiq.approximator import (
    SHAPIQ,
    SVARM,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    KernelSHAP,
    KernelSHAPIQ,
    OwenSamplingSV,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
    RegressionFBII,
    RegressionFSII,
    StratifiedSamplingSV,
    UnbiasedKernelSHAP,
)
from shapiq.game_theory.exact import ExactComputer
from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq_games.synthetic import SOUM

from .conftest import GROUND_TRUTH_INDICES, assert_iv_close

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# Approximators that are exact at budget = 2**n on an arbitrary game.
# (cls, max_order, index) — verified empirically against SOUM(n=5) at 2**5.
CONSISTENT_APPROXIMATORS: list[tuple[type, int, str]] = [
    (KernelSHAP, 1, "SV"),
    (KernelSHAPIQ, 2, "k-SII"),
    (InconsistentKernelSHAPIQ, 2, "k-SII"),
    (UnbiasedKernelSHAP, 1, "SV"),
    (RegressionFSII, 2, "FSII"),
    (RegressionFBII, 2, "FBII"),
    (SHAPIQ, 2, "SII"),
    (SHAPIQ, 2, "k-SII"),
    (SHAPIQ, 2, "STII"),
    (SVARMIQ, 2, "k-SII"),
    (SVARM, 1, "SV"),
]

# Sampling approximators — they converge only in the limit; we assert
# *error-decreases* as the budget grows, not exactness at any one budget.
SAMPLING_APPROXIMATORS: list[tuple[type, int, str]] = [
    (PermutationSamplingSV, 1, "SV"),
    (PermutationSamplingSII, 2, "k-SII"),
    (PermutationSamplingSTII, 2, "STII"),
    (OwenSamplingSV, 1, "SV"),
    (StratifiedSamplingSV, 1, "SV"),
]


def _approx_id(param: tuple[type, int, str]) -> str:
    cls, _, index = param
    return f"{cls.__name__}-{index}"


def _order_for(index: str) -> int:
    return 1 if index in {"SV", "BV"} else 2


def _max_abs_error(actual, expected) -> float:
    """Max absolute error across all interactions in ``expected``."""
    return max(
        abs(float(actual[interaction]) - float(expected.values[i]))
        for interaction, i in expected.interaction_lookup.items()
    )


# ===================================================================
# 1. ExactComputer vs SOUM.exact_values — both are analytical
# ===================================================================


@pytest.mark.parametrize("index", GROUND_TRUTH_INDICES)
class TestExactVsSOUM:
    """``ExactComputer(SOUM)`` must match ``SOUM.exact_values`` exactly."""

    def test_small_soum(self, soum_5, index):
        order = _order_for(index)
        expected = soum_5.exact_values(index, order)
        actual = ExactComputer(soum_5)(index, order=order)
        assert_iv_close(actual, expected, atol=1e-10)

    @pytest.mark.slow
    def test_medium_soum(self, soum_7, index):
        order = _order_for(index)
        expected = soum_7.exact_values(index, order)
        actual = ExactComputer(soum_7)(index, order=order)
        # n=7 FSII/FBII run through a larger LS solve; allow some numerical noise.
        assert_iv_close(actual, expected, atol=1e-8)


# ===================================================================
# 2. MoebiusConverter round-trip: ExactComputer -> Moebius -> index
# ===================================================================


@pytest.mark.parametrize("index", GROUND_TRUTH_INDICES)
class TestMoebiusConverter:
    """``MoebiusConverter`` on a Moebius IV produces the same values as the
    direct ExactComputer call for the target index.
    """

    def test_roundtrip_on_soum(self, soum_5, index):
        order = _order_for(index)
        # 1. Moebius values computed by ExactComputer on the game.
        moebius = ExactComputer(soum_5)("Moebius", order=soum_5.n_players)
        # 2. Convert Moebius -> target index via MoebiusConverter.
        converter = MoebiusConverter(moebius)
        converted = converter(index, order=order)
        # 3. Target values computed directly by ExactComputer.
        direct = ExactComputer(soum_5)(index, order=order)
        assert_iv_close(converted, direct, atol=1e-10)


# ===================================================================
# 3. Consistent approximators at budget = 2**n match SOUM ground truth
# ===================================================================


@pytest.mark.parametrize("config", CONSISTENT_APPROXIMATORS, ids=_approx_id)
class TestApproximatorAtFullBudget:
    """Regression-family and exhaustive Monte Carlo methods must agree with
    the analytical SOUM ground truth when given a full budget of 2^n.
    """

    def test_matches_ground_truth(self, soum_5, config):
        cls, max_order, index = config
        n = soum_5.n_players
        approx = cls(n=n, max_order=max_order, index=index, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = approx.approximate(2**n, soum_5)
        expected = soum_5.exact_values(index, max_order)
        assert_iv_close(result, expected, atol=1e-6)


# ===================================================================
# 4. Sampling approximators converge as budget grows (slow)
# ===================================================================


@pytest.mark.slow
@pytest.mark.parametrize("config", SAMPLING_APPROXIMATORS, ids=_approx_id)
class TestApproximatorConvergence:
    """Sampling-based approximators are not exact at any finite budget, but
    the error must decrease monotonically as the budget grows.
    """

    def test_error_decreases_with_budget(self, soum_7, config):
        cls, max_order, index = config
        n = soum_7.n_players
        expected = soum_7.exact_values(index, max_order)

        budgets = [2**n, 4 * 2**n, 16 * 2**n]
        errors = []
        for budget in budgets:
            approx = cls(n=n, max_order=max_order, index=index, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = approx.approximate(budget, soum_7)
            errors.append(_max_abs_error(result, expected))

        # Final error should be smaller than the initial error — sampling
        # methods must make progress with more budget.
        assert errors[-1] < errors[0], (
            f"{cls.__name__} ({index}): error did not decrease across "
            f"budgets {budgets}: errors={errors}"
        )


# ===================================================================
# 5. TreeExplainer vs ExactComputer on the tree's own game
# ===================================================================


@pytest.fixture(scope="module")
def _small_tree_setup():
    """Tiny decision tree on 5 features — small enough for ExactComputer."""
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 5))
    y = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + rng.normal(0, 0.05, size=40)
    model = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)
    x_explain = X[0]
    return model, x_explain


class TestTreeExplainerVsExactComputer:
    """Running ExactComputer on ``TreeSHAPIQXAI.value_function`` must agree
    with the TreeExplainer output (which is ``TreeSHAPIQXAI.exact_values``).
    """

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("index", ["SV", "k-SII"])
    def test_matches(self, _small_tree_setup, index):
        from shapiq_games.benchmark.treeshapiq_xai import TreeSHAPIQXAI

        model, x = _small_tree_setup
        game = TreeSHAPIQXAI(x=x, tree_model=model, normalize=True, verbose=False)
        order = _order_for(index)

        via_exact_computer = ExactComputer(game)(index, order=order)
        via_tree_explainer = game.exact_values(index=index, order=order)

        assert_iv_close(via_exact_computer, via_tree_explainer, atol=1e-6)
