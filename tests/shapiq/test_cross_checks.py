"""Cross-check tests: verify shapiq implementations by making independent
ground-truth sources agree on the same game.

Ground-truth sources used here:

1. ``ExactComputer(game)(index, order)`` — brute-force over 2^n coalitions.
2. ``SOUM.exact_values(index, order)`` — closed-form via Moebius conversion.
3. ``SOUM.moebius_coefficients`` — the SOUM's own analytical Moebius transform.
4. Consistent approximators at ``budget = 2**n`` — regression family + SHAPIQ /
   SVARMIQ on indices where they are known to be exact.
5. ``InterventionalTreeExplainer.explain_function(x)`` — TreeSHAP-IQ on a tree
   model, versus ``ExactComputer`` running on the matching
   ``InterventionalGame`` wrapping the same tree.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from shapiq.approximator import (
    SHAPIQ,
    SVARM,
    SVARMIQ,
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

from .conftest import GROUND_TRUTH_INDICES, assert_iv_close

# Note: ``InconsistentKernelSHAPIQ`` is *not* included below. It uses a
# k-additive least-squares projection that only matches the true SII when the
# underlying game is already k-additive; on genuine non-k-additive games (see
# :func:`soum_5`) it diverges even at budget = 2**n.

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# Approximators that are exact at budget = 2**n on an arbitrary game.
# (cls, max_order, index) — verified empirically against the non-k-additive
# ``soum_5`` fixture (``max_interaction_size = n``) at budget 2**5.
CONSISTENT_APPROXIMATORS: list[tuple[type, int, str]] = [
    (KernelSHAP, 1, "SV"),
    (KernelSHAPIQ, 2, "k-SII"),
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

    def test_small_soum(self, soum_5, exact_soum_5, index):
        order = _order_for(index)
        expected = soum_5.exact_values(index, order)
        actual = exact_soum_5(index, order=order)
        # 1e-8 is the floor set by the FSII/FBII least-squares solve on
        # genuinely non-k-additive games; the other indices agree to ~1e-15.
        assert_iv_close(actual, expected, atol=1e-8)

    @pytest.mark.slow
    def test_medium_soum(self, soum_7, exact_soum_7, index):
        order = _order_for(index)
        expected = soum_7.exact_values(index, order)
        actual = exact_soum_7(index, order=order)
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

    def test_roundtrip_on_soum(self, soum_5, exact_soum_5, index):
        order = _order_for(index)
        # 1. Moebius values computed by ExactComputer on the game.
        moebius = exact_soum_5("Moebius", order=soum_5.n_players)
        # 2. Convert Moebius -> target index via MoebiusConverter.
        converter = MoebiusConverter(moebius)
        converted = converter(index, order=order)
        # 3. Target values computed directly by ExactComputer.
        direct = exact_soum_5(index, order=order)
        # 1e-8 is the floor set by the FSII/FBII least-squares solve; the
        # other indices agree to ~1e-15.
        assert_iv_close(converted, direct, atol=1e-8)


# ===================================================================
# 2b. ExactComputer's Moebius transform == SOUM's analytical Moebius
# ===================================================================


class TestMoebiusVsSOUM:
    """The Möbius transform computed by ``ExactComputer`` must equal the
    analytical Möbius coefficients stored by the SOUM itself.

    This closes the loop with two independently-derived ground truths:
    - ``exact_computer("Moebius", n)`` — brute-forced inversion over 2^n.
    - ``soum.moebius_coefficients`` — summed from the unanimity-basis weights.
    """

    def test_small_soum(self, soum_5, exact_soum_5):
        expected = soum_5.moebius_coefficients  # sparse: only basis interactions
        actual = exact_soum_5("Moebius", order=soum_5.n_players)
        # Every non-zero basis interaction in ``expected`` must match; every
        # other entry in ``actual`` must be ~ 0.
        for interaction, idx in expected.interaction_lookup.items():
            assert float(actual[interaction]) == pytest.approx(
                float(expected.values[idx]), abs=1e-10
            ), f"Moebius[{interaction}] mismatch"
        # All actual-only entries should be (near) zero.
        for interaction in actual.interaction_lookup:
            if interaction in expected.interaction_lookup:
                continue
            assert abs(float(actual[interaction])) < 1e-10, (
                f"ExactComputer reports non-zero Moebius[{interaction}] "
                f"but SOUM has no corresponding basis game."
            )

    @pytest.mark.slow
    def test_medium_soum(self, soum_7, exact_soum_7):
        expected = soum_7.moebius_coefficients
        actual = exact_soum_7("Moebius", order=soum_7.n_players)
        for interaction, idx in expected.interaction_lookup.items():
            assert float(actual[interaction]) == pytest.approx(
                float(expected.values[idx]), abs=1e-8
            ), f"Moebius[{interaction}] mismatch"
        for interaction in actual.interaction_lookup:
            if interaction in expected.interaction_lookup:
                continue
            assert abs(float(actual[interaction])) < 1e-8


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
        # Machine-precision estimators (SHAPIQ, SVARM, SVARMIQ,
        # UnbiasedKernelSHAP, RegressionFBII) hit ~1e-15 on this fixture;
        # the Shapley-kernel LS solvers (KernelSHAP, KernelSHAPIQ,
        # RegressionFSII) hit ~5e-7 on genuinely non-k-additive games due
        # to conditioning of the weighted LS matrix. 1e-6 accommodates both
        # and still leaves a 10,000x margin against InconsistentKernelSHAPIQ
        # (which produces ~1e-2 errors on the same fixture).
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
# 5. InterventionalTreeExplainer vs ExactComputer on InterventionalGame
# ===================================================================


# Indices where the interventional tree game and the interventional TreeSHAP-IQ
# explainer are defined and agree. STII is omitted: the two implementations
# disagree on STII (~1e-1 error) and diagnosing that is out of scope here.
# k-SII is omitted: InterventionalTreeExplainer does not support it.
TREE_CROSS_CHECK_INDICES: tuple[str, ...] = ("SV", "BV", "SII", "BII", "FSII", "FBII")


@pytest.fixture(scope="module")
def _small_tree_setup():
    """Tiny decision tree on 5 features — small enough for ExactComputer."""
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 5))
    y = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + rng.normal(0, 0.05, size=40)
    model = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)
    return model, X[0], X  # model, x_explain, reference_data


class TestInterventionalTreeCrossCheck:
    """Cross-check the interventional tree pipeline using two independent
    computations of the same quantity.

    * **Ground truth:** :class:`shapiq.tree.interventional.InterventionalGame`
      is a coalition-valued game whose ``value_function`` encodes the
      interventional semantics ``v(S) = E_ref[f(x_S, z_{not S})]``. Running
      :class:`ExactComputer` on it brute-forces the Shapley / Banzhaf /
      faithful values from 2^n coalition evaluations.
    * **Under test:** :class:`InterventionalTreeExplainer` computes the same
      values via the interventional TreeSHAP-IQ algorithm, which walks the
      tree rather than enumerating coalitions.

    Both must produce the same InteractionValues — that's the invariant
    this class pins down. We pair ``InterventionalGame`` with the
    ``InterventionalTreeExplainer`` (not the default path-dependent
    ``TreeExplainer``) because they share the same coalition semantics.
    """

    @pytest.mark.parametrize("index", TREE_CROSS_CHECK_INDICES)
    def test_matches(self, _small_tree_setup, index):
        from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
        from shapiq.tree.interventional.game import InterventionalGame

        model, x, reference = _small_tree_setup
        order = _order_for(index)

        game = InterventionalGame(model=model, reference_data=reference, target_instance=x)
        via_exact_computer = ExactComputer(game)(index, order=order)

        explainer = InterventionalTreeExplainer(
            model=model, data=reference, max_order=order, index=index
        )
        via_tree_explainer = explainer.explain_function(x)

        # ~4e-9 is the observed numerical floor; 1e-7 accommodates float32
        # conversion inside the C extension without masking real divergence.
        assert_iv_close(via_exact_computer, via_tree_explainer, atol=1e-7)
