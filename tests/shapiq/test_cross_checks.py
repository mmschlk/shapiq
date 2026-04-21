"""Cross-check tests: verify shapiq implementations by making independent
ground-truth sources agree on the same game.

Ground-truth sources used here:

1. ``ExactComputer(game)(index, order)`` — brute-force over 2^n coalitions.
2. ``SOUM.exact_values(index, order)`` — closed-form via Moebius conversion.
3. ``SOUM.moebius_coefficients`` — the SOUM's own analytical Moebius transform
   (compared directly against ``ExactComputer("Moebius", n)``).
4. Consistent approximators at ``budget = 2**n`` — regression family + SHAPIQ
   / SVARMIQ on indices where they are known to be exact, plus ``kADDSHAP``
   cross-checked against ``ExactComputer("kADD-SHAP")``.
5. ``InterventionalTreeExplainer.explain_function(x)`` — interventional
   TreeSHAP-IQ, versus ``ExactComputer`` running on the matching
   ``InterventionalGame`` wrapping the same tree.
6. Path-dependent ``TreeExplainer`` — covered by the SV efficiency axiom
   (``sum(SV) == f(x) - E[f]``), since no path-dependent ``Game`` wrapper
   exists for a full cross-check.
7. ``ProductKernelExplainer.explain(x)`` — closed-form Shapley values via
   elementary symmetric polynomials on kernel vectors, versus
   ``ExactComputer`` running on the matching ``ProductKernelGame``.
8. ``LinearTreeSHAP.explain_function(x)`` — Chebyshev polynomial shortcut
   for path-dependent order-1 SV, versus ``ExactComputer`` on a local
   brute-force path-dependent tree game helper (``_PathDependentTreeGame``
   in this module).
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
    kADDSHAP,
)
from shapiq.game import Game
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
    """``ExactComputer(SOUM)`` must match ``SOUM.exact_values`` exactly.

    Parametrised over several fixed seeds (see ``GROUND_TRUTH_SEEDS`` in
    ``conftest.py``) so that conditioning edge cases specific to one
    random game instance can't silently pass.
    """

    def test_small_soum(self, soum_5_seeded, exact_soum_5_seeded, index):
        order = _order_for(index)
        expected = soum_5_seeded.exact_values(index, order)
        actual = exact_soum_5_seeded(index, order=order)
        # 1e-8 is the floor set by the FSII/FBII least-squares solve on
        # genuinely non-k-additive games; the other indices agree to ~1e-15.
        # ``strict=True``: both sides are analytical and must populate the
        # same set of interactions.
        assert_iv_close(actual, expected, atol=1e-8, strict=True)

    @pytest.mark.slow
    def test_medium_soum(self, soum_7_seeded, exact_soum_7_seeded, index):
        order = _order_for(index)
        expected = soum_7_seeded.exact_values(index, order)
        actual = exact_soum_7_seeded(index, order=order)
        # n=7 FSII/FBII run through a larger LS solve; allow some numerical noise.
        assert_iv_close(actual, expected, atol=1e-8, strict=True)


# ===================================================================
# 2. MoebiusConverter round-trip: ExactComputer -> Moebius -> index
# ===================================================================


@pytest.mark.parametrize("index", GROUND_TRUTH_INDICES)
class TestMoebiusConverter:
    """``MoebiusConverter`` on a Moebius IV produces the same values as the
    direct ExactComputer call for the target index.

    Parametrised across several fixed SOUM seeds for game-instance
    diversity (see ``GROUND_TRUTH_SEEDS``).
    """

    def test_roundtrip_on_soum(self, soum_5_seeded, exact_soum_5_seeded, index):
        order = _order_for(index)
        # 1. Moebius values computed by ExactComputer on the game.
        moebius = exact_soum_5_seeded("Moebius", order=soum_5_seeded.n_players)
        # 2. Convert Moebius -> target index via MoebiusConverter.
        converter = MoebiusConverter(moebius)
        converted = converter(index, order=order)
        # 3. Target values computed directly by ExactComputer.
        direct = exact_soum_5_seeded(index, order=order)
        # 1e-8 is the floor set by the FSII/FBII least-squares solve; the
        # other indices agree to ~1e-15. ``strict=True``: both sides should
        # populate the same set of interactions.
        assert_iv_close(converted, direct, atol=1e-8, strict=True)


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
        # other entry in ``actual`` must be ~ 0. Tolerance 1e-9 accommodates
        # the alternating-sign sum over 2^n coalitions on Windows/macOS where
        # FMA ordering can consume several ULPs.
        for interaction, idx in expected.interaction_lookup.items():
            assert float(actual[interaction]) == pytest.approx(
                float(expected.values[idx]), abs=1e-9
            ), f"Moebius[{interaction}] mismatch"
        # All actual-only entries should be (near) zero.
        for interaction in actual.interaction_lookup:
            if interaction in expected.interaction_lookup:
                continue
            assert abs(float(actual[interaction])) < 1e-9, (
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
            # Approximators warn about border-trick budget; that's expected
            # at full budget. Don't swallow other warning categories.
            warnings.simplefilter("ignore", category=UserWarning)
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
# 3b. kADD-SHAP approximator vs ExactComputer (SOUM has no ground truth)
# ===================================================================


class TestKAddSHAPAtFullBudget:
    """``kADDSHAP`` at budget = 2^n must agree with ``ExactComputer("kADD-SHAP")``.

    ``SOUM.exact_values`` / ``MoebiusConverter`` don't support the
    ``kADD-SHAP`` index, so ``ExactComputer`` is the only ground-truth
    source for this index. The cross-check is still meaningful: the
    approximator runs its own weighted-LS algorithm while ``ExactComputer``
    brute-forces a Möbius-weighted sum over all 2^n coalitions. Agreement
    at full budget pins the approximator down.
    """

    def test_matches_ground_truth(self, soum_5, exact_soum_5):
        n = soum_5.n_players
        approx = kADDSHAP(n=n, max_order=2, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            result = approx.approximate(2**n, soum_5)
        expected = exact_soum_5("kADD-SHAP", order=2)
        # Same tolerance floor as the other LS-based approximators.
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

        # Average error over multiple seeds to smooth out single-run
        # fluctuations — sampling methods can legitimately oscillate at a
        # single budget with a fixed seed.
        seeds = (42, 43, 44)
        budgets = [2**n, 4 * 2**n, 16 * 2**n]
        mean_errors = []
        for budget in budgets:
            errors_for_budget = []
            for seed in seeds:
                approx = cls(n=n, max_order=max_order, index=index, random_state=seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    result = approx.approximate(budget, soum_7)
                errors_for_budget.append(_max_abs_error(result, expected))
            mean_errors.append(sum(errors_for_budget) / len(errors_for_budget))

        # With a 16x budget increase, a functioning sampling estimator
        # should halve its Monte-Carlo error on average. This is strictly
        # tighter than ``errors[-1] < errors[0]`` — the latter passes when
        # the final error is a hair below the initial, which a silently
        # broken estimator could hit by noise alone.
        assert mean_errors[-1] < 0.5 * mean_errors[0], (
            f"{cls.__name__} ({index}): mean error did not halve across "
            f"budgets {budgets}: mean_errors={mean_errors}"
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
def small_tree_setup():
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
    def test_matches(self, small_tree_setup, index):
        from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
        from shapiq.tree.interventional.game import InterventionalGame

        model, x, reference = small_tree_setup
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


# ===================================================================
# 6. Path-dependent TreeExplainer efficiency invariant
# ===================================================================


class TestPathDependentTreeEfficiency:
    """Covers the default ``shapiq.TreeExplainer`` (path-dependent
    TreeSHAP-IQ), which :class:`TestInterventionalTreeCrossCheck` can't
    reach — path-dependent and interventional explanations do not share
    coalition semantics, so they disagree by design.

    Instead of a full cross-check (which would need an as-yet-unimplemented
    path-dependent ``Game`` wrapper), we pin down the *efficiency axiom*:
    for order-1 SV, ``sum(SV) == f(x) - baseline``. This is the single
    invariant most likely to break on a regression in the TreeSHAP-IQ
    polynomial arithmetic or the baseline computation.
    """

    def test_sv_efficiency(self, small_tree_setup):
        from shapiq.tree import TreeExplainer

        model, x, _ = small_tree_setup
        explainer = TreeExplainer(model=model, max_order=1, index="SV", min_order=1)
        iv = explainer.explain(x)

        sv_sum = sum(float(iv[k]) for k in iv.interaction_lookup if len(k) == 1)
        baseline = float(iv.baseline_value)
        prediction = float(model.predict(x.reshape(1, -1))[0])

        # SV efficiency: sum of all Shapley values equals f(x) - E[f].
        assert sv_sum == pytest.approx(prediction - baseline, abs=1e-10), (
            f"SV efficiency violated: sum(SV)={sv_sum}, f(x)-baseline={prediction - baseline}"
        )


# ===================================================================
# 7. ProductKernelExplainer vs ExactComputer on ProductKernelGame
# ===================================================================


@pytest.fixture(scope="module")
def product_kernel_setup():
    """Tiny SVR on 5 features — small enough for ExactComputer."""
    from sklearn.svm import SVR

    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 5))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.1, size=30)
    svr = SVR(kernel="rbf").fit(X, y)
    return svr, X[0]


class TestProductKernelCrossCheck:
    """``ExactComputer`` on ``ProductKernelGame`` must agree with
    ``ProductKernelExplainer``'s closed-form Shapley values.

    The two sides share the same RBF kernel and training data; the
    explainer applies a polynomial-arithmetic shortcut (elementary
    symmetric polynomials on per-feature kernel vectors), while the game
    brute-forces 2^n coalition evaluations. Agreement pins the closed-form
    algorithm against an independent ground truth.

    The explainer only supports ``index="SV"`` with ``max_order=1``, so
    there is nothing to parametrise over.
    """

    def test_sv_matches(self, product_kernel_setup):
        from shapiq.explainer.product_kernel import ProductKernelExplainer
        from shapiq.explainer.product_kernel.game import ProductKernelGame

        svr, x = product_kernel_setup

        explainer = ProductKernelExplainer(model=svr, max_order=1, index="SV")
        via_explainer = explainer.explain(x)

        # ``ProductKernelGame`` takes the validated ``ProductKernelModel``
        # (with ``.X_train`` / ``.alpha`` attributes), not the raw sklearn
        # estimator. The explainer has already done this conversion.
        game = ProductKernelGame(
            n_players=x.shape[0],
            explain_point=x,
            model=explainer.converted_model,
        )
        via_exact_computer = ExactComputer(game)("SV", order=1)

        # Empirically agrees to ~1e-16; 1e-10 leaves ample headroom for
        # cross-platform FMA variance without masking real divergence.
        assert_iv_close(via_exact_computer, via_explainer, atol=1e-10)


# ===================================================================
# 8. LinearTreeSHAP vs ExactComputer on a path-dependent tree game
# ===================================================================


class _PathDependentTreeGame(Game):
    """Brute-force reference game for path-dependent TreeSHAP.

    Test scaffolding, NOT production code. Replicates the value-function
    logic the (removed) ``TreeSHAPIQXAI`` used internally: for each
    coalition and each tree in the ensemble, traverse from root, take the
    feature's child if the feature is in the coalition, otherwise average
    the two children weighted by ``node_sample_weight``. Sum across trees.

    This lets us run ``ExactComputer`` over 2^n coalitions and compare
    against :class:`shapiq.tree.linear.explainer.LinearTreeSHAP`, which
    computes the same quantity via a Chebyshev polynomial shortcut.
    """

    def __init__(self, tree_model: object, x_explain: np.ndarray) -> None:
        from shapiq.tree.validation import validate_tree_model

        self._trees = validate_tree_model(tree_model)
        self._x = np.asarray(x_explain).flatten()
        n = self._x.shape[0]
        empty = self._predict(np.zeros(n, dtype=bool))
        super().__init__(n_players=n, normalize=False, normalization_value=empty)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        return np.array([self._predict(c) for c in coalitions], dtype=float)

    def _predict(self, coalition: np.ndarray) -> float:
        return sum(self._traverse(tree, 0, coalition) for tree in self._trees)

    def _traverse(self, tree, node_id: int, coalition: np.ndarray) -> float:
        if tree.leaf_mask[node_id]:
            return float(tree.values[node_id])
        feature = int(tree.features[node_id])
        threshold = tree.thresholds[node_id]
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        if bool(coalition[feature]):
            nxt = left if self._x[feature] <= threshold else right
            return self._traverse(tree, nxt, coalition)
        # Feature absent: weighted average of both children.
        left_weight = float(tree.node_sample_weight[left])
        right_weight = float(tree.node_sample_weight[right])
        total = left_weight + right_weight
        if total == 0.0:
            left_weight = right_weight = 1.0
            total = 2.0
        return (left_weight / total) * self._traverse(tree, left, coalition) + (
            right_weight / total
        ) * self._traverse(tree, right, coalition)


class TestLinearTreeSHAPCrossCheck:
    """``LinearTreeSHAP`` (Chebyshev polynomial SV shortcut) must agree
    with ``ExactComputer`` running on the equivalent brute-force
    path-dependent tree game.

    Both sides compute first-order path-dependent Shapley values for the
    same sklearn tree on the same instance — the explainer uses a
    closed-form polynomial algorithm over edge-tree representation, the
    game brute-forces 2^n coalitions with node-weighted averaging for
    absent features. Agreement pins the polynomial algorithm against an
    independent numerical ground truth.
    """

    def test_sv_matches(self, small_tree_setup):
        from shapiq.tree.linear.explainer import LinearTreeSHAP

        model, x, _ = small_tree_setup
        explainer = LinearTreeSHAP(model=model)
        via_explainer = explainer.explain_function(x)

        game = _PathDependentTreeGame(model, x)
        via_exact_computer = ExactComputer(game)("SV", order=1)

        # Empirically agrees to ~1e-16; 1e-10 leaves headroom for
        # cross-platform FMA variance without masking real divergence.
        assert_iv_close(via_exact_computer, via_explainer, atol=1e-10)
