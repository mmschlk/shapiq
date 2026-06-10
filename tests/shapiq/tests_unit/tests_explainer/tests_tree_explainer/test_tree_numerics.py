"""Tests for the exact Vandermonde solves in ``shapiq.tree._numerics``."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebpts2

from shapiq.tree import RepresentationLimitError, TreeExplainer, TreeModel
from shapiq.tree._numerics import (
    _solve_vandermonde_cached,
    clear_solver_cache,
    solve_vandermonde,
)
from shapiq.tree.linear.explainer import LinearTreeSHAP, get_norm_weight
from shapiq.tree.treeshapiq import TreeSHAPIQ


def _solve_exact_reference(points: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Oracle: exact rational Gaussian elimination for ``vander(points).T x = rhs``."""
    n = len(points)
    p = [Fraction(t) for t in points.tolist()]
    rows = [
        [p[j] ** (n - 1 - k) for j in range(n)] + [Fraction(v)] for k, v in enumerate(rhs.tolist())
    ]
    for col in range(n):
        pivot = next(r for r in range(col, n) if rows[r][col] != 0)
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(n):
            if r != col and rows[r][col] != 0:
                factor = rows[r][col] / rows[col][col]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[col], strict=True)]
    return np.array([float(rows[k][n] / rows[k][k]) for k in range(n)])


@pytest.mark.parametrize("grid_size", [2, 4, 8, 12, 16, 20])
def test_matches_explicit_inverse_on_well_conditioned_grids(grid_size):
    """On well-conditioned systems the result agrees with the previous inverse.

    Larger grids are deliberately excluded here: above size ~20 the explicit
    inverse starts drifting at the ~1e-7 level (at size 26 the peak prefix
    condition number is already ~4e11), so the old formulation stops being a
    valid reference; those sizes are covered by the exact-rational oracle test
    below instead.
    """
    D = chebpts2(grid_size)
    rng = np.random.default_rng(0)
    for i in range(2, grid_size + 1):
        rhs = rng.normal(size=i)
        expected = np.linalg.inv(np.vander(D[:i]).T).dot(rhs)
        np.testing.assert_allclose(solve_vandermonde(D[:i], rhs), expected, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("grid_size", [12, 28, 35, 45])
def test_bitwise_equal_to_exact_rational_solution(grid_size):
    """Every prefix solve equals the exact rational solution rounded to float64.

    Sizes 28-45 are exactly the regime where the previous ``inv``-based code lost
    precision (peak prefix condition numbers above 1e12) or silently returned wrong
    values (rank-deficient to machine precision from size ~32).
    """
    D = chebpts2(grid_size)
    rng = np.random.default_rng(1)
    for i in (2, grid_size // 2, grid_size):  # interior prefixes are the worst-conditioned
        rhs = rng.normal(size=i)
        exact = _solve_exact_reference(D[:i], rhs)
        result = solve_vandermonde(D[:i], rhs)
        np.testing.assert_array_equal(result, exact)


def test_replaced_inverse_construction_was_wrong_in_the_failure_band():
    """Regression guard: the construction this module replaced fails where the solver is exact.

    At grid size 32 with the library's own right-hand side the explicit inverse is
    off by an absolute error of ~2e2 — entries of magnitude ~3e7 that downstream
    inner products cancel to an O(prediction) result, so an absolute error of that
    size destroys the result entirely. This test fails if the exact solver is ever
    swapped back for the inverse.
    """
    D = chebpts2(32)
    rhs = 1.0 / get_norm_weight(31)
    exact = _solve_exact_reference(D, rhs)
    np.testing.assert_array_equal(solve_vandermonde(D, rhs), exact)
    try:
        old = np.linalg.inv(np.vander(D).T) @ rhs
        assert np.max(np.abs(old - exact)) > 1.0  # measured ~2e2; anything >1 is fatal downstream
    except np.linalg.LinAlgError:
        pass  # an outright failure is equally disqualifying for the old construction


def test_issue_545_scenario_raises_instead_of_returning_wrong_values():
    """The end-to-end scenario of issue #545 is refused instead of silently wrong.

    A decision tree fit on one-hot-style data (every sample distinguished by its
    own indicator feature) is forced into a depth-39 chain over 39 features, the
    structure that previously returned silently wrong Shapley values (interpolation
    degree in the ~32-59 failure band, no warning). Both the LinearTreeSHAP degree
    (tree depth) and the TreeSHAPIQ degree (features in the tree) exceed the
    representation limit, so there is no re-route and construction must raise.
    """
    sklearn_tree = pytest.importorskip("sklearn.tree")
    x_data = np.eye(40)
    y_target = np.arange(40, dtype=float)
    model = sklearn_tree.DecisionTreeRegressor(random_state=0).fit(x_data, y_target)
    assert model.get_depth() >= 30  # in the previously silent failure band
    assert len({int(f) for f in model.tree_.feature if f >= 0}) >= 30  # no re-route possible
    with pytest.raises(RepresentationLimitError):
        TreeExplainer(model=model, index="SV", max_order=1)


def test_single_point_system():
    """The size-1 system is the identity."""
    np.testing.assert_array_equal(solve_vandermonde(np.array([0.3]), np.array([2.5])), [2.5])


def test_non_chebyshev_clustered_grid_is_solved_exactly():
    """Custom interpolation grids (LinearTreeSHAP(base_func=...)) are handled exactly."""
    grid = np.linspace(0.0, 1.0, 18) ** 8 + 0.01  # clustered near zero
    rng = np.random.default_rng(2)
    rhs = rng.normal(size=18)
    np.testing.assert_array_equal(solve_vandermonde(grid, rhs), _solve_exact_reference(grid, rhs))


def test_coincident_nodes_raise():
    """Genuinely singular systems (repeated nodes) raise instead of returning garbage."""
    with pytest.raises(ValueError, match="pairwise distinct"):
        solve_vandermonde(np.array([0.1, 0.5, 0.5]), np.ones(3))


def test_mismatched_input_shapes_raise():
    """Mismatched or non-1-D inputs are rejected up front with a clear message."""
    with pytest.raises(ValueError, match="equal length"):
        solve_vandermonde(np.array([0.1, 0.3, 0.5]), np.ones(2))
    with pytest.raises(ValueError, match="1-D"):
        solve_vandermonde(np.ones((2, 2)), np.ones((2, 2)))


def test_non_finite_inputs_raise():
    """NaN or inf nodes/right-hand sides are rejected with a clear message."""
    with pytest.raises(ValueError, match="must be finite"):
        solve_vandermonde(np.array([0.1, np.nan, 0.5]), np.ones(3))
    with pytest.raises(ValueError, match="must be finite"):
        solve_vandermonde(np.array([0.1, 0.3, 0.5]), np.array([1.0, np.inf, 1.0]))


def test_nearly_coincident_nodes_raise_a_documented_error():
    """Nodes packed at consecutive float64 ulps push the exact solution beyond the
    float64 range; the solver raises the documented ValueError instead of an opaque
    OverflowError."""
    nodes = [0.5]
    for _ in range(21):
        nodes.append(float(np.nextafter(nodes[-1], 1.0)))
    with pytest.raises(ValueError, match="too close together"):
        solve_vandermonde(np.array(nodes), np.ones(len(nodes)))


def test_nodes_below_the_first_rung_resolution_climb_the_ladder():
    """Distinct nodes that collapse at 128 fractional bits are not misreported.

    Tiny-magnitude nodes can be genuinely distinct in float64 yet land on the
    same scaled integer at the first precision rung; the solver must climb to a
    finer rung and solve exactly rather than report them as coincident.
    """
    a = 2.0**-100
    points = np.array([a, a + 2.0**-152])  # identical at 128 bits, distinct at 256
    rhs = np.array([1.0, 3.0])
    np.testing.assert_array_equal(
        solve_vandermonde(points, rhs), _solve_exact_reference(points, rhs)
    )


def _deep_chain_tree(depth: int, n_features: int | None = None):
    """A deterministic decision tree of exactly the requested depth.

    Builds a left-leaning chain: every internal node splits on a fresh feature
    (or cycles through ``n_features`` features when given) and has a leaf on the
    right. The depth is exact by construction, so the deep-tree tests never
    silently skip.
    """
    n_nodes = 2 * depth + 1
    children_left = np.full(n_nodes, -1)
    children_right = np.full(n_nodes, -1)
    features = np.full(n_nodes, -2)
    thresholds = np.full(n_nodes, np.nan)
    values = np.zeros(n_nodes)
    weights = np.ones(n_nodes)
    rng = np.random.default_rng(0)
    # Internal nodes 0..depth-1 form the left chain; node i's right child is the
    # leaf depth+i, and the last internal node's left child is the leaf 2*depth.
    for level in range(depth):
        children_left[level] = level + 1 if level < depth - 1 else 2 * depth
        children_right[level] = depth + level
        features[level] = level if n_features is None else level % n_features
        thresholds[level] = 0.5
        weights[level] = depth - level + 1  # number of leaves below the node
    for leaf in range(depth, 2 * depth + 1):
        values[leaf] = float(rng.normal())
    return TreeModel(
        children_left=children_left,
        children_right=children_right,
        children_missing=children_left.copy(),
        features=features,
        thresholds=thresholds,
        values=values,
        node_sample_weight=weights,
    )


@pytest.mark.parametrize(("depth", "tolerance"), [(20, 1e-6), (24, 1e-4), (28, 1e-2)])
def test_deep_tree_explanations_satisfy_completeness(depth, tolerance):
    """Deep trees produce values satisfying the completeness axiom.

    The tolerances are the documented downstream float64 bound (``max|N| * 1e-13``):
    with exact coefficients the only remaining error is the double-precision
    evaluation of the (exponentially large) N entries, which scales with the
    largest entry.
    """
    tree = _deep_chain_tree(depth)
    explainer = LinearTreeSHAP(tree)
    x = np.zeros(depth)  # walks the full left chain through every split
    explanation = explainer.explain_function(x)
    prediction = tree.predict_one(x)
    gap = float(np.sum(explanation.values) - prediction)
    assert abs(gap) < tolerance * max(1.0, abs(prediction))


@pytest.mark.parametrize(("depth", "is_supported"), [(29, True), (30, False)])
def test_representation_limit_boundary_is_pinned(depth, is_supported):
    """The largest supported interpolation degree on the default grids is exactly 29.

    This pins the "~29" quoted in the error message, the CHANGELOG, and the
    docstrings to the actual refuse/accept boundary, so a drift in the grids or
    the limit cannot silently decouple the prose from the behavior.
    """
    tree = _deep_chain_tree(depth)
    if is_supported:
        explainer = LinearTreeSHAP(tree)
        x = np.zeros(depth)
        gap = float(np.sum(explainer.explain_function(x).values)) - tree.predict_one(x)
        assert abs(gap) < 1e-2 * max(1.0, abs(tree.predict_one(x)))
    else:
        with pytest.raises(RepresentationLimitError):
            LinearTreeSHAP(tree)


@pytest.mark.parametrize(("depth", "is_supported"), [(25, True), (26, False)])
def test_treeshapiq_representation_limit_boundary_is_pinned(depth, is_supported):
    """TreeSHAPIQ's boundary is lower than LinearTreeSHAP's 29/30.

    The identity N matrix (rhs of ones, built for every index) crosses the
    magnitude limit at degree 26 already, so it is the binding constraint for
    all TreeSHAPIQ indices; this pins the "~25" quoted in the error message.
    """
    tree = _deep_chain_tree(depth)
    if is_supported:
        TreeSHAPIQ(model=tree, max_order=2, index="SII")
    else:
        with pytest.raises(RepresentationLimitError):
            TreeSHAPIQ(model=tree, max_order=2, index="SII")


@pytest.mark.parametrize("depth", [35, 60])
def test_too_deep_trees_are_refused_with_a_clear_error(depth):
    """Beyond the float64 representation limit the explainer refuses loudly.

    Depth 35 previously returned silently wrong values (the ~32-59 band where the
    explicit inverse is rank-deficient to machine precision); depth 60 crashed
    with an unexplained ``LinAlgError``. The coefficients are now exact at any
    depth, but their magnitude exceeds what the downstream float64 pipeline can
    evaluate without destroying the result, so construction raises an
    explanatory error instead.
    """
    tree = _deep_chain_tree(depth)
    with pytest.raises(RepresentationLimitError, match=r"interpolation degree.*too large"):
        LinearTreeSHAP(tree)


def test_clear_solver_cache_forces_recomputation():
    """clear_solver_cache drops memoized solutions; a fresh solve still matches the oracle."""
    D = chebpts2(10)
    rhs = np.ones(10)
    solve_vandermonde(D, rhs)
    assert _solve_vandermonde_cached.cache_info().currsize > 0
    clear_solver_cache()
    assert _solve_vandermonde_cached.cache_info().currsize == 0
    np.testing.assert_array_equal(solve_vandermonde(D, rhs), _solve_exact_reference(D, rhs))


def test_solver_results_are_cached_and_isolated():
    """Identical (points, rhs) solves hit the cache, and callers cannot corrupt it."""
    D = chebpts2(14)
    rhs = np.ones(14)
    first = solve_vandermonde(D, rhs)
    first[0] = 12345.0  # mutate the returned copy
    second = solve_vandermonde(D, rhs)
    assert second[0] != 12345.0  # the cached value must be unaffected
    np.testing.assert_array_equal(second, _solve_exact_reference(D, rhs))


def test_deep_narrow_tree_routes_to_treeshapiq():
    """A deep tree over few features is re-routed instead of refused.

    LinearTreeSHAP's interpolation degree is the tree depth (above the limit
    here), but TreeSHAPIQ's is min(depth, features in the tree) = 5, so
    ``TreeExplainer`` falls back to TreeSHAPIQ and still produces order-1
    Shapley values satisfying completeness.
    """
    depth = 40
    with pytest.raises(RepresentationLimitError):
        LinearTreeSHAP(_deep_chain_tree(depth, n_features=5))

    tree = _deep_chain_tree(depth, n_features=5)
    explainer = TreeExplainer(model=[tree], index="SV", max_order=1)
    x = np.zeros(depth)
    explanation = explainer.explain(x)
    prediction = tree.predict_one(x)
    gap = abs(float(np.sum(explanation.values)) - prediction)
    assert gap < 1e-9 * max(1.0, abs(prediction))


def test_deep_narrow_tree_is_rerouted_for_sii_as_well():
    """The re-route covers both order-1 fast-path indices, not just ``"SV"``.

    ``index="SII"`` with ``max_order=1`` also takes the LinearTreeSHAP fast path;
    the explainer validation normalizes it to ``"SV"`` (with a warning), which is
    exactly the label the re-route assigns, so the re-routed result satisfies the
    same completeness.
    """
    depth = 40
    tree = _deep_chain_tree(depth, n_features=5)
    with pytest.warns(UserWarning, match="SII generalizes 'SV'"):
        explainer = TreeExplainer(model=[tree], index="SII", max_order=1)
    assert len(explainer._treeshapiq_explainers) == 1
    x = np.zeros(depth)
    explanation = explainer.explain(x)
    prediction = tree.predict_one(x)
    gap = abs(float(np.sum(explanation.values)) - prediction)
    assert gap < 1e-9 * max(1.0, abs(prediction))


def test_treeshapiq_n_matrices_are_gated_too():
    """The representation limit also fires inside TreeSHAPIQ's own N matrices.

    For higher-order indices there is no re-route: a tree whose feature-bounded
    interpolation degree exceeds the limit is refused at construction instead of
    returning silently wrong interaction values.
    """
    tree = _deep_chain_tree(35)  # 35 distinct features -> TreeSHAPIQ degree 35
    with pytest.raises(RepresentationLimitError):
        TreeSHAPIQ(model=tree, max_order=2, index="k-SII")


def test_mixed_ensemble_keeps_the_fast_path_for_shallow_trees():
    """Only the over-deep tree leaves the LinearTreeSHAP fast path in an ensemble.

    The shallow tree stays on LinearTreeSHAP, the deep-narrow tree is re-routed to
    TreeSHAPIQ, and the mixed aggregation still satisfies completeness for the
    two-tree ensemble.
    """
    assert issubclass(RepresentationLimitError, ValueError)
    depth = 40
    deep_tree = _deep_chain_tree(depth, n_features=5)
    shallow_tree = _deep_chain_tree(8)
    explainer = TreeExplainer(model=[shallow_tree, deep_tree], index="SV", max_order=1)
    assert len(explainer._lineartreeshap_explainers) == 1
    assert len(explainer._treeshapiq_explainers) == 1

    x = np.zeros(depth)
    explanation = explainer.explain(x)
    prediction = shallow_tree.predict_one(x) + deep_tree.predict_one(x)
    gap = abs(float(np.sum(explanation.values)) - prediction)
    assert gap < 1e-9 * max(1.0, abs(prediction))
