from __future__ import annotations

import numpy as np
import pytest

from shapiq.game_theory.exact import ExactComputer
from shapiq.tree import InterventionalGame, InterventionalTreeSHAPIQ

SEED = 1337
np.random.seed(SEED)


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FBII", 1),
        ("FSII", 2),
        ("STII", 2),
    ],
)
def test_correct_calculation_dt_reg_index_order(dt_reg_model, reg_data, index, order):
    X_train, X_test, _y_train, _y_test = reg_data
    model = dt_reg_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeSHAPIQ
    own_interventional_explainer = InterventionalTreeSHAPIQ(
        model, X_train, index=index, max_order=order
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(model, X_train, point_to_explain.flatten())
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementatoin matches Exact Computer
    for _i, interaction in enumerate(own_interactions.keys()):
        if len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
        ("STII", 3),
    ],
)
def test_correct_calculation_dt_clas_index_order(dt_clf_model, cls_data, index, order):
    CLASS_INDEX = 1
    X_train, X_test, _, _ = cls_data
    model = dt_clf_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeSHAPIQ
    own_interventional_explainer = InterventionalTreeSHAPIQ(
        model,
        X_train,
        index=index,
        max_order=order,
        class_index=CLASS_INDEX,
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(
        model, X_train, point_to_explain.flatten(), class_index=CLASS_INDEX
    )
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementatoin matches Exact Computer
    for _i, interaction in enumerate(own_interactions.keys()):
        if len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
        ("STII", 3),
    ],
)
def test_correct_calculation_rf_reg_index_order(rf_reg_model, reg_data, index, order):
    X_train, X_test, _y_train, _y_test = reg_data
    model = rf_reg_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeSHAPIQ
    own_interventional_explainer = InterventionalTreeSHAPIQ(
        model, X_train, max_order=order, index=index
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(model, X_train, point_to_explain.flatten())
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementatoin matches Exact Computer
    for _i, interaction in enumerate(own_interactions.keys()):
        if len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
        ("STII", 3),
    ],
)
def test_correct_calculation_rf_clas_index_order(rf_clf_model, cls_data, index, order):
    CLASS_INDEX = 1
    X_train, X_test, _, _ = cls_data
    model = rf_clf_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeSHAPIQ
    own_interventional_explainer = InterventionalTreeSHAPIQ(
        model,
        X_train,
        max_order=order,
        index=index,
        class_index=CLASS_INDEX,
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(
        model, X_train, point_to_explain.flatten(), class_index=CLASS_INDEX
    )
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementatoin matches Exact Computer
    for interaction in own_interactions:
        if len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
        ("STII", 3),
    ],
)
def test_correct_calculation_xgb_reg_index_order(xgb_reg_model, reg_data, index, order):
    X_train, X_test, _, _ = reg_data
    model = xgb_reg_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeSHAPIQ
    own_interventional_explainer = InterventionalTreeSHAPIQ(
        model, X_train, max_order=order, index=index
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(model, X_train, point_to_explain.flatten())
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementation matches Exact Computer
    for _, interaction in enumerate(own_interactions.keys()):
        # Using 1e-5 tolerance due to float32 vs float64 precision differences in XGBoost calculations
        if len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-5,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
        ("STII", 3),
    ],
)
def test_correct_calculation_xgb_clas_index_order(xgb_clf_model, cls_data, index, order):
    CLASS_INDEX = 1
    X_train, X_test, _y_train, _y_test = cls_data
    model = xgb_clf_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeSHAPIQ
    own_interventional_explainer = InterventionalTreeSHAPIQ(
        model,
        X_train,
        max_order=order,
        index=index,
        class_index=CLASS_INDEX,
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(
        model, X_train, point_to_explain.flatten(), class_index=CLASS_INDEX
    )
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementatoin matches Exact Computer
    for _, interaction in enumerate(own_interactions.keys()):
        if len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
        ("STII", 3),
    ],
)
def test_correct_calculation_lgbm_reg_index_order(lightgbm_reg_model, reg_data, index, order):
    X_train, X_test, _y_train, _y_test = reg_data
    model = lightgbm_reg_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeSHAPIQ
    own_interventional_explainer = InterventionalTreeSHAPIQ(
        model, X_train, max_order=order, index=index
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(model, X_train, point_to_explain.flatten())
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementation matches Exact Computer
    for _, interaction in enumerate(own_interactions.keys()):
        if len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
        ("STII", 3),
    ],
)
def test_correct_calculation_lgbm_clas_index_order(lightgbm_clf_model, cls_data, index, order):
    CLASS_INDEX = 1
    X_train, X_test, _, _ = cls_data
    model = lightgbm_clf_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeSHAPIQ
    own_interventional_explainer = InterventionalTreeSHAPIQ(
        model,
        X_train,
        max_order=order,
        index=index,
        class_index=CLASS_INDEX,
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(
        model, X_train, point_to_explain.flatten(), class_index=CLASS_INDEX
    )
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementatoin matches Exact Computer
    for _, interaction in enumerate(own_interactions.keys()):
        if len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )


@pytest.fixture(params=["structural", "sparse"])
def interventional_route(request, monkeypatch):
    """Both C routes of ``InterventionalTreeSHAPIQ``: the structural layout, or the sparse
    per-explanation kernel (forced by a zero memory budget)."""
    import shapiq.tree.interventional.computer as interventional_module

    if request.param == "sparse":
        monkeypatch.setattr(interventional_module, "_STRUCTURAL_MAX_ROWS", 0)
    return request.param


@pytest.mark.parametrize(("index", "order"), [("STII", 4), ("FSII", 4)])
def test_correct_calculation_index_order_4(
    dt_reg_model, reg_data, index, order, interventional_route
):
    """Indices with any-order closed forms match the ExactComputer at order 4 on both routes.

    The dense correctness tests above stop at order 3; at order 4 the weights come from
    ``weight_func`` in ``weights.cpp`` on both the structural and the sparse kernel. This
    guards the per-index dispatch there (STII and FSII fell back to the top-order-only
    general path before).
    """
    X_train, X_test, _y_train, _y_test = reg_data
    model = dt_reg_model
    point_to_explain = X_test[0:1]

    own_interventional_explainer = InterventionalTreeSHAPIQ(
        model, X_train, index=index, max_order=order
    )
    assert own_interventional_explainer._use_sparse_path == (interventional_route == "sparse")
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    interventional_game = InterventionalGame(model, X_train, point_to_explain.flatten())
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    for interaction in own_interactions:
        if len(interaction) == 0:
            continue
        assert np.isclose(
            own_interactions[interaction],
            game_interactions.get(interaction, 0),
            atol=1e-6,
        )


def _perfect_tree_with_distinct_features(depth: int, seed: int = 0):
    """A perfect binary tree of the given depth splitting each node on its own feature.

    It holds ``2**depth - 1`` features while a root-to-leaf path holds only ``depth``, so a
    modest depth reaches a feature space wide enough to overflow the sparse kernel's keys.
    """
    n_decision_nodes = 2**depth - 1
    node = np.arange(2 ** (depth + 1) - 1)
    is_leaf = node >= n_decision_nodes
    node_depth = np.floor(np.log2(node + 1)).astype(int)
    rng = np.random.default_rng(seed)
    tree = {
        "children_left": np.where(is_leaf, -1, 2 * node + 1),
        "children_right": np.where(is_leaf, -1, 2 * node + 2),
        "children_missing": np.where(is_leaf, -1, 2 * node + 1),
        "features": np.where(is_leaf, -2, node),
        "thresholds": np.where(is_leaf, -2.0, 0.0),
        "node_sample_weight": 2.0 ** (depth - node_depth),
        "values": np.where(is_leaf, rng.normal(size=node.size), 0.0),
    }
    return tree, rng.normal(size=n_decision_nodes)


def test_sparse_path_flat_keys_match_the_bitset_fallback(monkeypatch):
    """The sparse kernel's flat subset map and its BitSet-keyed fallback agree.

    The sparse kernel keys each subset as an int64 in base ``n_features + 1`` and falls back
    to a BitSet-keyed map once ``(n_features + 1) ** max_order`` overflows an int64. With 255
    features, order 3 runs the flat map and order 8 the fallback; SII values do not depend on
    the maximum order, so both must touch the same subsets with the same values. Feature 0
    splits the root, so subsets of different sizes with and without a leading 0 all occur --
    an encoding that let them collide would fail here.
    """
    import shapiq.tree.interventional.computer as interventional_module

    monkeypatch.setattr(interventional_module, "_STRUCTURAL_MAX_ROWS", 0)
    tree, x = _perfect_tree_with_distinct_features(depth=8)
    data = np.random.default_rng(1).normal(size=(6, 255))
    flat = InterventionalTreeSHAPIQ(tree, data, max_order=3, index="SII")
    fallback = InterventionalTreeSHAPIQ(tree, data, max_order=8, index="SII")
    assert flat._use_sparse_path
    assert fallback._use_sparse_path

    from_flat = flat.explain(x).dict_values
    from_fallback = fallback.explain(x).dict_values

    shared = {key for key in from_flat if 0 < len(key) <= 3}
    assert len(shared) > 100  # precondition: the comparison covers real interactions
    assert any(len(key) == 3 and key[0] == 0 for key in shared)  # ... with a leading feature 0
    assert {key for key in from_fallback if 0 < len(key) <= 3} == shared
    for interaction in shared:
        assert from_fallback[interaction] == pytest.approx(from_flat[interaction], abs=1e-12)


@pytest.mark.parametrize(("index", "order"), [("SV", 1), ("SII", 2)])
def test_interventional_float64_point_matches_model(index, order):
    """A float64 explain point must route identically to the float32 model it explains.

    Regression test for a precision bug in the dense E/R path: tree thresholds and
    the reference data are float32 (XGBoost evaluates splits in float32), but the
    explain point was routed at its incoming float64 precision. A feature value
    lying between the float32 and float64 representation of a split threshold then
    flipped the routing, sending the explanation to the wrong leaf and disagreeing
    with the true model (and the exact interventional game) by a whole leaf value.
    """
    xgboost = pytest.importorskip("xgboost")
    from sklearn.datasets import make_regression

    # Deep ensemble so root->leaf paths split the same feature repeatedly and
    # produce thresholds prone to the float32/float64 boundary collision.
    X, y = make_regression(n_samples=300, n_features=5, noise=0.1, random_state=2)
    model = xgboost.XGBRegressor(n_estimators=10, max_depth=6, random_state=0).fit(X, y)

    X_train = X[:25]
    # float64 explain point that lands on the wrong leaf pre-fix.
    point_to_explain = X[250].astype(np.float64)
    assert point_to_explain.dtype == np.float64

    explainer = InterventionalTreeSHAPIQ(model, X_train, index=index, max_order=order)
    own_interactions = explainer.explain_function(point_to_explain).interactions

    game = InterventionalGame(model, X_train, point_to_explain)
    exact_values = ExactComputer(game)(index, order)
    game_interactions = exact_values.interactions

    for interaction in own_interactions:
        if len(interaction) == 0:
            continue
        assert np.isclose(
            own_interactions[interaction],
            game_interactions.get(interaction, 0),
            atol=1e-4,
        )
