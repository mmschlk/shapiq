"""Tests for the compiled interventional tree kernel."""

from __future__ import annotations

from itertools import combinations

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    BII,
    SII,
    SV,
    CoMoebius,
    InterventionalTreeGame,
    Moebius,
    TreeExplainer,
    TreeModel,
    WeightedBII,
    to_tree_model,
)
from shapiq.explainers import _tree

N_PLAYERS = 6


def test_the_extension_is_built():
    # dev and CI environments must compile the kernel; pure-python installs
    # fall back silently, but here a missing build is a broken build
    assert _tree._cext_accumulate is not None


def random_forest_game(seed=0):
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415 - requires sklearn

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(300, N_PLAYERS))
    labels = (
        features[:, 0] * features[:, 1]
        - 2.0 * features[:, 2]
        + features[:, 3] * features[:, 4] * features[:, 5]
    )
    model = RandomForestRegressor(n_estimators=8, max_depth=5, random_state=seed).fit(
        features, labels
    )
    return InterventionalTreeGame(
        to_tree_model(model),
        inputs=features[0],
        baseline=features.mean(axis=0),
    )


CARDINAL_INDICES = [
    SV(),
    SII(order=2),
    BII(order=3),
    WeightedBII(p=0.3, order=2),
    Moebius(order=4),
    CoMoebius(order=3),
]


@pytest.mark.parametrize("index", CARDINAL_INDICES, ids=repr)
def test_kernel_and_python_paths_agree_exactly(index, monkeypatch):
    game = random_forest_game()
    order = game.n_players if index.order is None else index.order
    if order > _tree._CEXT_MAX_ORDER:
        pytest.skip("the kernel serves orders up to four")
    with_kernel = TreeExplainer(game, index).explain()
    monkeypatch.setattr(_tree, "_cext_accumulate", None)
    pure_python = TreeExplainer(game, index).explain()
    assert set(with_kernel.attributions) == set(pure_python.attributions)
    for interaction, total in pure_python.attributions.items():
        assert jnp.allclose(with_kernel.attributions[interaction], total, rtol=1e-6, atol=1e-7)


def test_moebius_of_deep_supports_falls_back_to_python(monkeypatch):
    calls = []
    original = _tree._accumulate_python

    def spying_python(*args, **kwargs):
        calls.append("python")
        return original(*args, **kwargs)

    monkeypatch.setattr(_tree, "_accumulate_python", spying_python)
    game = random_forest_game()
    # Moebius resolves to order n_players > 4, beyond the packing limit
    TreeExplainer(game, Moebius()).explain()
    assert calls == ["python"]


def test_vector_leaf_values_run_through_the_kernel(monkeypatch):
    two_class = TreeModel(
        children_left=[1, -1, -1],
        children_right=[2, -1, -1],
        features=[0, -2, -2],
        thresholds=[0.5, np.nan, np.nan],
        values=[[0.0, 0.0], [0.8, 0.2], [0.1, 0.9]],
    )
    game = InterventionalTreeGame(
        two_class,
        inputs=np.ones(N_PLAYERS),
        baseline=np.zeros(N_PLAYERS),
    )
    assert _tree._use_cext(game, order=1)
    with_kernel = TreeExplainer(game, SV()).explain()
    assert with_kernel((0,)).shape == (2,)
    monkeypatch.setattr(_tree, "_cext_accumulate", None)
    pure_python = TreeExplainer(game, SV()).explain()
    for player in range(N_PLAYERS):
        assert jnp.allclose(with_kernel((player,)), pure_python((player,)), atol=1e-7)


def test_multiclass_forests_run_through_the_kernel(monkeypatch):
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415 - requires sklearn

    rng = np.random.default_rng(4)
    features = rng.normal(size=(300, N_PLAYERS))
    labels = np.digitize(features[:, 0] * features[:, 1] + features[:, 2], [-0.5, 0.5])
    model = RandomForestClassifier(n_estimators=6, max_depth=4, random_state=0)
    model.fit(features, labels)
    game = InterventionalTreeGame(
        to_tree_model(model), inputs=features[0], baseline=features.mean(axis=0)
    )
    assert game.value_shape == (3,)
    assert _tree._use_cext(game, order=2)
    with_kernel = TreeExplainer(game, SII(order=2)).explain()
    monkeypatch.setattr(_tree, "_cext_accumulate", None)
    pure_python = TreeExplainer(game, SII(order=2)).explain()
    assert set(with_kernel.attributions) == set(pure_python.attributions)
    for interaction, total in pure_python.attributions.items():
        assert jnp.allclose(with_kernel.attributions[interaction], total, rtol=1e-6, atol=1e-7)


def test_kernel_handles_the_empty_interaction():
    from shapiq import DenseCoalitionArray  # noqa: PLC0415

    game = random_forest_game(seed=3)
    explanation = TreeExplainer(game, CoMoebius(order=2)).explain()
    ends = game(
        DenseCoalitionArray(jnp.asarray([[False] * N_PLAYERS, [True] * N_PLAYERS])),
    )
    # the co-Moebius order-0 attribution is the grand total on the centered game
    assert jnp.allclose(explanation(()), ends[1] - ends[0], atol=1e-4)


def test_kernel_serves_the_shipped_parity_suite():
    # the closed form (kernel path) must match brute force on the forest
    from shapiq import ExactExplainer  # noqa: PLC0415

    game = random_forest_game(seed=1)
    closed_form = TreeExplainer(game, SII(order=2)).explain()
    exact = ExactExplainer(game, SII(order=2)).explain()
    for size in (1, 2):
        for interaction in combinations(range(N_PLAYERS), size):
            assert jnp.allclose(closed_form(interaction), exact(interaction), atol=1e-4)


def test_table_extents_from_different_leaves_do_not_crash():
    # regression: max_present and max_absent come from different leaves; the
    # infeasible cross pairs must not be evaluated when filling the table
    def right_chain(depth):
        n_nodes = 2 * depth + 1
        left = [-1] * n_nodes
        right = [-1] * n_nodes
        feats = [-2] * n_nodes
        thresholds = [np.nan] * n_nodes
        node = 0
        for level in range(depth):
            left[node] = node + 1
            right[node] = node + 2
            feats[node] = level
            thresholds[node] = 0.5
            node += 2
        values = [float(i) for i in range(n_nodes)]
        return TreeModel(
            children_left=left,
            children_right=right,
            features=feats,
            thresholds=thresholds,
            values=values,
        )

    def left_chain(depth):
        chain = right_chain(depth)
        # swap the branches: the chain continues on the baseline's side, so
        # deep leaves accumulate many ABSENT constraints instead
        return TreeModel(
            children_left=chain.children_right,
            children_right=chain.children_left,
            features=chain.features,
            thresholds=chain.thresholds,
            values=chain.values,
        )

    game = InterventionalTreeGame(
        [right_chain(5), left_chain(5)],
        inputs=np.ones(N_PLAYERS),  # routes right: right_chain leaves need many present
        baseline=np.zeros(N_PLAYERS),  # routes left: left_chain leaves need many absent
    )
    tables = game.leaf_constraints
    max_present = max(int(t.present.sum(axis=1).max()) for t in tables)
    max_absent = max(int(t.absent.sum(axis=1).max()) for t in tables)
    assert max_present + max_absent > N_PLAYERS  # the trap is armed
    from shapiq import ExactExplainer  # noqa: PLC0415

    closed_form = TreeExplainer(game, SV()).explain()
    exact = ExactExplainer(game, SV()).explain()
    for player in range(N_PLAYERS):
        assert jnp.allclose(closed_form((player,)), exact((player,)), atol=1e-4)


def test_kernel_validates_inconsistent_buffers():
    kernel = _tree._cext_accumulate
    assert kernel is not None
    offsets = np.asarray([0, 2], dtype=np.int64)
    small = np.zeros(1, dtype=np.int64)
    values = np.zeros(1, dtype=np.float64)
    table = np.zeros((3, 3, 2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="shorter than the offsets claim"):
        # the leaf claims two members but the buffer holds one
        kernel(offsets, small, np.asarray([0, 0], dtype=np.int64), small, values, table, 2, 2, 1, 1, 1)
    zero_offsets = np.asarray([0, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="non-decreasing"):
        kernel(
            np.asarray([0, -1], dtype=np.int64),
            small,
            zero_offsets,
            small,
            values,
            table,
            2,
            2,
            1,
            1,
            1,
        )
    with pytest.raises(ValueError, match="declared coefficient-table extents"):
        small_table = np.zeros((2, 2, 2, 2), dtype=np.float64)
        kernel(
            np.asarray([0, 2], dtype=np.int64),
            np.zeros(2, dtype=np.int64),
            zero_offsets,
            small,
            values,
            small_table,
            1,
            1,
            1,
            1,
            1,
        )
    with pytest.raises(ValueError, match="16-bit packing"):
        kernel(
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([70000], dtype=np.int64),
            zero_offsets,
            small,
            values,
            table,
            2,
            2,
            1,
            1,
            1,
        )
    with pytest.raises(ValueError, match="at most four players"):
        kernel(zero_offsets, small, zero_offsets, small, values, table, 2, 2, 1, 5, 1)
    with pytest.raises(ValueError, match="disagree on the leaf count"):
        kernel(zero_offsets, small, np.asarray([0], dtype=np.int64), small, values, table, 2, 2, 1, 1, 1)
    with pytest.raises(ValueError, match="width must be positive"):
        kernel(zero_offsets, small, zero_offsets, small, values, table, 2, 2, 1, 1, 0)
    with pytest.raises(ValueError, match="disagree on the leaf count"):
        # one leaf of width two needs two doubles, not one
        kernel(
            np.asarray([0, 0], dtype=np.int64),
            small,
            np.asarray([0, 0], dtype=np.int64),
            small,
            values,
            table,
            2,
            2,
            1,
            1,
            2,
        )


def test_single_node_trees_and_identical_points_stay_serviceable():
    from shapiq import ExactExplainer  # noqa: PLC0415

    lone_leaf = TreeModel(
        children_left=[-1],
        children_right=[-1],
        features=[-2],
        thresholds=[np.nan],
        values=[3.5],
    )
    game = InterventionalTreeGame(lone_leaf, inputs=np.ones(N_PLAYERS), baseline=np.zeros(N_PLAYERS))
    explanation = TreeExplainer(game, SV()).explain()
    assert jnp.allclose(explanation.baseline, 3.5)
    assert jnp.allclose(explanation((0,)), 0.0)
    # identical points: every split routes both ways the same, nothing is constrained
    identical = InterventionalTreeGame(
        random_forest_game().trees,
        inputs=np.zeros(N_PLAYERS),
        baseline=np.zeros(N_PLAYERS),
    )
    closed_form = TreeExplainer(identical, SV()).explain()
    exact = ExactExplainer(identical, SV()).explain()
    for player in range(N_PLAYERS):
        assert jnp.allclose(closed_form((player,)), exact((player,)), atol=1e-5)


def test_sparse_weight_indices_survive_large_player_counts():
    # comb-to-float overflow near 1023 players is skipped for zero weights,
    # so the Moebius family stays serviceable at any size
    wide = TreeModel(
        children_left=[1, -1, -1],
        children_right=[2, -1, -1],
        features=[700, -2, -2],
        thresholds=[0.5, np.nan, np.nan],
        values=[0.0, 1.0, 4.0],
    )
    game = InterventionalTreeGame(wide, inputs=np.ones(1200), baseline=np.zeros(1200))
    explanation = TreeExplainer(game, Moebius(order=2)).explain()
    assert jnp.allclose(explanation((700,)), 3.0, atol=1e-5)  # m({700}) = 4 - 1
