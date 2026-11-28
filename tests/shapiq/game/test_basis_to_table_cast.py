"""Tests for fast basis/layout casters across all Dense and Sparse materialized games."""

import jax
import jax.numpy as jnp
import pytest
from flextype import (
    CastUnavailable,
    cast as flex_cast,
    try_cast,
)

from shapiq.game import (
    DenseCoalitionGame,
    DenseFourierGame,
    DenseMoebiusGame,
    FinitePlayerGame,
    SparseCoalitionGame,
    SparseFourierGame,
    SparseMoebiusGame,
    dense as dense_module,
    subset_patterns,
)
from shapiq.index import ExactComputer
from shapiq.players import FinitePlayers
from shapiq.value import VectorValue


def _eval_reference_table(game: FinitePlayerGame) -> jax.Array:
    """Materialize expected coalition values in DenseCoalitionGame order directly via ``eval``."""
    n_players = game.n_players
    coalitions = subset_patterns(n_players).reshape(2**n_players, *(1,) * game.ndim, n_players)
    return jnp.moveaxis(game(coalitions), 0, -1)


@pytest.mark.parametrize("n_players", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("game_cls", [SparseMoebiusGame, SparseFourierGame, SparseCoalitionGame])
def test_sparse_game_to_dense_coalition_matches_eval_all_subsets(n_players: int, game_cls):
    players = FinitePlayers(n_players)
    all_sets = subset_patterns(n_players)
    key = jax.random.key(100 + n_players)
    values = jax.random.normal(key, (all_sets.shape[0],), dtype=jnp.float32)
    if game_cls is SparseCoalitionGame:
        game = game_cls(values=values, interactions=all_sets, players=players)
    else:
        game = game_cls(values=values, interactions=all_sets, players=players, additivity=n_players)

    expected = _eval_reference_table(game)
    table = DenseCoalitionGame.from_game(game)

    assert isinstance(table, DenseCoalitionGame)
    assert table.shape == ()
    assert table.players is players
    assert table.values.dtype == jnp.float32
    assert jnp.allclose(table.values, expected, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(table(all_sets), game(all_sets), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("n_players", [0, 1, 2, 3, 4])
def test_all_six_materialized_types_round_trip_every_pair(n_players: int):
    players = FinitePlayers(n_players)
    all_sets = subset_patterns(n_players)
    key = jax.random.key(200 + n_players)
    raw = jax.random.normal(key, (2, all_sets.shape[0]), dtype=jnp.float32)
    base = DenseCoalitionGame(raw, players)

    all_classes = (
        DenseCoalitionGame,
        DenseMoebiusGame,
        DenseFourierGame,
        SparseCoalitionGame,
        SparseMoebiusGame,
        SparseFourierGame,
    )
    instances = {cls: cls.from_game(base) for cls in all_classes}
    for src_cls, src in instances.items():
        assert isinstance(src, src_cls)
        assert jnp.allclose(src(all_sets[:, None]), base(all_sets[:, None]), atol=1e-4)
        for dst_cls in all_classes:
            converted = flex_cast(src, dst_cls)
            assert isinstance(converted, dst_cls)
            assert jnp.allclose(converted(all_sets[:, None]), base(all_sets[:, None]), atol=1e-4)
            assert jnp.allclose(
                converted.moebius_coefficients(all_sets[:, None]),
                base.moebius_coefficients(all_sets[:, None]),
                atol=1e-4,
            )
            assert jnp.allclose(
                converted.fourier_coefficients(all_sets[:, None]),
                base.fourier_coefficients(all_sets[:, None]),
                atol=1e-4,
            )


@pytest.mark.parametrize("n_players", [0, 1, 2, 3, 4, 5])
def test_one_additive_sparse_moebius_to_dense_uses_specialized_path(
    n_players: int,
    monkeypatch: pytest.MonkeyPatch,
):
    players = FinitePlayers(n_players)
    all_sets = subset_patterns(n_players)
    if n_players == 0:
        interactions = jnp.zeros((1, 0), dtype=bool)
        values = jnp.array([3.25], dtype=jnp.float32)
    else:
        perm = jnp.arange(n_players - 1, -1, -1)
        singletons = jnp.eye(n_players, dtype=bool)[perm]
        interactions = jnp.concatenate([jnp.zeros((1, n_players), dtype=bool), singletons], axis=0)
        values = jnp.arange(1, n_players + 2, dtype=jnp.float32) * jnp.where(
            jnp.arange(n_players + 1) % 2 == 0, 1.0, -0.5
        )

    additive = SparseMoebiusGame(
        values=values, interactions=interactions, players=players, additivity=1
    )
    expected = _eval_reference_table(additive)

    def fail_zeta(*args, **kwargs):
        msg = "1-additive SparseMoebiusGame -> DenseCoalitionGame should use O(2**n) doubling."
        raise AssertionError(msg)

    def fail_eval(*args, **kwargs):
        msg = "Fast SparseMoebiusGame -> DenseCoalitionGame cast must not call eval()."
        raise AssertionError(msg)

    with monkeypatch.context() as mp:
        mp.setattr(dense_module, "_fast_zeta_transform", fail_zeta)
        mp.setattr(FinitePlayerGame, "eval", fail_eval)
        table = DenseCoalitionGame.from_game(additive)

    assert isinstance(table, DenseCoalitionGame)
    assert table.players is players
    assert jnp.allclose(table.values, expected, atol=1e-5)
    assert jnp.allclose(table(all_sets), additive(all_sets), atol=1e-5)


@pytest.mark.parametrize("game_cls", [SparseMoebiusGame, SparseFourierGame])
def test_zero_additivity_and_empty_interactions_and_empty_batches(game_cls):
    n_players = 3
    players = FinitePlayers(n_players)

    const_interactions = jnp.zeros((2, n_players), dtype=bool)
    const_values = jnp.array([[2.5, -0.5], [1.0, 3.0]], dtype=jnp.float32)
    const_game = game_cls(
        values=const_values,
        interactions=const_interactions,
        players=players,
        additivity=0,
    )
    const_table = DenseCoalitionGame.from_game(const_game)
    assert jnp.allclose(const_table.values, _eval_reference_table(const_game))
    assert jnp.allclose(const_table.values, jnp.array([[2.0] * 8, [4.0] * 8], dtype=jnp.float32))

    for batch_shape in [(), (2, 3)]:
        empty_inter = game_cls(
            values=jnp.zeros((*batch_shape, 0), dtype=jnp.float32),
            interactions=jnp.zeros((0, n_players), dtype=bool),
            players=players,
            additivity=0,
        )
        empty_table = DenseCoalitionGame.from_game(empty_inter)
        assert empty_table.shape == batch_shape
        assert empty_table.values.shape == (*batch_shape, 2**n_players)
        assert empty_table.values.dtype == jnp.float32
        assert jnp.array_equal(
            empty_table.values, jnp.zeros((*batch_shape, 2**n_players), dtype=jnp.float32)
        )

    for batch_shape in [(0,), (2, 0)]:
        empty_batch = game_cls(
            values=jnp.zeros((*batch_shape, 1), dtype=jnp.float32),
            interactions=jnp.zeros((1, n_players), dtype=bool),
            players=players,
            additivity=0,
        )
        empty_batch_table = DenseCoalitionGame.from_game(empty_batch)
        assert empty_batch_table.shape == batch_shape
        assert empty_batch_table.values.shape == (*batch_shape, 2**n_players)
        assert empty_batch_table.values.dtype == jnp.float32


@pytest.mark.parametrize("game_cls", [SparseMoebiusGame, SparseFourierGame])
def test_zero_player_basis_games_to_dense(game_cls):
    players = FinitePlayers(0)

    dup_game = game_cls(
        values=jnp.array([[1.5, 2.5, -1.0], [4.0, -0.5, 0.25]], dtype=jnp.float32),
        interactions=jnp.zeros((3, 0), dtype=bool),
        players=players,
        additivity=0,
    )
    dup_table = DenseCoalitionGame.from_game(dup_game)
    assert dup_table.shape == (2,)
    assert dup_table.values.shape == (2, 1)
    assert jnp.allclose(dup_table.values, _eval_reference_table(dup_game))
    assert jnp.allclose(dup_table.values, jnp.array([[3.0], [3.75]], dtype=jnp.float32))

    empty_game = game_cls(
        values=jnp.zeros((0,), dtype=jnp.float32),
        interactions=jnp.zeros((0, 0), dtype=bool),
        players=players,
        additivity=0,
    )
    empty_table = DenseCoalitionGame.from_game(empty_game)
    assert empty_table.shape == ()
    assert empty_table.values.shape == (1,)
    assert jnp.array_equal(empty_table.values, jnp.array([0.0], dtype=jnp.float32))


@pytest.mark.parametrize("game_cls", [SparseMoebiusGame, SparseFourierGame, SparseCoalitionGame])
def test_batched_broadcasted_interactions_players_and_duplicates(game_cls):
    n_players = 3
    players = FinitePlayers(
        jnp.array([[[0, 1, 2], [0, 1, 3], [1, 2, 3]]]),
        latent_n_players=4,
    )
    interactions = jnp.array(
        [
            [
                [
                    [1, 0, 1],
                    [0, 0, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                    [0, 0, 0],
                ]
            ],
            [
                [
                    [1, 1, 0],
                    [0, 0, 1],
                    [0, 0, 1],
                    [1, 1, 1],
                    [1, 0, 0],
                ]
            ],
        ],
        dtype=bool,
    )
    values = jax.random.normal(jax.random.key(7), (2, 3, 5), dtype=jnp.float32)

    cases = [
        interactions[:1],
        interactions,
        jnp.broadcast_to(interactions, (2, 3, 5, n_players)),
    ]
    for inter in cases:
        if game_cls is SparseCoalitionGame:
            game = game_cls(values=values, interactions=inter, players=players)
        else:
            game = game_cls(
                values=values, interactions=inter, players=players, additivity=n_players
            )
        expected = _eval_reference_table(game)
        table = DenseCoalitionGame.from_game(game)
        assert table.shape == (2, 3)
        assert table.players is players
        assert jnp.allclose(table.values, expected, atol=1e-5, rtol=1e-5)


def test_one_additive_moebius_batched_broadcasted_and_duplicate_interactions():
    players = FinitePlayers(jnp.array([[0, 1, 2], [0, 2, 3]]), latent_n_players=4)
    interactions = jnp.array(
        [
            [[[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0]]],
            [[[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0]]],
        ],
        dtype=bool,
    )
    values = jnp.array(
        [
            [[1.0, 2.0, -3.0, 0.5, 4.0], [-1.0, 0.25, 2.0, 1.75, 0.5]],
            [[3.0, -1.0, 2.5, 5.0, -2.0], [0.5, 1.5, -0.5, -2.0, 4.0]],
        ],
        dtype=jnp.float32,
    )
    game = SparseMoebiusGame(
        values=values, interactions=interactions, players=players, additivity=1
    )
    expected = _eval_reference_table(game)
    table = DenseCoalitionGame.from_game(game)
    assert table.shape == (2, 2)
    assert table.players is players
    assert jnp.allclose(table.values, expected, atol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_basis_to_dense_preserves_floating_dtypes(dtype):
    players = FinitePlayers(3)
    inter = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=bool)
    vals = jnp.array([1.0, 2.0, -0.5], dtype=dtype)
    for game in (
        SparseMoebiusGame(values=vals, interactions=inter, players=players, additivity=2),
        SparseMoebiusGame(values=vals, interactions=inter, players=players, additivity=1),
        SparseFourierGame(values=vals, interactions=inter, players=players, additivity=1),
        SparseCoalitionGame(values=vals, interactions=inter, players=players),
    ):
        table = DenseCoalitionGame.from_game(game)
        assert table.values.dtype == dtype
        assert jnp.allclose(table.values, _eval_reference_table(game))


def test_basis_to_dense_works_under_jax_jit():
    players = FinitePlayers(3)
    inter_batched = jnp.array(
        [
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 0]],
        ],
        dtype=bool,
    )
    vals = jnp.array([[1.0, 2.0, -1.5, 0.5], [3.0, -2.0, 4.0, 1.0]], dtype=jnp.float32)

    mg = SparseMoebiusGame(values=vals, interactions=inter_batched, players=players, additivity=2)
    fg = SparseFourierGame(values=vals, interactions=inter_batched, players=players, additivity=2)
    am = SparseMoebiusGame(
        values=jnp.array([[1.0, 2.0, -0.5], [3.0, -1.0, 4.0]], dtype=jnp.float32),
        interactions=jnp.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=bool),
        players=players,
        additivity=1,
    )
    am0 = SparseMoebiusGame(
        values=jnp.array([[2.0, 3.0]], dtype=jnp.float32),
        interactions=jnp.zeros((2, 3), dtype=bool),
        players=players,
        additivity=0,
    )
    zero_p = SparseMoebiusGame(
        values=jnp.array([1.0, 2.0], dtype=jnp.float32),
        interactions=jnp.zeros((2, 0), dtype=bool),
        players=FinitePlayers(0),
        additivity=0,
    )

    jit_cast = jax.jit(DenseCoalitionGame.from_game)
    for game in (mg, fg, am, am0, zero_p):
        eager = DenseCoalitionGame.from_game(game)
        jitted = jit_cast(game)
        assert isinstance(jitted, DenseCoalitionGame)
        assert jnp.allclose(jitted.values, eager.values, atol=1e-5)
        assert jnp.allclose(jitted.values, _eval_reference_table(game), atol=1e-5)


def test_basis_to_dense_cast_unavailable_for_vector_and_non_floating_values():
    players = FinitePlayers(2)
    inter = jnp.array([[0, 0], [1, 0]], dtype=bool)

    vec = VectorValue(jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32))
    for vec_game in (
        SparseMoebiusGame(values=vec, interactions=inter, players=players, additivity=1),
        SparseMoebiusGame(values=vec, interactions=inter, players=players, additivity=2),
        SparseFourierGame(values=vec, interactions=inter, players=players, additivity=1),
        SparseCoalitionGame(values=vec, interactions=inter, players=players),
    ):
        assert try_cast(vec_game, DenseCoalitionGame, max_cost=10.0) is None
        with pytest.raises(CastUnavailable):
            DenseCoalitionGame.from_game(vec_game)

    for base_game in (
        SparseMoebiusGame(
            values=jnp.array([1.0, 2.0], dtype=jnp.float32),
            interactions=jnp.zeros((2, 2), dtype=bool),
            players=players,
            additivity=0,
        ),
        SparseMoebiusGame(
            values=jnp.array([1.0, 2.0], dtype=jnp.float32),
            interactions=inter,
            players=players,
            additivity=1,
        ),
        SparseMoebiusGame(
            values=jnp.array([1.0, 2.0], dtype=jnp.float32),
            interactions=inter,
            players=players,
            additivity=2,
        ),
        SparseFourierGame(
            values=jnp.array([1.0, 2.0], dtype=jnp.float32),
            interactions=inter,
            players=players,
            additivity=2,
        ),
    ):
        leaves, treedef = jax.tree_util.tree_flatten(base_game)
        int_leaves = [
            leaf.astype(jnp.int32) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf
            for leaf in leaves
        ]
        int_game = jax.tree_util.tree_unflatten(treedef, int_leaves)
        assert try_cast(int_game, DenseCoalitionGame, max_cost=10.0) is None
        assert try_cast(int_game, DenseCoalitionGame) is None
        with pytest.raises(CastUnavailable):
            DenseCoalitionGame.from_game(int_game)


@pytest.mark.parametrize(
    ("game_cls", "additivity", "coalition_index"),
    [
        (SparseMoebiusGame, 2, 3),
        (SparseMoebiusGame, 1, 3),
        (SparseFourierGame, 1, 0),
    ],
)
def test_basis_to_dense_low_precision_transforms_need_an_explicit_wider_cast(
    game_cls, additivity, coalition_index
):
    players = FinitePlayers(2)
    interactions = jnp.array([[0, 0], [1, 0], [0, 1]], dtype=bool)
    values = jnp.array([40000.0, 40000.0, -40000.0], dtype=jnp.float16)
    game = game_cls(
        values=values, interactions=interactions, players=players, additivity=additivity
    )

    narrow = DenseCoalitionGame.from_game(game)
    wide = DenseCoalitionGame.from_game(game.astype(jnp.float32))
    assert narrow.dtype == jnp.float16
    assert jnp.isinf(narrow.values[coalition_index])
    assert wide.dtype == jnp.float32
    assert wide.values[coalition_index] == 40000.0
    assert jnp.array_equal(wide.values, _eval_reference_table(game.astype(jnp.float32)))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("game_cls", [SparseMoebiusGame, SparseFourierGame])
def test_duplicate_coefficients_scatter_in_the_storage_dtype(dtype, game_cls):
    n = 4096
    values = jnp.ones(n, dtype=dtype)
    interactions = jnp.ones((n, 1), dtype=bool)
    game = game_cls(
        values=values, interactions=interactions, players=FinitePlayers(1), additivity=1
    )

    singleton = jnp.zeros(2, dtype=dtype).at[jnp.ones(n, dtype=int)].add(values)[1]
    expected = (
        jnp.stack((0 * singleton, singleton))
        if game_cls is SparseMoebiusGame
        else jnp.stack((singleton, -singleton))
    )
    assert jnp.array_equal(DenseCoalitionGame.from_game(game).values, expected)
    wide = DenseCoalitionGame.from_game(game.astype(jnp.float32))
    assert wide.values[0 if game_cls is SparseFourierGame else 1] == n


def test_basis_to_dense_index_width_guard_at_x32_boundary():
    with jax.enable_x64(new_val=False):
        players_31 = FinitePlayers(31)
        inter_31 = jnp.zeros((1, 31), dtype=bool)
        vals = jnp.ones((1,), dtype=jnp.float32)

        for game in (
            SparseMoebiusGame(values=vals, interactions=inter_31, players=players_31, additivity=2),
            SparseMoebiusGame(values=vals, interactions=inter_31, players=players_31, additivity=1),
            SparseFourierGame(values=vals, interactions=inter_31, players=players_31, additivity=1),
        ):
            with pytest.raises(ValueError, match="integer width"):
                jax.eval_shape(DenseCoalitionGame.from_game, game)

        players_30 = FinitePlayers(30)
        inter_30 = jnp.zeros((1, 30), dtype=bool)
        for game in (
            SparseMoebiusGame(values=vals, interactions=inter_30, players=players_30, additivity=2),
            SparseMoebiusGame(values=vals, interactions=inter_30, players=players_30, additivity=1),
            SparseFourierGame(values=vals, interactions=inter_30, players=players_30, additivity=1),
        ):
            abstract_table = jax.eval_shape(DenseCoalitionGame.from_game, game)
            assert abstract_table.values.shape == (1 << 30,)


def test_basis_to_dense_cost_routing_and_no_eval_on_fast_path(monkeypatch: pytest.MonkeyPatch):
    players = FinitePlayers(3)
    inter = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=bool)
    vals = jnp.array([[1.0, 2.0, -1.0], [0.5, 1.5, 2.5]], dtype=jnp.float32)

    mg = SparseMoebiusGame(values=vals, interactions=inter, players=players, additivity=2)
    am = SparseMoebiusGame(values=vals, interactions=inter, players=players, additivity=1)
    fg = SparseFourierGame(values=vals, interactions=inter, players=players, additivity=2)
    tbl = DenseCoalitionGame(jnp.arange(16, dtype=jnp.float32).reshape(2, 8), players)

    expected_mg = _eval_reference_table(mg)
    expected_am = _eval_reference_table(am)
    expected_fg = _eval_reference_table(fg)

    def fail_eval(*args, **kwargs):
        msg = "Fast basis -> DenseCoalitionGame cast must not call eval()."
        raise AssertionError(msg)

    with monkeypatch.context() as mp:
        mp.setattr(FinitePlayerGame, "eval", fail_eval)

        # 1-additive SparseMoebiusGame: cost 2.0 (> 1.0, < 5.0).
        assert try_cast(am, DenseCoalitionGame, max_cost=1.0) is None
        assert try_cast(am, DenseCoalitionGame, max_cost=1.9) is None
        am_res = try_cast(am, DenseCoalitionGame, max_cost=2.0)
        assert am_res is not None
        assert isinstance(am_res.value, DenseCoalitionGame)
        assert jnp.allclose(am_res.value.values, expected_am)
        assert jnp.allclose(ExactComputer.from_game(am).table.values, expected_am)

        # SparseMoebiusGame (k>1) and SparseFourierGame: cost 5.0 (scatter 2 + transform 3).
        for game, expected in ((mg, expected_mg), (fg, expected_fg)):
            assert try_cast(game, DenseCoalitionGame, max_cost=1.0) is None
            assert try_cast(game, DenseCoalitionGame, max_cost=4.9) is None
            res = try_cast(game, DenseCoalitionGame, max_cost=5.0)
            assert res is not None
            assert isinstance(res.value, DenseCoalitionGame)
            assert jnp.allclose(res.value.values, expected)
            assert jnp.allclose(DenseCoalitionGame.from_game(game).values, expected)
            assert jnp.allclose(ExactComputer.from_game(game).table.values, expected)

    # ArithmeticGame and ReductionGame keep their DenseCoalitionGame-leaf-only max_cost=1.0 contract.
    for expr in (
        tbl + mg,
        tbl + am,
        tbl + fg,
        mg.sum(axis=0),
        am.sum(axis=0),
        fg.sum(axis=0),
    ):
        assert try_cast(expr, DenseCoalitionGame, max_cost=1.0) is None
        assert try_cast(expr, DenseCoalitionGame, max_cost=999.0) is None
        fallback = try_cast(expr, DenseCoalitionGame, max_cost=1000.0)
        assert fallback is not None
        assert isinstance(fallback.value, DenseCoalitionGame)
        assert jnp.allclose(fallback.value.values, _eval_reference_table(expr), atol=1e-5)
