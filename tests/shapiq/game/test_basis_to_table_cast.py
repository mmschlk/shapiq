"""Tests for fast basis-to-TableGame casters (MoebiusGame, AdditiveMoebiusGame, FourierGame)."""

import jax
import jax.numpy as jnp
import pytest
from flextype import CastUnavailable, try_cast

from shapiq.game import (
    TableGame,
    moebius as moebius_module,
)
from shapiq.game.discretederivative import subset_patterns
from shapiq.game.exact import ExactComputer
from shapiq.game.finite import FinitePlayerGame
from shapiq.game.fourier import FourierGame
from shapiq.game.moebius import AdditiveMoebiusGame, MoebiusGame
from shapiq.players import FinitePlayers
from shapiq.value import VectorValue


def _eval_reference_table(game: FinitePlayerGame) -> jax.Array:
    """Materialize expected coalition values in TableGame order directly via ``eval``."""
    n_players = game.n_players
    coalitions = subset_patterns(n_players).reshape(2**n_players, *(1,) * game.ndim, n_players)
    return jnp.moveaxis(game(coalitions), 0, -1)


@pytest.mark.parametrize("n_players", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("game_cls", [MoebiusGame, FourierGame])
def test_basis_game_to_table_matches_eval_all_subsets_and_asymmetric_weights(
    n_players: int,
    game_cls: type[MoebiusGame] | type[FourierGame],
):
    players = FinitePlayers(n_players)
    all_sets = subset_patterns(n_players)
    # Asymmetric per-interaction weights catch any bit-order or endianness mismatch.
    key = jax.random.key(100 + n_players)
    values = jax.random.normal(key, (all_sets.shape[0],), dtype=jnp.float32)
    game = game_cls(
        values=values,
        interactions=all_sets,
        players=players,
        additivity=n_players,
    )

    expected = _eval_reference_table(game)
    table = TableGame.from_game(game)

    assert isinstance(table, TableGame)
    assert table.shape == ()
    assert table.players is players
    assert table.values.dtype == jnp.float32
    assert jnp.allclose(table.values, expected, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(table(all_sets), game(all_sets), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("n_players", [0, 1, 2, 3, 4, 5])
def test_additive_moebius_to_table_matches_eval_and_uses_specialized_path(
    n_players: int,
    monkeypatch: pytest.MonkeyPatch,
):
    players = FinitePlayers(n_players)
    all_sets = subset_patterns(n_players)
    # Include empty set plus permuted singletons (when n_players > 0).
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

    additive = AdditiveMoebiusGame(values=values, interactions=interactions, players=players)
    expected = _eval_reference_table(additive)

    def fail_zeta(*args, **kwargs):
        msg = "AdditiveMoebiusGame -> TableGame should use the O(2**n) specialization."
        raise AssertionError(msg)

    def fail_eval(*args, **kwargs):
        msg = "Fast AdditiveMoebiusGame -> TableGame cast must not call eval()."
        raise AssertionError(msg)

    with monkeypatch.context() as mp:
        mp.setattr(moebius_module, "_fast_zeta_transform", fail_zeta)
        mp.setattr(FinitePlayerGame, "eval", fail_eval)
        table = TableGame.from_game(additive)

    assert isinstance(table, TableGame)
    assert table.players is players
    assert jnp.allclose(table.values, expected, atol=1e-5)
    assert jnp.allclose(table(all_sets), additive(all_sets), atol=1e-5)


@pytest.mark.parametrize("game_cls", [MoebiusGame, FourierGame, AdditiveMoebiusGame])
def test_zero_additivity_and_empty_interactions_and_empty_batches(game_cls):
    n_players = 3
    players = FinitePlayers(n_players)

    # Constant (0-additive) game with nonzero empty coefficient(s).
    const_interactions = jnp.zeros((2, n_players), dtype=bool)
    const_values = jnp.array([[2.5, -0.5], [1.0, 3.0]], dtype=jnp.float32)
    const_game = game_cls(
        values=const_values,
        interactions=const_interactions,
        players=players,
        additivity=0,
    )
    const_table = TableGame.from_game(const_game)
    assert jnp.allclose(const_table.values, _eval_reference_table(const_game))
    assert jnp.allclose(const_table.values, jnp.array([[2.0] * 8, [4.0] * 8], dtype=jnp.float32))

    # Empty n_interactions == 0 (scalar and batched).
    for batch_shape in [(), (2, 3)]:
        empty_inter = game_cls(
            values=jnp.zeros((*batch_shape, 0), dtype=jnp.float32),
            interactions=jnp.zeros((0, n_players), dtype=bool),
            players=players,
            additivity=0,
        )
        empty_table = TableGame.from_game(empty_inter)
        assert empty_table.shape == batch_shape
        assert empty_table.values.shape == (*batch_shape, 2**n_players)
        assert empty_table.values.dtype == jnp.float32
        assert jnp.array_equal(
            empty_table.values, jnp.zeros((*batch_shape, 2**n_players), dtype=jnp.float32)
        )

    # Empty batch dimensions (0 in game.shape).
    for batch_shape in [(0,), (2, 0)]:
        empty_batch = game_cls(
            values=jnp.zeros((*batch_shape, 1), dtype=jnp.float32),
            interactions=jnp.zeros((1, n_players), dtype=bool),
            players=players,
            additivity=0,
        )
        empty_batch_table = TableGame.from_game(empty_batch)
        assert empty_batch_table.shape == batch_shape
        assert empty_batch_table.values.shape == (*batch_shape, 2**n_players)
        assert empty_batch_table.values.dtype == jnp.float32


@pytest.mark.parametrize("game_cls", [MoebiusGame, FourierGame, AdditiveMoebiusGame])
def test_zero_player_basis_games_to_table(game_cls):
    players = FinitePlayers(0)

    # Duplicate empty interactions must sum into the single coalition entry.
    dup_game = game_cls(
        values=jnp.array([[1.5, 2.5, -1.0], [4.0, -0.5, 0.25]], dtype=jnp.float32),
        interactions=jnp.zeros((3, 0), dtype=bool),
        players=players,
        additivity=0,
    )
    dup_table = TableGame.from_game(dup_game)
    assert dup_table.shape == (2,)
    assert dup_table.values.shape == (2, 1)
    assert jnp.allclose(dup_table.values, _eval_reference_table(dup_game))
    assert jnp.allclose(dup_table.values, jnp.array([[3.0], [3.75]], dtype=jnp.float32))

    # Zero players and zero interactions.
    empty_game = game_cls(
        values=jnp.zeros((0,), dtype=jnp.float32),
        interactions=jnp.zeros((0, 0), dtype=bool),
        players=players,
        additivity=0,
    )
    empty_table = TableGame.from_game(empty_game)
    assert empty_table.shape == ()
    assert empty_table.values.shape == (1,)
    assert jnp.array_equal(empty_table.values, jnp.array([0.0], dtype=jnp.float32))


@pytest.mark.parametrize("game_cls", [MoebiusGame, FourierGame])
def test_batched_broadcasted_interactions_players_and_duplicates(game_cls):
    n_players = 3
    # Partially broadcasted players (1, 3) with distinct subsets from 4 latent players.
    players = FinitePlayers(
        jnp.array([[[0, 1, 2], [0, 1, 3], [1, 2, 3]]]),
        latent_n_players=4,
    )
    # Partially broadcasted interactions (2, 1, 5, 3) with unordered and duplicate sets.
    interactions = jnp.array(
        [
            [
                [
                    [1, 0, 1],
                    [0, 0, 0],
                    [1, 0, 1],  # duplicate
                    [0, 1, 0],
                    [0, 0, 0],  # duplicate empty
                ]
            ],
            [
                [
                    [1, 1, 0],
                    [0, 0, 1],
                    [0, 0, 1],  # duplicate
                    [1, 1, 1],
                    [1, 0, 0],
                ]
            ],
        ],
        dtype=bool,
    )
    values = jax.random.normal(jax.random.key(7), (2, 3, 5), dtype=jnp.float32)

    # Test shared-shape (1, 1, 5, 3), partially broadcasted (2, 1, 5, 3), and full (2, 3, 5, 3).
    cases = [
        interactions[:1],
        interactions,
        jnp.broadcast_to(interactions, (2, 3, 5, n_players)),
    ]
    for inter in cases:
        game = game_cls(values=values, interactions=inter, players=players, additivity=n_players)
        expected = _eval_reference_table(game)
        table = TableGame.from_game(game)
        assert table.shape == (2, 3)
        assert table.players is players
        assert jnp.allclose(table.values, expected, atol=1e-5, rtol=1e-5)


def test_additive_moebius_batched_broadcasted_and_duplicate_interactions():
    players = FinitePlayers(jnp.array([[0, 1, 2], [0, 2, 3]]), latent_n_players=4)  # shape (2,)
    # Broadcasted interactions (2, 1, 5, 3) with duplicate empty and duplicate singletons.
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
    game = AdditiveMoebiusGame(values=values, interactions=interactions, players=players)
    expected = _eval_reference_table(game)
    table = TableGame.from_game(game)
    assert table.shape == (2, 2)
    assert table.players is players
    assert jnp.allclose(table.values, expected, atol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_basis_to_table_preserves_floating_dtypes(dtype):
    players = FinitePlayers(3)
    inter = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=bool)
    vals = jnp.array([1.0, 2.0, -0.5], dtype=dtype)
    for game in (
        MoebiusGame(values=vals, interactions=inter, players=players, additivity=1),
        AdditiveMoebiusGame(values=vals, interactions=inter, players=players),
        FourierGame(values=vals, interactions=inter, players=players, additivity=1),
    ):
        table = TableGame.from_game(game)
        assert table.values.dtype == dtype
        assert jnp.allclose(table.values, _eval_reference_table(game))


def test_basis_to_table_works_under_jax_jit():
    players = FinitePlayers(3)
    inter_batched = jnp.array(
        [
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 0]],
        ],
        dtype=bool,
    )
    vals = jnp.array([[1.0, 2.0, -1.5, 0.5], [3.0, -2.0, 4.0, 1.0]], dtype=jnp.float32)

    mg = MoebiusGame(values=vals, interactions=inter_batched, players=players, additivity=2)
    fg = FourierGame(values=vals, interactions=inter_batched, players=players, additivity=2)
    am = AdditiveMoebiusGame(
        values=jnp.array([[1.0, 2.0, -0.5], [3.0, -1.0, 4.0]], dtype=jnp.float32),
        interactions=jnp.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=bool),
        players=players,
    )
    am0 = AdditiveMoebiusGame(
        values=jnp.array([[2.0, 3.0]], dtype=jnp.float32),
        interactions=jnp.zeros((2, 3), dtype=bool),
        players=players,
        additivity=0,
    )
    zero_p = MoebiusGame(
        values=jnp.array([1.0, 2.0], dtype=jnp.float32),
        interactions=jnp.zeros((2, 0), dtype=bool),
        players=FinitePlayers(0),
        additivity=0,
    )

    jit_cast = jax.jit(TableGame.from_game)
    for game in (mg, fg, am, am0, zero_p):
        eager = TableGame.from_game(game)
        jitted = jit_cast(game)
        assert isinstance(jitted, TableGame)
        assert jnp.allclose(jitted.values, eager.values, atol=1e-5)
        assert jnp.allclose(jitted.values, _eval_reference_table(game), atol=1e-5)


def test_basis_to_table_cast_unavailable_for_vector_and_non_floating_values():
    players = FinitePlayers(2)
    inter = jnp.array([[0, 0], [1, 0]], dtype=bool)

    vec = VectorValue(jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32))
    for vec_game in (
        MoebiusGame(values=vec, interactions=inter, players=players, additivity=1),
        AdditiveMoebiusGame(values=vec, interactions=inter, players=players),
        FourierGame(values=vec, interactions=inter, players=players, additivity=1),
    ):
        assert try_cast(vec_game, TableGame, max_cost=10.0) is None
        with pytest.raises(CastUnavailable):
            TableGame.from_game(vec_game)

    # Non-floating jax.Array values (bypassing runtime Float annotation via pytree unflatten),
    # including MoebiusGame with additivity 0 and 1 so the AdditiveGame route is also exercised.
    for base_game in (
        MoebiusGame(
            values=jnp.array([1.0, 2.0], dtype=jnp.float32),
            interactions=jnp.zeros((2, 2), dtype=bool),
            players=players,
            additivity=0,
        ),
        MoebiusGame(
            values=jnp.array([1.0, 2.0], dtype=jnp.float32),
            interactions=inter,
            players=players,
            additivity=1,
        ),
        MoebiusGame(
            values=jnp.array([1.0, 2.0], dtype=jnp.float32),
            interactions=inter,
            players=players,
            additivity=2,
        ),
        AdditiveMoebiusGame(
            values=jnp.array([1.0, 2.0], dtype=jnp.float32),
            interactions=inter,
            players=players,
        ),
        FourierGame(
            values=jnp.array([1.0, 2.0], dtype=jnp.float32),
            interactions=inter,
            players=players,
            additivity=2,
        ),
    ):
        leaves, treedef = jax.tree_util.tree_flatten(base_game)
        # Replace the float values leaf with int32 while keeping interactions/players intact.
        int_leaves = [
            leaf.astype(jnp.int32) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf
            for leaf in leaves
        ]
        int_game = jax.tree_util.tree_unflatten(treedef, int_leaves)
        assert try_cast(int_game, TableGame, max_cost=10.0) is None
        assert try_cast(int_game, TableGame) is None
        with pytest.raises(CastUnavailable):
            TableGame.from_game(int_game)


def test_basis_to_table_low_precision_accumulation_avoids_overflow_and_scatter_loss():
    # Cancellation before float16 overflow: [40000, 40000, -40000] on {}, {0}, {1}.
    players_2 = FinitePlayers(2)
    inter_2 = jnp.array([[0, 0], [1, 0], [0, 1]], dtype=bool)
    vals_f16 = jnp.array([40000.0, 40000.0, -40000.0], dtype=jnp.float16)

    for game in (
        MoebiusGame(values=vals_f16, interactions=inter_2, players=players_2, additivity=1),
        AdditiveMoebiusGame(values=vals_f16, interactions=inter_2, players=players_2),
    ):
        table = TableGame.from_game(game)
        assert table.values.dtype == jnp.float16
        # Full coalition {0, 1} must be 40000, not inf from intermediate 80000 overflow.
        assert jnp.array_equal(table.values[3], jnp.float16(40000.0))
        assert jnp.array_equal(table.values, _eval_reference_table(game))

    fg_f16 = FourierGame(values=vals_f16, interactions=inter_2, players=players_2, additivity=1)
    fg_table = TableGame.from_game(fg_f16)
    assert fg_table.values.dtype == jnp.float16
    # Empty coalition {} sums all three coefficients -> 40000, not inf.
    assert jnp.array_equal(fg_table.values[0], jnp.float16(40000.0))
    assert jnp.array_equal(fg_table.values, _eval_reference_table(fg_f16))

    # Many duplicate interactions must not saturate float16/bfloat16 accumulators.
    n_dups = 4096
    players_1 = FinitePlayers(1)
    dup_singleton = jnp.ones((n_dups, 1), dtype=bool)
    dup_empty = jnp.zeros((n_dups, 1), dtype=bool)
    for dtype in (jnp.float16, jnp.bfloat16):
        ones = jnp.ones((n_dups,), dtype=dtype)
        for game, expected in (
            (
                MoebiusGame(
                    values=ones, interactions=dup_singleton, players=players_1, additivity=1
                ),
                jnp.array([0.0, float(n_dups)], dtype=dtype),
            ),
            (
                AdditiveMoebiusGame(values=ones, interactions=dup_singleton, players=players_1),
                jnp.array([0.0, float(n_dups)], dtype=dtype),
            ),
            (
                FourierGame(
                    values=ones, interactions=dup_singleton, players=players_1, additivity=1
                ),
                jnp.array([float(n_dups), -float(n_dups)], dtype=dtype),
            ),
            (
                MoebiusGame(values=ones, interactions=dup_empty, players=players_1, additivity=0),
                jnp.array([float(n_dups), float(n_dups)], dtype=dtype),
            ),
            (
                AdditiveMoebiusGame(
                    values=ones, interactions=dup_empty, players=players_1, additivity=0
                ),
                jnp.array([float(n_dups), float(n_dups)], dtype=dtype),
            ),
            (
                FourierGame(values=ones, interactions=dup_empty, players=players_1, additivity=0),
                jnp.array([float(n_dups), float(n_dups)], dtype=dtype),
            ),
        ):
            table = TableGame.from_game(game)
            assert table.values.dtype == dtype
            assert jnp.array_equal(table.values, expected)


def test_basis_to_table_index_width_guard_at_x32_boundary():
    with jax.enable_x64(new_val=False):
        players_31 = FinitePlayers(31)
        inter_31 = jnp.zeros((1, 31), dtype=bool)
        vals = jnp.ones((1,), dtype=jnp.float32)

        for game in (
            MoebiusGame(values=vals, interactions=inter_31, players=players_31, additivity=1),
            AdditiveMoebiusGame(values=vals, interactions=inter_31, players=players_31),
            FourierGame(values=vals, interactions=inter_31, players=players_31, additivity=1),
        ):
            with pytest.raises(ValueError, match="integer width"):
                jax.eval_shape(TableGame.from_game, game)

        players_30 = FinitePlayers(30)
        inter_30 = jnp.zeros((1, 30), dtype=bool)
        for game in (
            MoebiusGame(values=vals, interactions=inter_30, players=players_30, additivity=1),
            AdditiveMoebiusGame(values=vals, interactions=inter_30, players=players_30),
            FourierGame(values=vals, interactions=inter_30, players=players_30, additivity=1),
        ):
            abstract_table = jax.eval_shape(TableGame.from_game, game)
            assert abstract_table.values.shape == (1 << 30,)


def test_basis_to_table_cost_routing_and_no_eval_on_fast_path(monkeypatch: pytest.MonkeyPatch):
    players = FinitePlayers(3)
    inter = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=bool)
    vals = jnp.array([[1.0, 2.0, -1.0], [0.5, 1.5, 2.5]], dtype=jnp.float32)

    mg = MoebiusGame(values=vals, interactions=inter, players=players, additivity=2)
    am = AdditiveMoebiusGame(values=vals, interactions=inter, players=players)
    fg = FourierGame(values=vals, interactions=inter, players=players, additivity=2)
    tbl = TableGame(jnp.arange(16, dtype=jnp.float32).reshape(2, 8), players)

    expected_mg = _eval_reference_table(mg)
    expected_am = _eval_reference_table(am)
    expected_fg = _eval_reference_table(fg)

    def fail_eval(*args, **kwargs):
        msg = "Fast basis -> TableGame cast must not call eval()."
        raise AssertionError(msg)

    with monkeypatch.context() as mp:
        mp.setattr(FinitePlayerGame, "eval", fail_eval)

        # AdditiveMoebiusGame: cost 2.0 (> 1.0, < 5.0).
        assert try_cast(am, TableGame, max_cost=1.0) is None
        assert try_cast(am, TableGame, max_cost=1.9) is None
        am_res = try_cast(am, TableGame, max_cost=2.0)
        assert am_res is not None
        assert isinstance(am_res.value, TableGame)
        assert jnp.allclose(am_res.value.values, expected_am)
        assert jnp.allclose(ExactComputer.from_game(am).table.values, expected_am)

        # MoebiusGame and FourierGame: cost 5.0 (> 1.0, << 1000.0).
        for game, expected in ((mg, expected_mg), (fg, expected_fg)):
            assert try_cast(game, TableGame, max_cost=1.0) is None
            assert try_cast(game, TableGame, max_cost=4.9) is None
            res = try_cast(game, TableGame, max_cost=5.0)
            assert res is not None
            assert isinstance(res.value, TableGame)
            assert jnp.allclose(res.value.values, expected)
            assert jnp.allclose(TableGame.from_game(game).values, expected)
            assert jnp.allclose(ExactComputer.from_game(game).table.values, expected)

    # ArithmeticGame and ReductionGame keep their TableGame-leaf-only max_cost=1.0 contract.
    for expr in (
        tbl + mg,
        tbl + am,
        tbl + fg,
        mg.sum(axis=0),
        am.sum(axis=0),
        fg.sum(axis=0),
    ):
        assert try_cast(expr, TableGame, max_cost=1.0) is None
        assert try_cast(expr, TableGame, max_cost=999.0) is None
        fallback = try_cast(expr, TableGame, max_cost=1000.0)
        assert fallback is not None
        assert isinstance(fallback.value, TableGame)
        assert jnp.allclose(fallback.value.values, _eval_reference_table(expr), atol=1e-5)
