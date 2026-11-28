"""Sparse and dense basis games reproduce the coalition table and cross-basis coefficients."""

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
import pytest

from shapiq.game import (
    Basis,
    CastGame,
    CoalitionDefinedGame,
    DenseCoalitionGame,
    DenseFourierGame,
    DenseMoebiusGame,
    FinitePlayerGame,
    KnownAdditivity,
    MoebiusDefinedGame,
    SparseCoalitionGame,
    SparseFourierGame,
    SparseMoebiusGame,
    subset_patterns,
)
from shapiq.players import FinitePlayers
from shapiq.value import VectorValue

N_PLAYERS = 3
SETS = subset_patterns(N_PLAYERS)
TABLES = jnp.array(
    [[0.0, 1.0, 2.0, 4.0, 3.0, 5.0, 6.0, 10.0], [1.0, -2.0, 0.5, 3.0, 2.0, 0.0, 7.0, -1.0]]
)


def _overlap(left: jax.Array, right: jax.Array) -> jax.Array:
    return jnp.sum(left[:, None, :] & right[None, :, :], axis=-1)


def _moebius(tables: jax.Array) -> SparseMoebiusGame:
    subset = ~jnp.any(SETS[None, :, :] & ~SETS[:, None, :], axis=-1)  # [T, S]: S ⊆ T
    sign = (-1.0) ** (jnp.sum(SETS, axis=-1)[:, None] - _overlap(SETS, SETS))
    return SparseMoebiusGame(
        values=tables @ (subset * sign).T,
        interactions=SETS,
        players=FinitePlayers(N_PLAYERS),
        additivity=N_PLAYERS,
    )


def _fourier(tables: jax.Array) -> SparseFourierGame:
    chi = (-1.0) ** _overlap(SETS, SETS)
    return SparseFourierGame(
        values=tables @ chi / 2**N_PLAYERS,
        interactions=SETS,
        players=FinitePlayers(N_PLAYERS),
        additivity=N_PLAYERS,
    )


def _sparse_coalition(tables: jax.Array) -> SparseCoalitionGame:
    return SparseCoalitionGame(
        values=tables,
        interactions=SETS,
        players=FinitePlayers(N_PLAYERS),
    )


@pytest.mark.parametrize(
    "build",
    [
        _moebius,
        _fourier,
        _sparse_coalition,
        lambda t: DenseMoebiusGame.from_game(DenseCoalitionGame(t, FinitePlayers(N_PLAYERS))),
        lambda t: DenseFourierGame.from_game(DenseCoalitionGame(t, FinitePlayers(N_PLAYERS))),
    ],
)
def test_basis_game_reproduces_all_three_bases_batched_indexed_and_jitted(build):
    game = build(TABLES)
    table = DenseCoalitionGame(TABLES, FinitePlayers(N_PLAYERS))
    ref_moebius = _moebius(TABLES)
    ref_fourier = _fourier(TABLES)
    expected = table(SETS[:, None])  # (coalitions, games)

    assert game.shape == (2,)
    assert jnp.allclose(game(SETS[:, None]), expected, atol=1e-5)
    assert jnp.allclose(game[1](SETS), expected[:, 1], atol=1e-5)
    assert jnp.allclose(jax.jit(lambda g, c: g(c))(game, SETS[:, None]), expected, atol=1e-5)
    assert jnp.allclose(
        game.moebius_coefficients(SETS[:, None]),
        ref_moebius.moebius_coefficients(SETS[:, None]),
        atol=1e-5,
    )
    assert jnp.allclose(
        game.fourier_coefficients(SETS[:, None]),
        ref_fourier.fourier_coefficients(SETS[:, None]),
        atol=1e-5,
    )
    assert jnp.allclose(game.contributions(), ref_moebius.contributions(), atol=1e-5)
    if isinstance(game, KnownAdditivity):
        assert jax.jit(lambda g: g.additivity())(game) == N_PLAYERS


def test_sparse_fourier_and_moebius_compute_cross_basis_without_eval(monkeypatch):
    fourier = _fourier(TABLES)
    moebius = _moebius(TABLES)

    def fail_eval(*args, **kwargs):
        msg = "Sparse basis coefficient lookup must not call eval()"
        raise AssertionError(msg)

    monkeypatch.setattr(FinitePlayerGame, "eval", fail_eval)

    assert jnp.allclose(fourier.empty_value(), moebius.empty_value(), atol=1e-5)
    assert jnp.allclose(
        fourier.coefficients(SETS[:, None]), moebius.coefficients(SETS[:, None]), atol=1e-5
    )
    assert jnp.allclose(
        moebius.fourier_coefficients(SETS[:, None]),
        fourier.fourier_coefficients(SETS[:, None]),
        atol=1e-5,
    )
    for order in range(N_PLAYERS + 2):
        assert jnp.allclose(
            fourier.coefficients(order=order), moebius.coefficients(order=order), atol=1e-5
        )
        assert jnp.allclose(
            moebius.coefficients(order=order, basis=Basis.FOURIER),
            fourier.coefficients(order=order, basis=Basis.FOURIER),
            atol=1e-5,
        )

    zero_fourier = SparseFourierGame(
        values=jnp.array([[4.0, 3.0]], dtype=jnp.float32),
        interactions=jnp.zeros((2, 0), dtype=bool),
        players=FinitePlayers(0),
        additivity=0,
    )
    assert jnp.allclose(zero_fourier.empty_value(), jnp.array([7.0]))
    assert jnp.allclose(zero_fourier.coefficients(order=0), jnp.array([[7.0]]))
    assert zero_fourier.coefficients(order=1).shape == (1, 0)

    # High-order float16 Fourier scaling must use ldexp (no intermediate 2**16 overflow).
    n16 = 16
    full16 = jnp.ones((1, n16), dtype=bool)
    f16_game = SparseFourierGame(
        values=jnp.array([[2.0**-8], [0.0]], dtype=jnp.float16),
        interactions=full16,
        players=FinitePlayers(n16),
        additivity=n16,
    )
    f16_full = f16_game.coefficients(full16[0])
    assert f16_full.dtype == jnp.float16
    assert jnp.array_equal(f16_full, jnp.array([256.0, 0.0], dtype=jnp.float16))


@pytest.mark.parametrize("game_class", [SparseMoebiusGame, SparseFourierGame])
def test_additivity_must_bound_the_interactions(game_class):
    with pytest.raises(ValueError, match="additivity"):
        game_class(
            values=jnp.zeros(len(SETS)),
            interactions=SETS,
            players=FinitePlayers(N_PLAYERS),
            additivity=N_PLAYERS - 1,
        )


def test_one_additive_sparse_moebius_supports_offset_sparse_and_permuted_singletons():
    values = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    players = FinitePlayers(N_PLAYERS)
    singletons = jnp.eye(N_PLAYERS, dtype=bool)[jnp.array([2, 0, 1])]
    additive = SparseMoebiusGame(
        values=values, interactions=singletons, players=players, additivity=1
    )
    assert isinstance(additive, KnownAdditivity)
    assert additive.additivity() == 1
    assert jnp.allclose(additive.empty_value(), jnp.zeros((2,)))

    # Arbitrary empty-coalition offset and sparse singletons are supported directly.
    with_offset = SparseMoebiusGame(
        values=jnp.array([[5.0, 3.0, 1.0], [7.0, -2.0, 4.0]]),
        interactions=jnp.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=bool),
        players=players,
        additivity=1,
    )
    assert jnp.array_equal(with_offset.empty_value(), jnp.array([5.0, 7.0]))
    assert jnp.array_equal(
        with_offset.contributions(), jnp.array([[1.0, 0.0, 3.0], [4.0, 0.0, -2.0]])
    )
    assert jnp.allclose(
        with_offset(SETS[:, None]),
        with_offset.empty_value()[None, :]
        + jnp.einsum("cp,gp->cg", SETS.astype(jnp.float32), with_offset.contributions()),
    )

    # Constant (0-additive) SparseMoebiusGame is also supported.
    constant = SparseMoebiusGame(
        values=jnp.array([9.0]),
        interactions=jnp.array([[0, 0, 0]], dtype=bool),
        players=players,
        additivity=0,
    )
    assert constant.additivity() == 0
    assert jnp.array_equal(constant.empty_value(), jnp.array(9.0))
    assert jnp.array_equal(constant.contributions(), jnp.zeros(N_PLAYERS))
    assert jnp.array_equal(constant.coefficients(order=1), jnp.zeros(N_PLAYERS))
    assert jnp.allclose(constant(SETS), jnp.full((len(SETS),), 9.0))

    # Booleans and bounds below interaction sizes are rejected.
    with pytest.raises(ValueError, match="additivity"):
        SparseMoebiusGame(values=values, interactions=singletons, players=players, additivity=0)
    with pytest.raises(TypeError, match="bool"):
        SparseMoebiusGame(values=values, interactions=singletons, players=players, additivity=True)

    # Non-bool int subclasses normalize to plain int on both the game and affine wrappers.
    class _OrderInt(int):
        pass

    norm_one = SparseMoebiusGame(
        values=values, interactions=singletons, players=players, additivity=_OrderInt(1)
    )
    assert type(norm_one.additivity()) is int
    assert norm_one.additivity() == 1
    scaled_norm = 2.0 * norm_one
    assert type(scaled_norm.additivity()) is int
    assert scaled_norm.additivity() == 1

    # Omitting additivity infers the tight bound from concrete interactions.
    inferred = SparseMoebiusGame(values=values, interactions=singletons, players=players)
    assert inferred.additivity() == 1


def test_zero_player_sparse_moebius_evaluates_empty_coalition():
    players = FinitePlayers(0)
    empty = players.empty_coalition()

    scalar_zero = SparseMoebiusGame(
        values=jnp.zeros((0,), dtype=jnp.float32),
        interactions=jnp.zeros((0, 0), dtype=bool),
        players=players,
    )
    batched_zero = SparseMoebiusGame(
        values=jnp.zeros((2, 0), dtype=jnp.float32),
        interactions=jnp.zeros((0, 0), dtype=bool),
        players=players,
    )
    nonzero_offset = SparseMoebiusGame(
        values=jnp.array([[7.0], [-3.0]], dtype=jnp.float32),
        interactions=jnp.zeros((1, 0), dtype=bool),
        players=players,
        additivity=0,
    )

    for game, expected in (
        (scalar_zero, jnp.zeros((), dtype=jnp.float32)),
        (batched_zero, jnp.zeros((2,), dtype=jnp.float32)),
        (nonzero_offset, jnp.array([7.0, -3.0], dtype=jnp.float32)),
    ):
        assert game.additivity() == 0
        assert jnp.array_equal(game(empty), game.empty_value())
        assert jnp.array_equal(game(empty), expected)
        assert game.contributions().shape == (*game.shape, 0)
        assert jnp.array_equal(game.coefficients(order=0), expected[..., None])
        assert game.coefficients(order=1).shape == (*game.shape, 0)
        assert jnp.array_equal(jax.jit(lambda g, c: g(c))(game, empty), expected)


@pytest.mark.parametrize("game_class", [SparseMoebiusGame, SparseFourierGame])
def test_basis_game_supports_batched_and_broadcasted_interactions(game_class):
    players = FinitePlayers(N_PLAYERS)
    supports = jnp.array(
        [
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
        ],
        dtype=bool,
    )
    values = jnp.array(
        [
            [[1.0, 2.0, -3.0], [0.5, -1.0, 4.0]],
            [[2.0, 3.0, 5.0], [-2.0, 1.5, -0.5]],
        ],
        dtype=jnp.float32,
    )
    partial_game = game_class(
        values=values, interactions=supports[:, None, :, :], players=players, additivity=2
    )
    full_supports = jnp.broadcast_to(supports[:, None, :, :], (2, 2, 3, N_PLAYERS))
    full_game = game_class(values=values, interactions=full_supports, players=players, additivity=2)

    ref_tables = jnp.stack(
        [
            jnp.stack(
                [
                    game_class(
                        values=values[i, j],
                        interactions=supports[i],
                        players=players,
                        additivity=2,
                    )(SETS)
                    for j in range(2)
                ],
                axis=1,
            )
            for i in range(2)
        ],
        axis=1,
    )
    ref_moebius = jnp.stack(
        [
            jnp.stack(
                [
                    game_class(
                        values=values[i, j],
                        interactions=supports[i],
                        players=players,
                        additivity=2,
                    ).coefficients(SETS)
                    for j in range(2)
                ],
                axis=1,
            )
            for i in range(2)
        ],
        axis=1,
    )

    for game in (partial_game, full_game):
        assert game.shape == (2, 2)
        assert jnp.allclose(game(SETS[:, None, None]), ref_tables, atol=1e-5)
        assert jnp.allclose(
            jax.jit(lambda g, c: g(c))(game, SETS[:, None, None]), ref_tables, atol=1e-5
        )
        assert jnp.allclose(game.empty_value(), ref_tables[0], atol=1e-5)
        assert jnp.allclose(game.coefficients(SETS[:, None, None]), ref_moebius, atol=1e-5)
        assert jnp.allclose(
            jax.jit(lambda g, q: g.coefficients(q))(game, SETS[:, None, None]),
            ref_moebius,
            atol=1e-5,
        )
        assert jnp.allclose(game.coefficients(order=0), ref_moebius[0][..., None], atol=1e-5)
        assert jnp.allclose(
            game.coefficients(order=1),
            jnp.moveaxis(ref_moebius[jnp.array([1, 2, 4])], 0, -1),
            atol=1e-5,
        )
        assert jnp.allclose(
            game.coefficients(order=2),
            jnp.moveaxis(ref_moebius[jnp.array([3, 5, 6])], 0, -1),
            atol=1e-5,
        )
        assert jnp.allclose(game.coefficients(order=3), jnp.zeros((2, 2, 1)), atol=1e-5)
        assert jnp.allclose(game[1, 0](SETS), ref_tables[:, 1, 0], atol=1e-5)
        assert jnp.allclose(game.reshape(4)(SETS[:, None]), ref_tables.reshape(8, 4), atol=1e-5)
        paired = SETS[:2]
        assert jnp.allclose(
            game(paired),
            jnp.stack([ref_tables[0, :, 0], ref_tables[1, :, 1]], axis=-1),
            atol=1e-5,
        )

    with pytest.raises((TypeError, ValueError)):
        game_class(
            values=values[0],
            interactions=supports[:, None, :, :],
            players=players,
            additivity=2,
        )

    bad_supports = supports.at[1, 2].set(jnp.array([True, True, True]))
    with pytest.raises(ValueError, match="additivity"):
        game_class(
            values=values, interactions=bad_supports[:, None, :, :], players=players, additivity=2
        )


def test_sparse_moebius_batched_interactions_contributions_and_arithmetic():
    players = FinitePlayers(N_PLAYERS)
    interactions = jnp.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
        ],
        dtype=bool,
    )
    values = jnp.array([[5.0, 2.0, -1.0], [3.0, 4.0, 6.0]], dtype=jnp.float32)

    moebius = SparseMoebiusGame(
        values=values, interactions=interactions, players=players, additivity=1
    )
    assert moebius.shape == (2,)
    assert jnp.array_equal(moebius.empty_value(), jnp.array([5.0, 3.0]))
    assert jnp.array_equal(
        moebius.contributions(),
        jnp.array([[2.0, 0.0, -1.0], [0.0, 4.0, 6.0]]),
    )
    assert jnp.allclose(
        jax.jit(lambda g: g.contributions())(moebius),
        jnp.array([[2.0, 0.0, -1.0], [0.0, 4.0, 6.0]]),
    )
    scaled = 2.0 * moebius + 1.0
    assert scaled.additivity() == 1
    assert jnp.allclose(scaled(SETS[:, None]), 2.0 * moebius(SETS[:, None]) + 1.0)
    assert jnp.allclose(
        scaled.contributions(),
        jnp.array([[4.0, 0.0, -2.0], [0.0, 8.0, 12.0]]),
    )


def test_unsigned_complex_and_narrowing_basis_edge_cases():
    # 1. Unsigned coalition values promote to signed arithmetic in Möbius / Fourier / additivity.
    @dataclass(frozen=True)
    class _TableLike(CoalitionDefinedGame[int, jax.Array]):
        table: jax.Array
        _players: FinitePlayers[int]

        @property
        @override
        def players(self) -> FinitePlayers[int]:
            return self._players

        @property
        @override
        def shape(self) -> tuple[int, ...]:
            return self.table.shape[:-1]

        @override
        def _coalition_bool(self, coalition: jax.Array) -> jax.Array:
            weights = 1 << jnp.arange(self.n_players, dtype=jnp.int32)
            idx = jnp.sum(coalition.astype(jnp.int32) * weights, axis=-1)
            shape = jnp.broadcast_shapes(self.shape, idx.shape)
            table = jnp.broadcast_to(self.table, (*shape, self.table.shape[-1]))
            return jnp.take_along_axis(table, jnp.broadcast_to(idx, shape)[..., None], axis=-1)[
                ..., 0
            ]

    u8 = _TableLike(jnp.array([0, 1, 1, 2], dtype=jnp.uint8), FinitePlayers(2))
    assert jnp.array_equal(u8.moebius_coefficients(order=2), jnp.array([0], dtype=jnp.int32))
    assert jnp.allclose(u8.fourier_coefficients(order=1), jnp.array([-0.5, -0.5]))
    assert u8.additivity() == 1

    # 2. Complex Fourier coefficients preserve imaginary components through _scale_by_pow2.
    c_game = _TableLike(jnp.array([0.0 + 0.0j, 1.0 + 1.0j], dtype=jnp.complex64), FinitePlayers(1))
    c_cast = CastGame(c_game, jnp.complex64)
    assert jnp.allclose(c_cast.fourier_coefficients(order=1), jnp.array([-0.5 - 0.5j]))

    # 3. Integer SumReductionGame preserves fractional Fourier coefficients.
    int_pair = _TableLike(jnp.array([[0, 1], [0, 1]], dtype=jnp.int32), FinitePlayers(1))
    int_sum = int_pair.sum(axis=0, dtype=jnp.int32)
    assert jnp.allclose(
        int_sum.fourier_coefficients(jnp.array([[False], [True]])),
        jnp.array([1.0, -1.0]),
    )

    # 4. Affine order queries preserve VectorValue structure supplied by bias.
    scalar_dense = DenseCoalitionGame(jnp.array([0.0, 1.0, 2.0, 3.0]), FinitePlayers(2))
    vec_bias = VectorValue(jnp.array([10.0, 20.0]))
    shifted = scalar_dense + vec_bias
    assert isinstance(shifted.contributions(), VectorValue)
    assert isinstance(shifted.fourier_coefficients(order=1), VectorValue)
    assert jnp.allclose(
        shifted.contributions().values,
        shifted.coefficients(jnp.eye(2, dtype=bool)).values,
    )

    # 5. Narrowing integer cast does not falsely preserve additivity on wraparound.
    int_add = SparseMoebiusGame(
        values=jnp.array([100.0, 100.0]),
        interactions=jnp.array([[True, False], [False, True]]),
        players=FinitePlayers(2),
        additivity=1,
    )
    narrow = CastGame(CastGame(int_add, jnp.int32), jnp.int8)
    assert not isinstance(narrow, KnownAdditivity)
    assert jnp.array_equal(narrow.moebius_coefficients(order=2), jnp.array([-256]))
    assert narrow.additivity() == 2


def test_native_moebius_empty_batch_bounded_64p_cast_routing_and_eval_free_known_additivity():
    # 1. Empty-batch MoebiusDefinedGame does not recurse on empty_value().
    @dataclass(frozen=True)
    class _EmptyMoebius(MoebiusDefinedGame[int, jax.Array]):
        _players: FinitePlayers[int]

        @property
        @override
        def players(self) -> FinitePlayers[int]:
            return self._players

        @property
        @override
        def shape(self) -> tuple[int, ...]:
            return self._players.shape

        @override
        def _moebius_bool(self, interaction: jax.Array) -> jax.Array:
            batch = jnp.broadcast_shapes(self.shape, interaction.shape[:-1])
            return jnp.zeros(batch, dtype=jnp.float32)

    empty_m = _EmptyMoebius(FinitePlayers(jnp.zeros((0, 2), dtype=jnp.int32)))
    assert empty_m.empty_value().shape == (0,)

    # 2. Bounded 64-player MoebiusDefinedGame avoids 2**64 powerset allocation.
    @dataclass(frozen=True)
    class _Bounded64Moebius(MoebiusDefinedGame[int, jax.Array], KnownAdditivity[int, jax.Array]):
        _players: FinitePlayers[int]
        _additivity: int = 1

        @property
        @override
        def players(self) -> FinitePlayers[int]:
            return self._players

        @property
        @override
        def shape(self) -> tuple[int, ...]:
            return ()

        @override
        def _moebius_bool(self, interaction: jax.Array) -> jax.Array:
            is_singleton = jnp.sum(interaction, axis=-1) == 1
            return jnp.where(is_singleton, 1.0, 0.0)

    g64 = _Bounded64Moebius(FinitePlayers(64))
    assert jnp.allclose(g64.full_value(), 64.0)
    assert jnp.allclose(g64.fourier_coefficients(order=1), jnp.full((64,), -0.5))

    # 3. Custom MoebiusDefinedGame -> DenseCoalitionGame routes via DenseMoebiusGame (2**n queries).
    query_shapes: list[tuple[int, ...]] = []

    @dataclass(frozen=True)
    class _TrackedMoebius(MoebiusDefinedGame[int, jax.Array]):
        _players: FinitePlayers[int]

        @property
        @override
        def players(self) -> FinitePlayers[int]:
            return self._players

        @property
        @override
        def shape(self) -> tuple[int, ...]:
            return ()

        @override
        def _moebius_bool(self, interaction: jax.Array) -> jax.Array:
            query_shapes.append(interaction.shape)
            return jnp.sum(interaction, axis=-1).astype(jnp.float32)

    DenseCoalitionGame.from_game(_TrackedMoebius(FinitePlayers(4)))
    assert query_shapes == [(16, 4)]

    # 4. isinstance(wrapper, KnownAdditivity) never invokes coalition code.
    eval_calls = 0

    @dataclass(frozen=True)
    class _CountingCoalition(CoalitionDefinedGame[int, jax.Array]):
        _players: FinitePlayers[int]

        @property
        @override
        def players(self) -> FinitePlayers[int]:
            return self._players

        @property
        @override
        def shape(self) -> tuple[int, ...]:
            return (2,)

        @override
        def _coalition_bool(self, coalition: jax.Array) -> jax.Array:
            nonlocal eval_calls
            eval_calls += 1
            batch = jnp.broadcast_shapes(self.shape, coalition.shape[:-1])
            return jnp.zeros(batch, dtype=jnp.float32)

    child = _CountingCoalition(FinitePlayers(2))
    assert not isinstance(child.sum(axis=0), KnownAdditivity)
    assert not isinstance(CastGame(child, jnp.float32), KnownAdditivity)
    assert not isinstance(child / 2.0, KnownAdditivity)
    assert eval_calls == 0
