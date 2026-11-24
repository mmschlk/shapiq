"""Möbius and Fourier games reproduce the coalition table they were transformed from."""

import jax
import jax.numpy as jnp
import pytest
from flextype import cast as flex_cast

from shapiq.game.discretederivative import subset_patterns
from shapiq.game.finite import FinitePlayerGame
from shapiq.game.fourier import FourierGame
from shapiq.game.kadditive import AdditiveGame, KAdditiveGame
from shapiq.game.moebius import AdditiveMoebiusGame, MoebiusGame
from shapiq.game.table import TableGame
from shapiq.players import FinitePlayers

N_PLAYERS = 3
# Rows are coalitions in table order and double as the full list of interactions.
SETS = subset_patterns(N_PLAYERS)
TABLES = jnp.array(
    [[0.0, 1.0, 2.0, 4.0, 3.0, 5.0, 6.0, 10.0], [1.0, -2.0, 0.5, 3.0, 2.0, 0.0, 7.0, -1.0]]
)


def _overlap(left: jax.Array, right: jax.Array) -> jax.Array:
    return jnp.sum(left[:, None, :] & right[None, :, :], axis=-1)


def _moebius(tables: jax.Array) -> MoebiusGame:
    # m(T) = sum over S ⊆ T of (-1)**|T \ S| v(S)
    subset = ~jnp.any(SETS[None, :, :] & ~SETS[:, None, :], axis=-1)  # [T, S]: S ⊆ T
    sign = (-1.0) ** (jnp.sum(SETS, axis=-1)[:, None] - _overlap(SETS, SETS))
    return MoebiusGame(
        values=tables @ (subset * sign).T,
        interactions=SETS,
        players=FinitePlayers(N_PLAYERS),
        additivity=N_PLAYERS,
    )


def _fourier(tables: jax.Array) -> FourierGame:
    # f(T) = 2**-n sum over S of v(S) (-1)**|T & S|
    chi = (-1.0) ** _overlap(SETS, SETS)
    return FourierGame(
        values=tables @ chi / 2**N_PLAYERS,
        interactions=SETS,
        players=FinitePlayers(N_PLAYERS),
        additivity=N_PLAYERS,
    )


@pytest.mark.parametrize("build", [_moebius, _fourier])
def test_basis_game_reproduces_the_table_batched_indexed_and_jitted(build):
    game = build(TABLES)
    table = TableGame(TABLES, FinitePlayers(N_PLAYERS))
    expected = table(SETS[:, None])  # (coalitions, games)

    assert game.shape == (2,)
    assert jnp.allclose(game(SETS[:, None]), expected, atol=1e-5)
    assert jnp.allclose(game[1](SETS), expected[:, 1], atol=1e-5)
    assert jnp.allclose(jax.jit(lambda g, c: g(c))(game, SETS[:, None]), expected, atol=1e-5)
    assert jax.jit(lambda g: g.additivity)(game) == N_PLAYERS


def test_fourier_game_computes_moebius_coefficients_without_eval(monkeypatch):
    fourier = _fourier(TABLES)
    moebius = _moebius(TABLES)

    def fail_eval(*args, **kwargs):
        msg = "FourierGame coefficient lookup must not call eval()"
        raise AssertionError(msg)

    monkeypatch.setattr(FinitePlayerGame, "eval", fail_eval)

    assert jnp.allclose(fourier.empty_value(), moebius.empty_value(), atol=1e-5)
    assert jnp.allclose(
        fourier.coefficients(SETS[:, None]), moebius.coefficients(SETS[:, None]), atol=1e-5
    )
    for order in range(N_PLAYERS + 2):
        assert jnp.allclose(
            fourier.coefficients(order=order), moebius.coefficients(order=order), atol=1e-5
        )
    # Zero-player Fourier game and above-additivity queries work without eval or overflow.
    zero_fourier = FourierGame(
        values=jnp.array([[4.0, 3.0]], dtype=jnp.float32),
        interactions=jnp.zeros((2, 0), dtype=bool),
        players=FinitePlayers(0),
        additivity=0,
    )
    assert jnp.allclose(zero_fourier.empty_value(), jnp.array([7.0]))
    assert jnp.allclose(zero_fourier.coefficients(order=0), jnp.array([[7.0]]))
    assert zero_fourier.coefficients(order=1).shape == (1, 0)

    # High-order float16 Fourier scaling must use ldexp (no intermediate 2**16 overflow or 0*inf=nan).
    n16 = 16
    full16 = jnp.ones((1, n16), dtype=bool)
    f16_game = FourierGame(
        values=jnp.array([[2.0**-8], [0.0]], dtype=jnp.float16),
        interactions=full16,
        players=FinitePlayers(n16),
        additivity=n16,
    )
    f16_full = f16_game.coefficients(full16[0])
    assert f16_full.dtype == jnp.float16
    assert jnp.array_equal(f16_full, jnp.array([256.0, 0.0], dtype=jnp.float16))


def test_kadditive_and_additive_require_explicit_coefficient_and_contribution_methods():
    assert "_coefficients_for_interactions" in KAdditiveGame.__abstractmethods__
    assert "contributions" in AdditiveGame.__abstractmethods__


@pytest.mark.parametrize("game_class", [MoebiusGame, FourierGame])
def test_additivity_must_bound_the_interactions(game_class):
    with pytest.raises(ValueError, match="additivity"):
        game_class(
            values=jnp.zeros(len(SETS)),
            interactions=SETS,
            players=FinitePlayers(N_PLAYERS),
            additivity=N_PLAYERS - 1,
        )


def test_additive_moebius_game_supports_offset_sparse_and_permuted_singletons():
    values = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    players = FinitePlayers(N_PLAYERS)
    singletons = jnp.eye(N_PLAYERS, dtype=bool)[jnp.array([2, 0, 1])]
    additive = AdditiveMoebiusGame(values=values, interactions=singletons, players=players)
    stored = MoebiusGame(values=values, interactions=singletons, players=players, additivity=1)
    expected = stored(SETS[:, None])

    assert isinstance(additive, MoebiusGame)
    assert additive.additivity == 1
    assert jnp.allclose(additive.empty_value(), jnp.zeros((2,)))
    assert jnp.allclose(additive(SETS[:, None]), expected)
    assert jnp.allclose(additive[1](SETS), expected[:, 1])
    assert jnp.allclose(jax.jit(lambda g, c: g(c))(additive, SETS[:, None]), expected)

    # Arbitrary empty-coalition offset and sparse singletons are supported directly.
    with_offset = AdditiveMoebiusGame(
        values=jnp.array([[5.0, 3.0, 1.0], [7.0, -2.0, 4.0]]),
        interactions=jnp.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=bool),
        players=players,
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

    # Constant (0-additive) AdditiveMoebiusGame is also supported.
    constant = AdditiveMoebiusGame(
        values=jnp.array([9.0]),
        interactions=jnp.array([[0, 0, 0]], dtype=bool),
        players=players,
        additivity=0,
    )
    assert constant.additivity == 0
    assert jnp.array_equal(constant.empty_value(), jnp.array(9.0))
    assert jnp.array_equal(constant.contributions(), jnp.zeros(N_PLAYERS))
    assert jnp.array_equal(constant.coefficients(order=1), jnp.zeros(N_PLAYERS))
    assert jnp.allclose(constant(SETS), jnp.full((len(SETS),), 9.0))

    # Higher-order interactions, booleans, and additivity bounds > 1 are rejected.
    with pytest.raises(ValueError, match="additivity"):
        AdditiveMoebiusGame(
            values=jnp.array([1.0]),
            interactions=jnp.array([[1, 1, 0]], dtype=bool),
            players=players,
        )
    with pytest.raises((TypeError, ValueError)):
        AdditiveMoebiusGame(values=values, interactions=singletons, players=players, additivity=2)
    with pytest.raises(ValueError, match="additivity"):
        AdditiveMoebiusGame(values=values, interactions=singletons, players=players, additivity=0)
    with pytest.raises(TypeError, match="bool"):
        AdditiveMoebiusGame(
            values=values, interactions=singletons, players=players, additivity=True
        )
    with pytest.raises(TypeError, match="bool"):
        MoebiusGame(values=values, interactions=singletons, players=players, additivity=True)

    # Non-bool int subclasses normalize to plain int 0 or 1 on both the game and affine wrappers.
    class _OrderInt(int):
        pass

    norm_one = AdditiveMoebiusGame(
        values=values, interactions=singletons, players=players, additivity=_OrderInt(1)
    )
    assert type(norm_one.additivity) is int
    assert norm_one.additivity == 1
    scaled_norm = 2.0 * norm_one
    assert type(scaled_norm.additivity) is int
    assert scaled_norm.additivity == 1

    norm_zero = AdditiveMoebiusGame(
        values=jnp.array([9.0]),
        interactions=jnp.array([[0, 0, 0]], dtype=bool),
        players=players,
        additivity=_OrderInt(0),
    )
    assert type(norm_zero.additivity) is int
    assert norm_zero.additivity == 0


def test_zero_player_additive_moebius_game_evaluates_empty_coalition():
    players = FinitePlayers(0)
    empty = players.empty_coalition()

    scalar_zero = AdditiveMoebiusGame(
        values=jnp.zeros((0,), dtype=jnp.float32),
        interactions=jnp.zeros((0, 0), dtype=bool),
        players=players,
    )
    batched_zero = AdditiveMoebiusGame(
        values=jnp.zeros((2, 0), dtype=jnp.float32),
        interactions=jnp.zeros((0, 0), dtype=bool),
        players=players,
    )
    nonzero_offset = AdditiveMoebiusGame(
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
        assert game.additivity == 0
        assert jnp.array_equal(game(empty), game.empty_value())
        assert jnp.array_equal(game(empty), expected)
        assert game.contributions().shape == (*game.shape, 0)
        assert jnp.array_equal(game.coefficients(order=0), expected[..., None])
        assert game.coefficients(order=1).shape == (*game.shape, 0)
        assert jnp.array_equal(jax.jit(lambda g, c: g(c))(game, empty), expected)


@pytest.mark.parametrize("game_class", [MoebiusGame, FourierGame])
def test_basis_game_supports_batched_and_broadcasted_interactions(game_class):
    players = FinitePlayers(N_PLAYERS)
    # Two rows of distinct 3-interaction supports broadcast across shape (2, 2).
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
    # Partially broadcasted (2, 1, 3, 3) and fully per-game (2, 2, 3, 3) interactions.
    partial_game = game_class(
        values=values, interactions=supports[:, None, :, :], players=players, additivity=2
    )
    full_supports = jnp.broadcast_to(supports[:, None, :, :], (2, 2, 3, N_PLAYERS))
    full_game = game_class(values=values, interactions=full_supports, players=players, additivity=2)

    # Reference: evaluate each (i, j) element independently as a scalar game.
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
    )  # (8, 2, 2)
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
    )  # (8, 2, 2)

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
        # Indexing, reshape, and paired (2, P) coalition queries.
        assert jnp.allclose(game[1, 0](SETS), ref_tables[:, 1, 0], atol=1e-5)
        assert jnp.allclose(game.reshape(4)(SETS[:, None]), ref_tables.reshape(8, 4), atol=1e-5)
        paired = SETS[:2]  # (2, P) broadcasts against last game axis of (2, 2)
        assert jnp.allclose(
            game(paired),
            jnp.stack([ref_tables[0, :, 0], ref_tables[1, :, 1]], axis=-1),
            atol=1e-5,
        )

    # Extra leading singleton batch axis on interactions must not enlarge values's batch shape.
    with pytest.raises((TypeError, ValueError)):
        game_class(
            values=values[0],
            interactions=supports[:, None, :, :],
            players=players,
            additivity=2,
        )

    # An interaction exceeding additivity in any batch slice is rejected.
    bad_supports = supports.at[1, 2].set(jnp.array([True, True, True]))
    with pytest.raises(ValueError, match="additivity"):
        game_class(
            values=values, interactions=bad_supports[:, None, :, :], players=players, additivity=2
        )


def test_additive_moebius_game_supports_batched_interactions():
    players = FinitePlayers(N_PLAYERS)
    interactions = jnp.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
        ],
        dtype=bool,
    )
    values = jnp.array([[5.0, 2.0, -1.0], [3.0, 4.0, 6.0]], dtype=jnp.float32)

    moebius = MoebiusGame(values=values, interactions=interactions, players=players, additivity=1)
    additive = flex_cast(moebius, AdditiveGame)
    assert type(additive) is AdditiveMoebiusGame
    assert additive.interactions is moebius.interactions
    assert additive.shape == (2,)
    assert jnp.array_equal(additive.empty_value(), jnp.array([5.0, 3.0]))
    assert jnp.array_equal(
        additive.contributions(),
        jnp.array([[2.0, 0.0, -1.0], [0.0, 4.0, 6.0]]),
    )
    assert jnp.allclose(additive(SETS[:, None]), moebius(SETS[:, None]))
    assert jnp.allclose(
        jax.jit(lambda g: g.contributions())(additive),
        jnp.array([[2.0, 0.0, -1.0], [0.0, 4.0, 6.0]]),
    )
    # Arithmetic combination of batched-interaction additive games stays additive and accurate.
    scaled = 2.0 * additive + 1.0
    assert isinstance(scaled, AdditiveGame)
    assert jnp.allclose(scaled(SETS[:, None]), 2.0 * moebius(SETS[:, None]) + 1.0)
    assert jnp.allclose(
        scaled.contributions(),
        jnp.array([[4.0, 0.0, -2.0], [0.0, 8.0, 12.0]]),
    )

    # Zero-player and empty-batch batched interactions work cleanly.
    zero_p = FinitePlayers(0)
    zero_batched = AdditiveMoebiusGame(
        values=jnp.array([[7.0], [-2.0]], dtype=jnp.float32),
        interactions=jnp.zeros((2, 1, 0), dtype=bool),
        players=zero_p,
    )
    assert zero_batched.additivity == 0
    assert jnp.array_equal(zero_batched.empty_value(), jnp.array([7.0, -2.0]))
    assert zero_batched.contributions().shape == (2, 0)

    empty_batch = AdditiveMoebiusGame(
        values=jnp.zeros((0, 2), dtype=jnp.float32),
        interactions=jnp.zeros((0, 2, N_PLAYERS), dtype=bool),
        players=players,
    )
    assert empty_batch.shape == (0,)
    assert empty_batch.contributions().shape == (0, N_PLAYERS)
