"""Lazy finite game arithmetic, basis forwarding, and additivity propagation."""

from dataclasses import dataclass
from inspect import isabstract
from typing import Any, ClassVar, override
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError
from ndlike import jax_functions as jf
from ndlike.protected_axis.jax import JaxAxisProtected
from ndlike.tuple.jax import JaxBroadcastTuple

from shapiq.coalition.coalition import pack_bool_coalition
from shapiq.game import (
    AffineGame,
    ArithmeticGame,
    CastGame,
    CoalitionDefinedGame,
    DenseCoalitionGame,
    FinitePlayerGame,
    KnownAdditivity,
    MappedGame,
    ProductGame,
    SparseFourierGame,
    SparseMoebiusGame,
    subset_patterns,
)
from shapiq.players import FinitePlayers, LatentPlayers
from shapiq.value import VectorValue


def _table_game(n_players: int = 2, batch_shape: tuple[int, ...] = (2,)) -> DenseCoalitionGame:
    size = 1 << n_players
    count = int(np.prod(batch_shape)) * size
    values = jnp.arange(float(count)).reshape(*batch_shape, size)
    return DenseCoalitionGame(values, FinitePlayers(n_players))


def _additive_moebius(
    values: jax.Array,
    players: FinitePlayers,
    perm: list[int] | None = None,
) -> SparseMoebiusGame:
    n = players.n_players
    order = list(range(n)) if perm is None else perm
    interactions = jnp.eye(n, dtype=bool)[jnp.array(order)]
    return SparseMoebiusGame(
        values=values, interactions=interactions, players=players, additivity=1
    )


def _exact_moebius_table(values: jax.Array, n_players: int) -> jax.Array:
    """Compute exact Möbius coefficients in subset_patterns order from coalition tables."""
    sets = subset_patterns(n_players)
    subset = ~jnp.any(sets[None, :, :] & ~sets[:, None, :], axis=-1)
    overlap = jnp.sum(sets[:, None, :] & sets[None, :, :], axis=-1)
    sign = (-1.0) ** (jnp.sum(sets, axis=-1)[:, None] - overlap)
    return values @ (subset * sign).T


@dataclass(frozen=True)
class _VectorValue(JaxAxisProtected["_VectorValue"]):
    values: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"values": 1}
    permitted_functions: ClassVar[set] = {
        jf.jax_add,
        jf.jax_multiply,
        jf.jax_power,
        jf.jax_sum,
        jf.jax_where,
    }


# --- Endpoint access and abstract ArithmeticGame contracts ---


def test_finite_game_empty_and_full_values():
    players = FinitePlayers(jnp.array([[0, 2], [1, 2]]), latent_n_players=3)
    table = DenseCoalitionGame(
        jnp.array([[10.0, 11.0, 12.0, 15.0], [20.0, 21.0, 22.0, 25.0]]), players
    )
    assert jnp.array_equal(table.empty_value(), jnp.array([10.0, 20.0]))
    assert jnp.array_equal(table.full_value(), jnp.array([15.0, 25.0]))

    zero_player = DenseCoalitionGame(jnp.array([7.0]), FinitePlayers(0))
    assert zero_player.empty_value().shape == ()
    assert float(zero_player.empty_value()) == 7.0
    assert float(zero_player.full_value()) == 7.0


def test_arithmetic_game_is_abstract_and_validates_inputs():
    assert isabstract(ArithmeticGame)
    assert ArithmeticGame.__abstractmethods__ == {"_compute"}

    with pytest.raises(ValueError, match="at least one game"):
        AffineGame(JaxBroadcastTuple(()), JaxBroadcastTuple(()))
    with pytest.raises(ValueError, match="at least one game"):
        ProductGame(JaxBroadcastTuple(()))
    with pytest.raises(ValueError, match="at least one game"):
        MappedGame(JaxBroadcastTuple(()), lambda vs: vs[0])

    with pytest.raises(TypeError, match="FinitePlayerGame"):
        ProductGame(JaxBroadcastTuple((jnp.ones((2,)),)))

    with pytest.raises(ValueError, match="one weight per game"):
        AffineGame(JaxBroadcastTuple((_table_game(),)), JaxBroadcastTuple(()))

    with pytest.raises(TypeError, match="numeric dtype"):
        AffineGame(
            JaxBroadcastTuple((_table_game(),)),
            JaxBroadcastTuple((jnp.ones((), dtype=bool),)),
        )


@pytest.mark.parametrize("mismatch", ["ids", "count", "latent", "metadata", "batch"])
def test_arithmetic_game_rejects_incompatible_players(mismatch):
    catalog = LatentPlayers(["a", "b", "c"])
    left = FinitePlayers(jnp.array([0, 1]), players=catalog)
    right = {
        "ids": lambda: FinitePlayers(jnp.array([0, 2]), players=catalog),
        "count": lambda: FinitePlayers(jnp.array([0]), players=catalog),
        "latent": lambda: FinitePlayers(2),
        "metadata": lambda: FinitePlayers(jnp.array([0, 1]), players=["a", "b", "c"]),
        "batch": lambda: FinitePlayers(jnp.zeros((3, 2), dtype=int), players=catalog),
    }[mismatch]()
    g_left = DenseCoalitionGame(jnp.zeros((2, 4)), left)
    g_right = DenseCoalitionGame(jnp.zeros((*right.shape, 2**right.n_players)), right)
    with pytest.raises(ValueError, match=r"compatible player sets|cannot be broadcast"):
        _ = g_left + g_right


def test_arithmetic_game_accepts_broadcast_equivalent_players():
    catalog = LatentPlayers(["a", "b", "c"])
    left = FinitePlayers(jnp.array([0, 2]), players=catalog)
    right = FinitePlayers(jnp.array([[0, 2], [0, 2]]), players=catalog)
    total = DenseCoalitionGame(jnp.arange(4.0), left) + DenseCoalitionGame(jnp.ones((2, 4)), right)
    assert total.shape == (2,)
    assert jnp.array_equal(total(jnp.array([True, True])), jnp.array([4.0, 4.0]))
    assert jnp.array_equal(total[1](jnp.array([True, True])), jnp.array(4.0))
    assert total.reshape(1, 2).shape == (1, 2)


def test_arithmetic_game_rejects_indexed_updates_and_fieldwise_arithmetic():
    g = _table_game() + 1.0
    coal = jnp.array([True, False])
    with pytest.raises(TypeError, match="not permitted"):
        _ = g.at[0].set(g[0])
    # .at[index].get() remains read-only.
    assert jnp.array_equal(g.at[0].get()(coal), g[0](coal))
    # Representation-level fieldwise arithmetic remains disabled.
    assert not g.permitted_functions
    for bad_op in (
        lambda: jf.jax_floor_divide(g, 2),
        lambda: jf.jax_mean(g),
    ):
        with pytest.raises(TypeError):
            bad_op()
    # Public numeric ndlike functions route to lazy game arithmetic consistently with operators.
    g2 = _table_game() + 2.0
    assert isinstance(jf.jax_add(g, g2), AffineGame)
    assert jnp.allclose(jf.jax_add(g, g2)(coal), (g + g2)(coal))
    assert jnp.allclose(jf.jax_add(2.0, g)(coal), (2.0 + g)(coal))
    assert jnp.allclose(jf.jax_subtract(g, g2)(coal), (g - g2)(coal))
    assert jnp.allclose(jf.jax_subtract(2.0, g)(coal), (2.0 - g)(coal))
    assert isinstance(jf.jax_multiply(g, g2), ProductGame)
    assert jnp.allclose(jf.jax_multiply(g, g2)(coal), (g * g2)(coal))
    assert jnp.allclose(jf.jax_true_divide(g, g2)(coal), (g / g2)(coal))
    assert jnp.allclose(jf.jax_true_divide(g, 2.0)(coal), (g / 2.0)(coal))
    assert jnp.allclose(jf.jax_true_divide(2.0, g)(coal), (2.0 / g)(coal))
    assert isinstance(jf.jax_power(g, g2), MappedGame)
    assert jnp.allclose(jf.jax_power(g, g2)(coal), (g**g2)(coal))
    assert jnp.allclose(jf.jax_power(g, 2)(coal), (g**2)(coal))
    assert jnp.allclose(jf.jax_power(2.0, g)(coal), (2.0**g)(coal))
    assert jnp.allclose(jf.jax_negative(g)(coal), (-g)(coal))
    assert jf.jax_positive(g) is g


# --- Laziness, native coalition forwarding, and ndlike shaping ---


def test_arithmetic_is_lazy_and_forwards_packed_and_bool_natively(monkeypatch):
    g1 = _table_game()
    g2 = _table_game()
    coal = jnp.array([True, False])
    packed = pack_bool_coalition(coal)
    expected = 2.0 * g1(coal) - g2(coal) + 3.0

    calls = Mock(wraps=DenseCoalitionGame.eval)

    def recording_eval(self, coalition):
        return calls(self, coalition)

    monkeypatch.setattr(DenseCoalitionGame, "eval", recording_eval)
    combined = 2.0 * g1 - g2 + 3.0
    sliced = combined[1]
    reshaped = combined.reshape(1, 2)
    calls.assert_not_called()

    assert jnp.array_equal(combined(coal), expected)
    assert calls.call_count == 2
    assert jnp.array_equal(sliced(packed), expected[1])
    assert jnp.array_equal(reshaped(packed), expected.reshape(1, 2))

    # Packed coalitions pass through without unpacking for packed-native children.
    def fail_unpack(*args, **kwargs):
        msg = "unpacked"
        raise AssertionError(msg)

    monkeypatch.setattr(jnp, "unpackbits", fail_unpack)
    assert jnp.array_equal(combined(packed), expected)


def test_ndlike_shaping_on_affine_product_and_mapped_games():
    g1 = _table_game(2, (2, 1))
    g2 = _table_game(2, (1, 3))
    coal = jnp.array([True, True])

    for expr in (
        2.0 * g1 + g2 - 1.5,
        g1 * g2,
        MappedGame(JaxBroadcastTuple((g1, g2)), lambda vs: vs[0] ** 2 - vs[1]),
        g1 / (g2 + 1.0),
    ):
        assert expr.shape == (2, 3)
        base = expr(coal)
        assert jnp.allclose(expr[1](coal), base[1])
        assert jnp.allclose(expr[:, 2](coal), base[:, 2])
        assert jnp.allclose(expr.reshape(6)(coal), base.reshape(6))
        assert jnp.allclose(expr.T(coal), base.T)
        stacked = jf.jax_stack([expr, expr])
        assert stacked.shape == (2, 2, 3)
        assert jnp.allclose(stacked(coal), jnp.stack([base, base]))


# --- AffineGame batch contract and nested normalization ---


def test_affine_games_define_batch_and_direct_enlargement_is_rejected():
    g = _table_game(2, ())
    with pytest.raises(ValueError, match="cannot broadcast"):
        AffineGame(
            JaxBroadcastTuple((g,)),
            JaxBroadcastTuple((jnp.array([1.0, 2.0]),)),
        )
    with pytest.raises(ValueError, match="cannot broadcast"):
        AffineGame(
            JaxBroadcastTuple((g,)),
            JaxBroadcastTuple((jnp.array(1.0),)),
            bias=jnp.array([1.0, 2.0]),
        )


def test_affine_normalizes_nested_known_wrappers_including_refinements():
    p = FinitePlayers(3)
    t1 = _table_game(3, (2,))
    t2 = _table_game(3, (2,))
    inner_table = 2.0 * t1 + 5.0
    outer_table = 3.0 * inner_table - 4.0 * t2 + 7.0
    assert type(outer_table) is AffineGame
    assert outer_table.games.members == (t1, t2)
    assert jnp.allclose(outer_table.weights.members[0], 6.0)
    assert jnp.allclose(outer_table.weights.members[1], -4.0)
    assert jnp.allclose(outer_table.bias, 22.0)

    coal = jnp.array([True, False, True])
    assert jnp.allclose(outer_table(coal), 3.0 * (2.0 * t1(coal) + 5.0) - 4.0 * t2(coal) + 7.0)

    # Nested affine combinations of 1-additive and 2-additive games flatten cleanly.
    am1 = _additive_moebius(jnp.array([1.0, 2.0, 3.0]), p)
    am2 = _additive_moebius(jnp.array([4.0, 5.0, 6.0]), p)
    inner_add = 2.0 * am1 + 3.0
    assert inner_add.additivity() == 1
    outer_add = 5.0 * inner_add - am2 + 1.0
    assert outer_add.additivity() == 1
    assert isinstance(outer_add, AffineGame)
    assert outer_add.games.members == (am1, am2)
    assert jnp.allclose(outer_add.weights.members[0], 10.0)
    assert jnp.allclose(outer_add.weights.members[1], -1.0)
    assert jnp.allclose(outer_add.bias, 16.0)
    assert jnp.allclose(outer_add(coal), 5.0 * (2.0 * am1(coal) + 3.0) - am2(coal) + 1.0)

    sm2 = SparseMoebiusGame(
        values=jnp.array([1.0, 2.0]),
        interactions=jnp.array([[1, 0, 0], [1, 1, 0]], dtype=bool),
        players=p,
        additivity=2,
    )
    inner_k = 2.0 * sm2 + 4.0
    assert inner_k.additivity() == 2
    outer_k = -3.0 * inner_k + am1 + 2.0
    assert outer_k.additivity() == 2
    assert outer_k.games.members == (sm2, am1)
    assert jnp.allclose(outer_k.weights.members[0], -6.0)
    assert jnp.allclose(outer_k.bias, -10.0)
    assert jnp.allclose(outer_k(coal), -3.0 * (2.0 * sm2(coal) + 4.0) + am1(coal) + 2.0)


# --- Operators, broadcasting, division, and structured values ---


def test_operators_cover_all_forms_and_broadcast_extra_constant_axes():
    g1 = _table_game(2, (2,))
    g2 = _table_game(2, ()) + 1.0
    coal = jnp.array([True, False])
    c_scalar = 2.5
    c_jax = jnp.array([2.0, 4.0])
    c_np = np.array([3.0, 5.0])

    assert (+g1) is g1
    assert jnp.allclose((-g1)(coal), -g1(coal))
    assert jnp.allclose((g1 + g2)(coal), g1(coal) + g2(coal))
    assert jnp.allclose((g1 - g2)(coal), g1(coal) - g2(coal))
    assert jnp.allclose((g1 * g2)(coal), g1(coal) * g2(coal))
    assert jnp.allclose((g1 / g2)(coal), g1(coal) / g2(coal))

    for const in (c_scalar, c_jax, c_np):
        assert jnp.allclose((g1 + const)(coal), g1(coal) + const)
        assert jnp.allclose((const + g1)(coal), const + g1(coal))
        assert jnp.allclose((g1 - const)(coal), g1(coal) - const)
        assert jnp.allclose((const - g1)(coal), const - g1(coal))
        assert jnp.allclose((g1 * const)(coal), g1(coal) * const)
        assert jnp.allclose((const * g1)(coal), const * g1(coal))
        assert jnp.allclose((g1 / const)(coal), g1(coal) / const)
        assert jnp.allclose((const / g2)(coal), const / g2(coal))

    # Operator constant with extra batch axes broadcasts the game operand explicitly.
    scalar_game = _table_game(2, ()) + 1.0
    batched_const = jnp.array([[2.0, 4.0], [6.0, 8.0]])
    for expr, expected in [
        (scalar_game + batched_const, scalar_game(coal) + batched_const),
        (batched_const - scalar_game, batched_const - scalar_game(coal)),
        (batched_const * scalar_game, batched_const * scalar_game(coal)),
        (scalar_game / batched_const, scalar_game(coal) / batched_const),
        (batched_const / scalar_game, batched_const / scalar_game(coal)),
    ]:
        assert expr.shape == (2, 2)
        assert jnp.allclose(expr(coal), expected)
        assert jnp.allclose(expr[1, 0](coal), expected[1, 0])

    # Unsupported operands return NotImplemented -> TypeError.
    for bad in ("abc", True, jnp.array([True, False]), object()):
        with pytest.raises(TypeError):
            _ = g1 + bad
        with pytest.raises(TypeError):
            _ = bad + g1
        with pytest.raises(TypeError):
            _ = g1 * bad
        with pytest.raises(TypeError):
            _ = g1 / bad


def test_operators_avoid_spurious_float64_promotion_under_x64():
    with jax.enable_x64(new_val=True):
        g1 = DenseCoalitionGame(jnp.arange(4, dtype=jnp.float32), FinitePlayers(2))
        g2 = DenseCoalitionGame(jnp.ones(4, dtype=jnp.float32), FinitePlayers(2))
        coal = jnp.array([True, False])
        for expr in (g1 + g2, g1 - g2, -g1, g1 * g2, g1 / (g2 + g2)):
            assert expr.dtype == jnp.float32
            assert expr(coal).dtype == jnp.float32


def test_mapped_game_supports_multi_game_alignment_and_structured_values():
    g1 = _table_game(2, (2, 1))
    g2 = _table_game(2, (1, 3))
    coal = jnp.array([True, False])

    # Batch-local fn receives aligned shapes even when ignoring one operand or stacking.
    first_only = MappedGame(JaxBroadcastTuple((g1, g2)), lambda vs: vs[0])
    assert first_only(coal).shape == (2, 3)
    assert jnp.array_equal(first_only(coal), jnp.broadcast_to(g1(coal), (2, 3)))

    vec = MappedGame(
        JaxBroadcastTuple((_table_game(2, (2,)),)),
        lambda vs: _VectorValue(jnp.stack([vs[0], -vs[0]], axis=-1)),
    )
    scaled = 2.0 * vec - vec
    assert isinstance(scaled(coal), _VectorValue)
    assert jnp.array_equal(scaled(coal).values, vec(coal).values)
    offset = _VectorValue(jnp.array([[10.0, 20.0], [30.0, 40.0]]))
    shifted = vec + offset
    assert isinstance(shifted(coal), _VectorValue)
    assert jnp.array_equal(shifted(coal).values, vec(coal).values + offset.values)


# --- Direct Möbius & Fourier access and additivity propagation ---


def test_known_additivity_and_direct_moebius_access(monkeypatch):
    assert issubclass(SparseMoebiusGame, KnownAdditivity)
    assert issubclass(SparseFourierGame, KnownAdditivity)

    p = FinitePlayers(3)
    # Permuted singletons: row 0 -> player 2, row 1 -> player 0, row 2 -> player 1.
    am = _additive_moebius(jnp.array([[30.0, 10.0, 20.0], [-3.0, -1.0, -2.0]]), p, perm=[2, 0, 1])
    sm_1 = SparseMoebiusGame(
        values=jnp.array([[5.0, 3.0, 1.0], [7.0, -2.0, 4.0]]),
        interactions=jnp.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=bool),
        players=p,
        additivity=1,
    )
    sm_0 = SparseMoebiusGame(
        values=jnp.array([9.0]),
        interactions=jnp.array([[0, 0, 0]], dtype=bool),
        players=p,
        additivity=0,
    )
    sm_2 = SparseMoebiusGame(
        values=jnp.array([2.0, 4.0, 6.0]),
        interactions=jnp.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
        players=p,
        additivity=2,
    )

    def fail_eval(*args, **kwargs):
        msg = "eval must not be called for direct Möbius coefficient access"
        raise AssertionError(msg)

    monkeypatch.setattr(FinitePlayerGame, "eval", fail_eval)

    # Canonical player order on 1-additive SparseMoebiusGame.
    assert jnp.array_equal(am.contributions(), jnp.array([[10.0, 20.0, 30.0], [-1.0, -2.0, -3.0]]))
    assert jnp.array_equal(am.empty_value(), jnp.array([0.0, 0.0]))
    assert jnp.array_equal(am.coefficients(order=0), jnp.array([[0.0], [0.0]]))
    assert jnp.array_equal(am.coefficients(order=1), am.contributions())
    assert jnp.array_equal(am.coefficients(order=2), jnp.zeros((2, 3)))
    assert am.coefficients(order=4).shape == (2, 0)

    assert sm_1.additivity() == 1
    assert jnp.array_equal(sm_1.empty_value(), jnp.array([5.0, 7.0]))
    assert jnp.array_equal(sm_1.contributions(), jnp.array([[1.0, 0.0, 3.0], [4.0, 0.0, -2.0]]))

    assert sm_0.additivity() == 0
    assert jnp.array_equal(sm_0.empty_value(), jnp.array(9.0))
    assert jnp.array_equal(sm_0.contributions(), jnp.zeros(3))

    # AffineGame of 1-additive games (including division by scalar).
    combo_add = (2.0 * am - sm_1 + 10.0) / 2.0
    assert isinstance(combo_add, ArithmeticGame)
    assert combo_add.additivity() == 1
    assert jnp.allclose(combo_add.empty_value(), jnp.array([2.5, 1.5]))
    assert jnp.allclose(
        combo_add.contributions(),
        jnp.array([[9.5, 20.0, 28.5], [-3.0, -2.0, -2.0]]),
    )

    # Mixed-order batch + packed uint8 with dirty padding bits.
    mixed = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=bool)
    packed_mixed = pack_bool_coalition(mixed) | jnp.uint8(0b11111000)
    coeffs_bool = combo_add.coefficients(mixed[:, None])
    coeffs_packed = combo_add.coefficients(packed_mixed[:, None])
    expected_mixed = jnp.array(
        [
            [2.5, 1.5],
            [9.5, -3.0],
            [20.0, -2.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    assert jnp.allclose(coeffs_bool, expected_mixed)
    assert jnp.allclose(coeffs_packed, expected_mixed)

    # AffineGame of mixed 1-additive and 2-additive has known additivity 2.
    combo_k = 2.0 * sm_2 + am[0] + 3.0
    assert isinstance(combo_k, KnownAdditivity)
    assert combo_k.additivity() == 2
    assert not isinstance(_table_game(3, ()), KnownAdditivity)
    assert not isinstance(2.0 * _table_game(3, ()) + 1.0, KnownAdditivity)
    assert jnp.allclose(combo_k.empty_value(), 7.0)
    assert jnp.allclose(combo_k.coefficients(order=1), jnp.array([10.0, 28.0, 30.0]))
    assert jnp.allclose(combo_k.coefficients(order=2), jnp.array([0.0, 12.0, 0.0]))
    assert jnp.allclose(combo_k.coefficients(order=3), jnp.array([0.0]))
    assert jnp.allclose(combo_k.coefficients(mixed), jnp.array([7.0, 10.0, 28.0, 0.0, 0.0]))

    # Invalid coefficients() arguments.
    with pytest.raises(TypeError, match="either interaction or order"):
        combo_k.coefficients()
    with pytest.raises(TypeError, match="either interaction or order"):
        combo_k.coefficients(mixed, order=1)
    with pytest.raises(TypeError, match="integer"):
        combo_k.coefficients(order=True)
    with pytest.raises(ValueError, match="nonnegative"):
        combo_k.coefficients(order=-1)


def test_product_game_propagates_additivity_lazily_with_capped_bound_and_exact_coefficients(
    monkeypatch,
):
    p = FinitePlayers(3)
    am1 = _additive_moebius(jnp.array([1.0, 2.0, 3.0]), p)
    am2 = _additive_moebius(jnp.array([4.0, -1.0, 2.0]), p) + 5.0
    sm2 = SparseMoebiusGame(
        values=jnp.array([1.0, 2.0, -3.0]),
        interactions=jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=bool),
        players=p,
        additivity=2,
    )

    calls = Mock(wraps=FinitePlayerGame.eval)

    def recording_eval(self, coalition):
        return calls(self, coalition)

    monkeypatch.setattr(FinitePlayerGame, "eval", recording_eval)

    prod_1_1 = am1 * am2
    prod_2_2 = sm2 * sm2
    prod_mixed = am1 * _table_game(3, ())
    assert isinstance(prod_1_1, KnownAdditivity)
    assert isinstance(prod_2_2, KnownAdditivity)
    assert not isinstance(prod_mixed, KnownAdditivity)
    calls.assert_not_called()

    assert isinstance(prod_1_1, ProductGame)
    assert prod_1_1.additivity() == 2
    assert prod_2_2.additivity() == 3  # min(n_players=3, 2 + 2)
    assert type(prod_mixed) is ProductGame
    assert prod_mixed._known_additivity() is None

    all_sets = subset_patterns(3)
    table_vals = prod_1_1(all_sets)
    exact_moebius = _exact_moebius_table(table_vals, 3)

    assert jnp.allclose(prod_1_1.coefficients(all_sets), exact_moebius, atol=1e-5)
    assert jnp.allclose(prod_1_1.coefficients(order=0), exact_moebius[:1], atol=1e-5)
    assert jnp.allclose(
        prod_1_1.coefficients(order=1), exact_moebius[jnp.array([1, 2, 4])], atol=1e-5
    )
    assert jnp.allclose(
        prod_1_1.coefficients(order=2), exact_moebius[jnp.array([3, 5, 6])], atol=1e-5
    )

    calls.reset_mock()
    assert jnp.array_equal(prod_1_1.coefficients(order=3), jnp.zeros(1))
    assert jnp.array_equal(prod_1_1.coefficients(jnp.array([True, True, True])), jnp.array(0.0))
    calls.assert_not_called()

    table = jnp.array([0.0, 1.0, 2.0, 4.0, 3.0, 5.0, 6.0, 10.0])
    chi = (-1.0) ** jnp.sum(all_sets[:, None, :] & all_sets[None, :, :], axis=-1)
    fourier = SparseFourierGame(
        values=table @ chi / 8.0, interactions=all_sets, players=p, additivity=3
    )
    assert jnp.allclose(fourier.coefficients(all_sets), _exact_moebius_table(table, 3), atol=1e-5)


def test_flattening_rely_on_parent_contract_for_subclasses():
    p = FinitePlayers(3)
    am = _additive_moebius(jnp.array([1.0, 2.0, 3.0]), p)

    @dataclass(frozen=True, slots=True, weakref_slot=True)
    class TaggedAffine(AffineGame[Any, Any]):
        tag: str = "aff"

    sub_aff = TaggedAffine(JaxBroadcastTuple((am,)), JaxBroadcastTuple((jnp.array(1.0),)))
    assert type(sub_aff) is TaggedAffine

    coal = jnp.array([True, False, True])
    wrapped_aff = 2.0 * sub_aff + 1.0
    assert wrapped_aff.games.members == (am,)
    assert wrapped_aff.additivity() == 1
    assert jnp.allclose(wrapped_aff(coal), 2.0 * sub_aff(coal) + 1.0)


# --- Dynamic pytree leaves and JIT evaluation & shaping ---


def test_dynamic_leaves_and_jit_evaluation_and_shaping():
    players = FinitePlayers(jnp.array([[0, 1], [0, 2]]), latent_n_players=3)
    t1 = DenseCoalitionGame(jnp.arange(8.0).reshape(2, 4) + 1.0, players)
    t2 = DenseCoalitionGame(jnp.arange(8.0, 16.0).reshape(2, 4), players)
    am = _additive_moebius(jnp.array([[1.0, 2.0], [3.0, 4.0]]), players)
    coal = jnp.array([True, False])

    c_arr = jnp.array([2.0, 5.0])
    aff = 2.0 * t1 - t2 + c_arr
    add_aff = 3.0 * am + c_arr
    prod_k = am * (am + 1.0)
    quot_const = c_arr / t1
    quot_games = t2 / t1

    # Weights, biases, and constants are dynamic leaves (re-using compiled function without retracing).
    for game in (aff, add_aff, prod_k, quot_const, quot_games):
        leaves = jax.tree.leaves(game)
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)

        jitted_eval = jax.jit(lambda g, c: g(c))
        assert jnp.allclose(jitted_eval(game, coal), game(coal))

        jitted_slice = jax.jit(lambda g: g[1])
        sliced = jitted_slice(game)
        assert sliced.shape == ()
        assert jnp.allclose(sliced(coal), game[1](coal))

        jitted_reshape = jax.jit(lambda g: g.reshape(1, 2))
        reshaped = jitted_reshape(game)
        assert reshaped.shape == (1, 2)
        assert jnp.allclose(reshaped(coal), game(coal).reshape(1, 2))

    # JIT coefficient extraction on mixed-order batch (no value-dependent Python branching).
    mixed = jnp.array([[0, 0], [1, 0], [1, 1]], dtype=bool)
    assert jnp.allclose(
        jax.jit(lambda g, m: g.coefficients(m))(add_aff, mixed[:, None]),
        add_aff.coefficients(mixed[:, None]),
    )
    assert jnp.allclose(
        jax.jit(lambda g, m: g.coefficients(m))(prod_k, mixed[:, None]),
        prod_k.coefficients(mixed[:, None]),
    )

    # Constant in c / g is dynamic under JIT (no captured tracer).
    jitted_div = jax.jit(lambda c, g, s: (c / g)(s))
    assert jnp.allclose(jitted_div(c_arr, t1, coal), c_arr / t1(coal))
    assert jnp.allclose(
        jitted_div(jnp.array([10.0, 20.0]), t1, coal), jnp.array([10.0, 20.0]) / t1(coal)
    )

    # Invalid coalition batch still rejected on arithmetic games.
    with pytest.raises((TypeCheckError, ValueError)):
        aff(jnp.ones((3, 2), dtype=bool))


# --- Regression tests for JIT player validation, unsigned/division/MappedGame, structured values, nesting ---


def test_jit_construction_rejects_incompatible_distinct_traced_players():
    p1 = FinitePlayers(jnp.array([0, 2]), latent_n_players=4)
    p2 = FinitePlayers(jnp.array([0, 3]), latent_n_players=4)
    g1 = DenseCoalitionGame(jnp.arange(4.0), p1)
    g2 = DenseCoalitionGame(jnp.ones(4), p2)
    coal = jnp.array([True, False])

    with pytest.raises(TypeError, match="traced player IDs"):
        jax.jit(lambda a, b: (a + b)(coal))(g1, g2)
    with pytest.raises(TypeError, match="traced player IDs"):
        jax.jit(lambda a, b: (a * b)(coal))(g1, g2)

    # Same player object or single-game ops inside JIT still work.
    assert jnp.allclose(jax.jit(lambda a: (a + a)(coal))(g1), 2.0 * g1(coal))
    assert jnp.allclose(
        jax.jit(lambda a, b: (2.0 * a + b)(coal))(g1, jnp.array(3.0)), 2.0 * g1(coal) + 3.0
    )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _IntDenseCoalitionGame(
    CoalitionDefinedGame[Any, jax.Array], JaxAxisProtected["_IntDenseCoalitionGame"]
):
    values: jax.Array
    _players: FinitePlayers[Any]
    protected_axes: ClassVar[dict[str, int]] = {"values": 1, "_players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"_players"}

    @property
    @override
    def players(self) -> FinitePlayers[Any]:
        return self._players

    @override
    def _coalition_bool(self, coalition: jax.Array) -> jax.Array:
        weights = 1 << jnp.arange(self.n_players, dtype=jnp.int32)
        idx = jnp.sum(coalition.astype(jnp.int32) * weights, axis=-1)
        shape = jnp.broadcast_shapes(self.shape, idx.shape)
        table = jnp.broadcast_to(self.values, (*shape, self.values.shape[-1]))
        return jnp.take_along_axis(table, jnp.broadcast_to(idx, shape)[..., None], axis=-1)[..., 0]


def test_exact_integer_cast_preserves_additivity():
    class IntegerAdditiveGame(_IntDenseCoalitionGame, KnownAdditivity[Any, jax.Array]):
        _additivity: int = 1

        def contributions(self):
            return self.values[..., 1:] - self.values[..., :1]

    child = IntegerAdditiveGame(jnp.array([1, 4], dtype=jnp.int8), FinitePlayers(1))
    converted = child.astype(jnp.int32)
    assert converted.additivity() == 1
    assert converted.full_value().dtype == jnp.int32
    assert jnp.array_equal(converted.contributions(), jnp.array([3], dtype=jnp.int32))
    assert jnp.array_equal(converted.coefficients(order=1), converted.contributions())


def test_unsigned_negation_subtraction_and_mixed_dtypes_preserve_jax_semantics():
    p = FinitePlayers(2)
    u1 = _IntDenseCoalitionGame(jnp.array([0, 1, 5, 255], dtype=jnp.uint8), p)
    u2 = _IntDenseCoalitionGame(jnp.array([1, 2, 3, 10], dtype=jnp.uint8), p)
    i1 = _IntDenseCoalitionGame(jnp.array([0, 1, 5, 10], dtype=jnp.int8), p)
    f1 = DenseCoalitionGame(jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32), p)
    all_sets = subset_patterns(2)

    cases = [
        (-u1, -u1(all_sets)),
        (u1 - u2, u1(all_sets) - u2(all_sets)),
        (u1 - 1, u1(all_sets) - 1),
        (1 - u1, 1 - u1(all_sets)),
        (u1 - jnp.uint8(1), u1(all_sets) - jnp.uint8(1)),
        (jnp.uint8(1) - u1, jnp.uint8(1) - u1(all_sets)),
        (f1 - u1, f1(all_sets) - u1(all_sets)),
        (u1 - f1, u1(all_sets) - f1(all_sets)),
        (f1 - jnp.uint8(1), f1(all_sets) - jnp.uint8(1)),
        (i1 - u1, i1(all_sets) - u1(all_sets)),
        (2 * (-u1), 2 * (-u1(all_sets))),
        (2.0 * (-u1), 2.0 * (-u1(all_sets))),
        (f1 + (-u1), f1(all_sets) + (-u1(all_sets))),
    ]
    for expr, expected in cases:
        actual = expr(all_sets)
        assert expr.dtype == expected.dtype
        assert actual.dtype == expected.dtype
        assert jnp.array_equal(actual, expected)


def test_affine_flattening_preserves_integer_overflow_under_promotion_and_flattens_same_ring():
    p = FinitePlayers(2)
    f1 = DenseCoalitionGame(jnp.full(4, 1.0, dtype=jnp.float32), p)
    f2 = DenseCoalitionGame(jnp.full(4, 2.0, dtype=jnp.float32), p)
    u5 = _IntDenseCoalitionGame(jnp.full(4, 5, dtype=jnp.uint8), p)
    u3 = _IntDenseCoalitionGame(jnp.full(4, 3, dtype=jnp.uint8), p)
    i125 = _IntDenseCoalitionGame(jnp.full(4, 125, dtype=jnp.int8), p)
    i5 = _IntDenseCoalitionGame(jnp.full(4, 5, dtype=jnp.int8), p)
    i1 = _IntDenseCoalitionGame(jnp.full(4, 1, dtype=jnp.int8), p)
    i16 = _IntDenseCoalitionGame(jnp.full(4, 1, dtype=jnp.int16), p)
    coal = jnp.array([True, False])

    # Integer subexpressions widened by parent dtype stay unflattened so wraparound matches JAX.
    widened_cases = [
        (f1 + (u5 - u3), f1(coal) + (u5(coal) - u3(coal))),
        (2.0 * (i125 + i5), 2.0 * (i125(coal) + i5(coal))),
        (i16 + (i125 + i5), i16(coal) + (i125(coal) + i5(coal))),
        (jnp.int32(2) * (i125 + i5), jnp.int32(2) * (i125(coal) + i5(coal))),
    ]
    for expr, expected in widened_cases:
        actual = expr(coal)
        assert expr.dtype == expected.dtype
        assert actual.dtype == expected.dtype
        assert jnp.array_equal(actual, expected)

    # Mixed manually constructed integer term inside a float affine game is also preserved.
    mixed_inner = AffineGame(
        JaxBroadcastTuple((f1, i5)),
        JaxBroadcastTuple((jnp.float32(1.0), jnp.int8(100))),
    )
    mixed_outer = 2.0 * mixed_inner
    assert mixed_outer.games.members == (mixed_inner,)
    assert jnp.array_equal(mixed_outer(coal), 2.0 * (f1(coal) + jnp.int8(100) * i5(coal)))

    # Same-ring integer affine combinations still flatten and match modular wraparound.
    same_int8 = (i125 + i5) + i1
    assert same_int8.games.members == (i125, i5, i1)
    assert jnp.array_equal(same_int8(coal), (i125(coal) + i5(coal)) + i1(coal))

    same_uint8 = 2 * (u5 - u3)
    assert same_uint8.games.members == (u5, u3)
    assert jnp.array_equal(same_uint8(coal), 2 * (u5(coal) - u3(coal)))

    # Ordinary floating-point flattening with Python int weights still flattens cleanly.
    ordinary_f32 = 3 * (2 * f1 + 5) - 4 * f2 + 7
    assert ordinary_f32.games.members == (f1, f2)
    assert jnp.array_equal(ordinary_f32(coal), 3 * (2 * f1(coal) + 5) - 4 * f2(coal) + 7)


def test_direct_true_divide_avoids_reciprocal_underflow_and_matches_mixed_dtypes():
    p = FinitePlayers(2)
    big = jnp.float32(1e38)
    g_big = DenseCoalitionGame(jnp.full(4, big, dtype=jnp.float32), p)
    coal = jnp.array([True, False])

    assert jnp.allclose((g_big / big)(coal), jnp.float32(1.0))
    assert jnp.allclose((big / g_big)(coal), jnp.float32(1.0))
    assert jnp.allclose((g_big / g_big)(coal), jnp.float32(1.0))

    # Batched 1e38 game divided by batched 1e38 constant also avoids reciprocal underflow.
    g_big_b = DenseCoalitionGame(jnp.full((2, 4), big, dtype=jnp.float32), p)
    big_b = jnp.array([big, big], dtype=jnp.float32)
    assert jnp.allclose((g_big_b / big_b)(coal), jnp.ones(2, dtype=jnp.float32))
    assert jnp.allclose((big_b / g_big_b)(coal), jnp.ones(2, dtype=jnp.float32))

    # Additive game division by nonzero 1e38 preserves known additivity and avoids reciprocal underflow.
    am_big = _additive_moebius(jnp.array([big, 2.0 * big], dtype=jnp.float32), p)
    scaled_am = am_big / big
    assert scaled_am.additivity() == 1
    assert jnp.allclose(scaled_am.empty_value(), jnp.float32(0.0))
    assert jnp.allclose(scaled_am.coefficients(jnp.array([True, False])), jnp.float32(1.0))
    assert jnp.allclose(scaled_am.coefficients(jnp.array([False, True])), jnp.float32(2.0))
    assert jnp.allclose(scaled_am(jnp.array([True, False])), jnp.float32(1.0))
    assert jnp.allclose(scaled_am(jnp.array([False, True])), jnp.float32(2.0))
    assert jnp.allclose(scaled_am.contributions(), jnp.true_divide(am_big.contributions(), big))
    am_norm = _additive_moebius(jnp.array([1e30, 2e30], dtype=jnp.float32), p)
    assert jnp.allclose((am_norm / jnp.float32(1e30)).contributions(), jnp.array([1.0, 2.0]))

    # Division by concrete zero does NOT propagate known additivity.
    div_zero = am_big / 0.0
    assert div_zero._known_additivity() is None
    assert jnp.isnan(div_zero(jnp.array([False, False])))

    # Under JIT: concrete nonzero scalar and captured concrete JAX scalar work without tracing errors.
    assert jnp.allclose(
        jax.jit(lambda g: (g / 2.0).contributions())(am_big), am_big.contributions() / 2.0
    )
    captured_d = jnp.array(2.0, dtype=jnp.float32)
    captured_zero = jnp.array(0.0, dtype=jnp.float32)

    @jax.jit
    def jit_captured_div(c: jax.Array) -> jax.Array:
        return (am_big / captured_d)(c)

    @jax.jit
    def jit_captured_contribs() -> jax.Array:
        return (am_big / captured_d).contributions()

    @jax.jit
    def jit_captured_zero(c: jax.Array) -> jax.Array:
        return (am_big / captured_zero)(c)

    assert jnp.allclose(jit_captured_div(coal), am_big(coal) / captured_d)
    assert jnp.allclose(jit_captured_contribs(), am_big.contributions() / captured_d)
    assert jnp.isnan(jit_captured_zero(jnp.array([False, False])))
    jit_div = jax.jit(lambda g, c: (g / c)(coal))
    assert jnp.allclose(jit_div(g_big, big), jnp.float32(1.0))
    assert jnp.allclose(jit_div(g_big, jnp.float32(2.0)), big / jnp.float32(2.0))

    # Static nonzero proof uses normalized JAX dtype: underflow-to-zero / non-finite never propagate known additivity.
    with jax.enable_x64(new_val=False):
        div_underflow = am_big / 1e-50
        assert div_underflow._known_additivity() is None
        assert jnp.isnan(div_underflow(jnp.array([False, False])))

        @jax.jit
        def jit_underflow(c: jax.Array) -> jax.Array:
            return (am_big / 1e-50)(c)

        assert jnp.isnan(jit_underflow(jnp.array([False, False])))
        assert (am_big / np.array([1e-50, 2.0]))._known_additivity() is None
        assert (am_big / float("nan"))._known_additivity() is None
        assert (am_big / float("inf"))._known_additivity() is None
    with jax.enable_x64(new_val=True):
        assert (am_big / 1e-50)._known_additivity() is None
        assert jnp.isnan((am_big / 1e-50)(jnp.array([False, False])))
        am_f64 = _additive_moebius(jnp.array([1.0, 2.0], dtype=jnp.float64), p)
        assert (am_f64 / 1e-50)._known_additivity() == 1
        assert (am_big / jnp.float64(1e-50))._known_additivity() == 1

    # Mixed dtype semantics match jnp.true_divide.
    u1 = _IntDenseCoalitionGame(jnp.array([2, 4, 6, 8], dtype=jnp.uint8), p)
    u2 = _IntDenseCoalitionGame(jnp.array([1, 2, 3, 4], dtype=jnp.uint8), p)
    f16 = DenseCoalitionGame(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float16), p)
    for expr, expected in [
        (u1 / u2, u1(coal) / u2(coal)),
        (u1 / 2, u1(coal) / 2),
        (2 / u2, 2 / u2(coal)),
        (f16 / jnp.int32(2), f16(coal) / jnp.int32(2)),
    ]:
        assert expr.dtype == expected.dtype
        assert expr(coal).dtype == expected.dtype
        assert jnp.allclose(expr(coal), expected)


def test_mapped_game_dtype_contract_is_lazy_accurate_and_numeric(monkeypatch):
    g_f32 = _table_game(2, (2,))
    coal = jnp.array([True, False])

    calls = Mock(wraps=DenseCoalitionGame.eval)

    def recording_eval(self, c):
        return calls(self, c)

    monkeypatch.setattr(DenseCoalitionGame, "eval", recording_eval)

    # Arbitrary dtype-changing map (float32 -> complex64) infers dtype lazily without calling eval.
    c_game = MappedGame(JaxBroadcastTuple((g_f32,)), lambda vs: vs[0].astype(jnp.complex64) + 1j)
    calls.assert_not_called()
    assert c_game.dtype == jnp.complex64
    res = c_game(coal)
    assert res.dtype == jnp.complex64
    assert jnp.allclose(res, g_f32(coal).astype(jnp.complex64) + 1j)

    # Non-numeric mapped outputs and explicit non-numeric output_dtype are rejected lazily.
    with pytest.raises(TypeError, match="numeric"):
        MappedGame(JaxBroadcastTuple((g_f32,)), lambda vs: vs[0] > 0)
    with pytest.raises(TypeError, match="numeric"):
        MappedGame(JaxBroadcastTuple((g_f32,)), lambda vs: vs[0], output_dtype=jnp.bool_)

    # Explicit output_dtype works for structured-input fn that cannot be traced from array specs.
    vec_game = MappedGame(
        JaxBroadcastTuple((g_f32,)),
        lambda vs: _VectorValue(jnp.stack([vs[0], 2.0 * vs[0]], axis=-1)),
    )
    calls.reset_mock()
    mapped_vec = MappedGame(
        JaxBroadcastTuple((vec_game,)),
        lambda vs: _VectorValue(vs[0].values.astype(jnp.complex64)),
        output_dtype=jnp.complex64,
    )
    calls.assert_not_called()
    assert mapped_vec.dtype == jnp.complex64
    assert mapped_vec(coal).dtype == jnp.complex64

    # Mismatched explicit output_dtype raises at eval time.
    lying = MappedGame(
        JaxBroadcastTuple((g_f32,)),
        lambda vs: vs[0].astype(jnp.complex64),
        output_dtype=jnp.float32,
    )
    with pytest.raises(TypeError, match="expected float32"):
        lying(coal)


@dataclass(frozen=True)
class _StructuredPair(JaxAxisProtected["_StructuredPair"]):
    count: jax.Array
    vector: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"count": 0, "vector": 1}
    permitted_functions: ClassVar[set] = {
        jf.jax_add,
        jf.jax_multiply,
        jf.jax_sum,
        jf.jax_true_divide,
        jf.jax_where,
    }


def test_same_dtype_cast_applies_to_all_mapped_value_fields():
    table = _table_game(batch_shape=())
    mapped = MappedGame(
        JaxBroadcastTuple((table,)),
        lambda vs: _StructuredPair(
            vs[0].astype(jnp.float32), jnp.stack([vs[0], vs[0]], axis=-1).astype(jnp.float16)
        ),
        output_dtype=jnp.float32,
    )
    assert mapped.full_value().vector.dtype == jnp.float16
    converted = mapped.astype(jnp.float32)
    assert isinstance(converted, CastGame)
    value = converted.full_value()
    assert value.count.dtype == value.vector.dtype == jnp.float32


@pytest.mark.parametrize("structured", [False, True])
def test_explicit_mapped_dtype_strengthens_weak_result(structured):
    table = _table_game(batch_shape=())
    with jax.enable_x64(new_val=False):

        def fn(vs):
            shape = vs[0].shape
            return VectorValue(jnp.full((*shape, 2), 3.0)) if structured else jnp.full(shape, 3.0)

        mapped = MappedGame(JaxBroadcastTuple((table,)), fn, output_dtype=jnp.float32)
        assert not mapped.weak_type
        assert not mapped.full_value().weak_type


@dataclass(frozen=True)
class _VectorAdditiveGame(
    CoalitionDefinedGame[Any, _VectorValue],
    KnownAdditivity[Any, _VectorValue],
    JaxAxisProtected["_VectorAdditiveGame"],
):
    empty: _VectorValue
    contribs: jax.Array
    _players: FinitePlayers[Any]
    _additivity: int = 1
    protected_axes: ClassVar[dict[str, int]] = {"empty": 0, "contribs": 2, "_players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"_players"}

    @property
    @override
    def players(self) -> FinitePlayers[Any]:
        return self._players

    @override
    def empty_value(self) -> _VectorValue:
        return self.empty

    @override
    def contributions(self) -> _VectorValue:
        if self.n_players == 0 or self._additivity == 0:
            return _VectorValue(
                jnp.zeros(
                    (*self.shape, self.n_players, self.empty.values.shape[-1]),
                    dtype=self.empty.values.dtype,
                )
            )
        return _VectorValue(self.contribs)

    @override
    def _coalition_bool(self, coalition: jax.Array) -> _VectorValue:
        batch = jnp.broadcast_shapes(self.shape, coalition.shape[:-1])
        empty_v = jnp.broadcast_to(self.empty.values, (*batch, self.empty.values.shape[-1]))
        contribs_v = jnp.broadcast_to(
            self.contribs, (*batch, self.n_players, self.empty.values.shape[-1])
        )
        mask = jnp.broadcast_to(coalition, (*batch, self.n_players))[..., None]
        return _VectorValue(empty_v + jnp.sum(jnp.where(mask, contribs_v, 0.0), axis=-2))


def test_structured_values_batch_weights_bias_divisors_and_kadditive_zero_shortcuts():
    p = FinitePlayers(2)
    # Use protected width == batch size (2) to catch silent fieldwise misbroadcasting!
    vec_game = MappedGame(
        JaxBroadcastTuple((_table_game(2, (2,)),)),
        lambda vs: _VectorValue(jnp.stack([vs[0], vs[0] + 10.0], axis=-1)),
    )
    pair_child = _table_game(2, (2,))
    pair_game = MappedGame(
        JaxBroadcastTuple((pair_child,)),
        lambda vs: _StructuredPair(vs[0], jnp.stack([vs[0], vs[0] + 1.0, vs[0] + 2.0], axis=-1)),
        output_dtype=pair_child.dtype,
    )
    coal = jnp.array([True, False])
    w = jnp.array([10.0, 100.0])
    b = jnp.array([1.0, 2.0])
    d = jnp.array([2.0, 4.0])

    # Raw batch weights, bias, and divisors apply along logical batch axis, NOT protected axis.
    aff_vec = w * vec_game + b
    res_vec = aff_vec(coal)
    assert isinstance(res_vec, _VectorValue)
    assert jnp.allclose(res_vec.values, w[:, None] * vec_game(coal).values + b[:, None])

    div_pair = pair_game / d
    res_pair = div_pair(coal)
    assert isinstance(res_pair, _StructuredPair)
    assert jnp.allclose(res_pair.count, pair_game(coal).count / d)
    assert jnp.allclose(res_pair.vector, pair_game(coal).vector / d[:, None])

    rdiv_pair = d / (pair_game + 1.0)
    res_rdiv = rdiv_pair(coal)
    assert isinstance(res_rdiv, _StructuredPair)
    assert jnp.allclose(res_rdiv.count, d / (pair_game(coal).count + 1.0))
    assert jnp.allclose(res_rdiv.vector, d[:, None] / (pair_game(coal).vector + 1.0))

    # KnownAdditivity zero shortcuts preserve _VectorValue representation.
    v_empty = _VectorValue(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    v_contribs = jnp.array([[[10.0, 20.0], [30.0, 40.0]], [[5.0, 6.0], [7.0, 8.0]]])
    v_add = _VectorAdditiveGame(empty=v_empty, contribs=v_contribs, _players=p)

    above_order = v_add.coefficients(order=2)
    assert isinstance(above_order, _VectorValue)
    assert above_order.shape == (2, 1)
    assert above_order.values.shape == (2, 1, 2)
    assert jnp.array_equal(above_order.values, jnp.zeros((2, 1, 2)))

    empty_order = v_add.coefficients(order=5)
    assert isinstance(empty_order, _VectorValue)
    assert empty_order.shape == (2, 0)
    assert empty_order.values.shape == (2, 0, 2)

    above_explicit = v_add.coefficients(jnp.array([True, True]))
    assert isinstance(above_explicit, _VectorValue)
    assert above_explicit.shape == (2,)
    assert jnp.array_equal(above_explicit.values, jnp.zeros((2, 2)))

    empty_explicit = v_add.coefficients(jnp.zeros((0, 1, 2), dtype=bool))
    assert isinstance(empty_explicit, _VectorValue)
    assert empty_explicit.shape == (0, 2)
    assert empty_explicit.values.shape == (0, 2, 2)

    mixed_explicit = v_add.coefficients(jnp.array([[0, 0], [1, 0], [1, 1]], dtype=bool)[:, None])
    assert isinstance(mixed_explicit, _VectorValue)
    assert jnp.allclose(mixed_explicit.values[0], v_empty.values)
    assert jnp.allclose(mixed_explicit.values[1], v_contribs[:, 0, :])
    assert jnp.allclose(mixed_explicit.values[2], jnp.zeros((2, 2)))

    # Zero-player structured game has additivity 0 and preserves _VectorValue.
    v_zero = _VectorAdditiveGame(
        empty=v_empty, contribs=jnp.zeros((2, 0, 2)), _players=FinitePlayers(0), _additivity=0
    )
    assert v_zero.additivity() == 0
    zero_contribs = v_zero.contributions()
    assert isinstance(zero_contribs, _VectorValue)
    assert zero_contribs.shape == (2, 0)
    assert zero_contribs.values.shape == (2, 0, 2)
    zero_order1 = v_zero.coefficients(order=1)
    assert isinstance(zero_order1, _VectorValue)
    assert zero_order1.shape == (2, 0)
    assert zero_order1.values.shape == (2, 0, 2)
    assert (2.0 * v_zero + 1.0).additivity() == 0
    assert (v_zero / 2.0).additivity() == 0

    # Structured 0-additive game with inf/nan empty values must not pollute nonzero-order coeffs.
    const_inf = _VectorAdditiveGame(
        empty=_VectorValue(jnp.array([[jnp.inf, jnp.nan], [-jnp.inf, 1.0]], dtype=jnp.float32)),
        contribs=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        _players=p,
        _additivity=0,
    )
    singleton = jnp.array([True, False])
    for got in (
        const_inf.coefficients(singleton),
        jax.jit(lambda g, c: g.coefficients(c))(const_inf, singleton),
    ):
        assert isinstance(got, _VectorValue)
        assert jnp.array_equal(got.values, jnp.zeros((2, 2), dtype=jnp.float32))
    order1_inf = const_inf.coefficients(order=1)
    assert isinstance(order1_inf, _VectorValue)
    assert jnp.array_equal(order1_inf.values, jnp.zeros((2, 2, 2), dtype=jnp.float32))

    # Structured Möbius inversion must not leak inf/nan from discarded subsets eagerly or under jit.
    p3_inf = FinitePlayers(3)
    v_inf_add = _VectorAdditiveGame(
        empty=_VectorValue(jnp.array([jnp.inf, 5.0], dtype=jnp.float32)),
        contribs=jnp.zeros((3, 2), dtype=jnp.float32),
        _players=p3_inf,
    )
    v_inf_sq = v_inf_add**2
    assert v_inf_sq.additivity() == 2
    empty_and_full = jnp.array([[0, 0, 0], [1, 1, 1]], dtype=bool)
    for got_sq in (
        v_inf_sq.coefficients(empty_and_full),
        jax.jit(lambda g, q: g.coefficients(q))(v_inf_sq, empty_and_full),
    ):
        assert isinstance(got_sq, _VectorValue)
        assert jnp.array_equal(got_sq.values[0], jnp.array([jnp.inf, 25.0], dtype=jnp.float32))
        assert jnp.array_equal(got_sq.values[1], jnp.array([0.0, 0.0], dtype=jnp.float32))

    # 0-additive float32 game + mixed-dtype structured bias promotes vector field consistently.
    const_m = SparseMoebiusGame(
        values=jnp.array([1.0], dtype=jnp.float32),
        interactions=jnp.zeros((1, 2), dtype=bool),
        players=p,
        additivity=0,
    )
    mixed_bias = _StructuredPair(
        count=jnp.array(2.0, dtype=jnp.float32),
        vector=jnp.array([3, 4], dtype=jnp.int16),
    )
    shifted = const_m + mixed_bias
    assert shifted.additivity() == 0
    eval_out = shifted(singleton)
    explicit_out = shifted.coefficients(singleton)
    contribs_out = shifted.contributions()
    order1_out = shifted.coefficients(order=1)
    for out in (eval_out, explicit_out, contribs_out, order1_out):
        assert isinstance(out, _StructuredPair)
        assert out.count.dtype == jnp.float32
        assert out.vector.dtype == jnp.float32

    # Structured bias with inf/nan must not pollute non-empty Möbius coefficients.
    inf_bias = _VectorValue(jnp.array([jnp.inf, jnp.nan], dtype=jnp.float32))
    shifted_inf = const_m + inf_bias
    assert shifted_inf.additivity() == 0
    got_nonempty = shifted_inf.coefficients(singleton)
    assert isinstance(got_nonempty, _VectorValue)
    assert jnp.array_equal(got_nonempty.values, jnp.zeros((2,), dtype=jnp.float32))

    # Mixed float32/int16 structured Möbius inversion matches per-field dtypes.
    @dataclass(frozen=True)
    class _PairAdditiveGame(
        CoalitionDefinedGame[Any, _StructuredPair],
        KnownAdditivity[Any, _StructuredPair],
        JaxAxisProtected["_PairAdditiveGame"],
    ):
        empty: _StructuredPair
        contribs: _StructuredPair
        _players: FinitePlayers[Any]
        _additivity: int = 1
        protected_axes: ClassVar[dict[str, int]] = {"empty": 0, "contribs": 1, "_players": 0}
        broadcasted_batch: ClassVar[set[str]] = {"_players"}

        @property
        @override
        def players(self) -> FinitePlayers[Any]:
            return self._players

        @override
        def empty_value(self) -> _StructuredPair:
            return self.empty

        @override
        def contributions(self) -> _StructuredPair:
            return self.contribs

        @override
        def _coalition_bool(self, coalition: jax.Array) -> _StructuredPair:
            batch = jnp.broadcast_shapes(self.shape, coalition.shape[:-1])
            mask = jnp.broadcast_to(coalition, (*batch, self.n_players))
            c = jnp.broadcast_to(self.contribs.count, (*batch, self.n_players))
            v = jnp.broadcast_to(
                self.contribs.vector, (*batch, self.n_players, self.empty.vector.shape[-1])
            )
            return _StructuredPair(
                count=jnp.broadcast_to(self.empty.count, batch)
                + jnp.sum(jnp.where(mask, c, 0.0), axis=-1),
                vector=jnp.broadcast_to(self.empty.vector, (*batch, self.empty.vector.shape[-1]))
                + jnp.sum(jnp.where(mask[..., None], v, jnp.int16(0)), axis=-2, dtype=jnp.int16),
            )

    p3_mix = FinitePlayers(3)
    pair_add = _PairAdditiveGame(
        empty=_StructuredPair(
            count=jnp.array(1.0, dtype=jnp.float32),
            vector=jnp.array([2, 3], dtype=jnp.int16),
        ),
        contribs=_StructuredPair(
            count=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
            vector=jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.int16),
        ),
        _players=p3_mix,
    )
    pair_prod = pair_add * pair_add
    assert pair_prod.additivity() == 2
    out_of_bound_q = jnp.array([True, True, True])
    in_bound_q = jnp.array([True, True, False])
    empty_q = jnp.zeros((0, 3), dtype=bool)
    for res in (
        pair_prod.coefficients(order=0),
        pair_prod.coefficients(order=1),
        pair_prod.coefficients(order=2),
        pair_prod.coefficients(order=3),
        pair_prod.coefficients(order=4),
        pair_prod.coefficients(in_bound_q),
        pair_prod.coefficients(out_of_bound_q),
        jax.jit(lambda g, q: g.coefficients(q))(pair_prod, out_of_bound_q),
        pair_prod.coefficients(empty_q),
    ):
        assert isinstance(res, _StructuredPair)
        assert res.count.dtype == jnp.float32
        assert res.vector.dtype == jnp.sum(jnp.int16(0)).dtype

    v_prod = v_add * v_add
    assert v_prod.additivity() == 2
    prod_above = v_prod.coefficients(order=3)
    assert isinstance(prod_above, _VectorValue)
    assert prod_above.values.shape == (2, 0, 2)
    prod_c2 = v_prod.coefficients(jnp.array([True, True]))
    assert isinstance(prod_c2, _VectorValue)
    assert jnp.allclose(
        prod_c2.values,
        2.0 * v_contribs[:, 0, :] * v_contribs[:, 1, :],
    )

    v_aff = w * v_add + b
    assert v_aff.additivity() == 1
    assert isinstance(v_aff.coefficients(order=2), _VectorValue)
    assert jnp.allclose(v_aff.contributions().values, w[:, None, None] * v_contribs)

    p3 = FinitePlayers(3)
    s_add3 = _additive_moebius(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), p3)
    v_add3 = _VectorAdditiveGame(
        empty=v_empty,
        contribs=jnp.ones((2, 3, 2), dtype=jnp.float32),
        _players=p3,
    )
    coal3 = jnp.array([True, False, True])
    for mixed_aff in (s_add3 + v_add3, v_add3 + s_add3):
        assert mixed_aff.additivity() == 1
        assert isinstance(mixed_aff(coal3), _VectorValue)
        above = mixed_aff.coefficients(order=2)
        assert isinstance(above, _VectorValue)
        assert above.values.shape == (2, 3, 2)
        assert jnp.array_equal(above.values, jnp.zeros((2, 3, 2)))
        explicit_above = mixed_aff.coefficients(jnp.array([True, True, False]))
        assert isinstance(explicit_above, _VectorValue)
        assert jnp.array_equal(explicit_above.values, jnp.zeros((2, 2)))
    for mixed_prod in (s_add3 * v_add3, v_add3 * s_add3):
        assert mixed_prod.additivity() == 2
        assert isinstance(mixed_prod(coal3), _VectorValue)
        above = mixed_prod.coefficients(order=3)
        assert isinstance(above, _VectorValue)
        assert above.values.shape == (2, 1, 2)
        assert jnp.array_equal(above.values, jnp.zeros((2, 1, 2)))
        explicit_above = mixed_prod.coefficients(jnp.array([True, True, True]))
        assert isinstance(explicit_above, _VectorValue)
        assert jnp.array_equal(explicit_above.values, jnp.zeros((2, 2)))


def test_explicit_single_coefficient_delegates_without_materializing_contributions(monkeypatch):
    p = FinitePlayers(4)
    am1 = _additive_moebius(jnp.array([1.0, 2.0, 3.0, 4.0]), p)
    am2 = _additive_moebius(jnp.array([5.0, 6.0, 7.0, 8.0]), p)
    aff_add = 2.0 * am1 - am2 + 10.0
    div_add = aff_add / 2.0

    def fail_contributions(*args, **kwargs):
        msg = "contributions() must not be materialized for explicit interaction lookup"
        raise AssertionError(msg)

    monkeypatch.setattr(SparseMoebiusGame, "contributions", fail_contributions)
    monkeypatch.setattr(FinitePlayerGame, "contributions", fail_contributions)

    single = jnp.array([False, True, False, False])
    assert jnp.allclose(aff_add.coefficients(single), jnp.array(-2.0))
    assert jnp.allclose(div_add.coefficients(single), jnp.array(-1.0))
    assert jnp.allclose(aff_add.coefficients(p.empty_coalition()), jnp.array(10.0))
    assert jnp.allclose(div_add.coefficients(p.empty_coalition()), jnp.array(5.0))


def test_product_game_preserves_nesting_and_multiplication_order():
    p = FinitePlayers(2)
    g_big = DenseCoalitionGame(jnp.full(4, jnp.float32(1e25), dtype=jnp.float32), p)
    g_small = DenseCoalitionGame(jnp.full(4, jnp.float32(1e-25), dtype=jnp.float32), p)
    coal = jnp.array([True, False])

    inner = g_big * g_small
    left_nested = inner * g_big
    right_nested = g_big * inner
    assert left_nested.games.members == (inner, g_big)
    assert right_nested.games.members == (g_big, inner)

    # Order matters in float32: (1e25 * 1e-25) * 1e25 = 1e25, while (1e25 * 1e25) overflows to inf.
    assert jnp.isfinite(left_nested(coal))
    assert jnp.allclose(left_nested(coal), jnp.float32(1e25))
    assert jnp.isinf(((g_big * g_big) * g_small)(coal))


# --- Power operator (g ** c, c ** g, g ** h), jax_power, and k-additive refinement ---


def test_power_operators_dispatch_broadcast_modular_and_laziness(monkeypatch):
    g1 = _table_game(2, (2,)) + 1.0
    g2 = _table_game(2, ()) + 1.0
    coal = jnp.array([True, False])
    packed = pack_bool_coalition(coal)
    expected_g_c = jnp.power(g1(coal), 5)
    expected_c_g = jnp.power(2.0, g1(coal))
    expected_g_h = jnp.power(g1(coal), g2(coal))

    # Lazy construction, single eval per child (not m copies), and native packed forwarding.
    calls = Mock(wraps=DenseCoalitionGame.eval)

    def recording_eval(self, c):
        return calls(self, c)

    monkeypatch.setattr(DenseCoalitionGame, "eval", recording_eval)

    p_g_c = g1**5
    p_c_g = 2.0**g1
    p_g_h = g1**g2
    calls.assert_not_called()

    assert jnp.allclose(p_g_c(coal), expected_g_c)
    assert calls.call_count == 1
    calls.reset_mock()

    assert jnp.allclose(p_c_g(coal), expected_c_g)
    assert calls.call_count == 1
    calls.reset_mock()

    assert jnp.allclose(p_g_h(coal), expected_g_h)
    assert calls.call_count == 2

    def fail_unpack(*args, **kwargs):
        msg = "unpacked"
        raise AssertionError(msg)

    monkeypatch.setattr(jnp, "unpackbits", fail_unpack)
    assert jnp.allclose(p_g_c(packed), expected_g_c)
    assert jnp.allclose(p_c_g(packed), expected_c_g)
    assert jnp.allclose(p_g_h(packed), expected_g_h)

    monkeypatch.undo()

    # Scalar, JAX array, NumPy array, and batched broadcasting in both directions.
    for const in (2, 2.5, jnp.array([2.0, 3.0]), np.array([2.0, 3.0])):
        assert jnp.allclose((g1**const)(coal), jnp.power(g1(coal), const))
        assert jnp.allclose((const**g1)(coal), jnp.power(const, g1(coal)))

    batched_const = jnp.array([[2.0, 3.0], [4.0, 5.0]])
    for expr, expected in [
        (g2**batched_const, jnp.power(g2(coal), batched_const)),
        (batched_const**g2, jnp.power(batched_const, g2(coal))),
    ]:
        assert expr.shape == (2, 2)
        assert jnp.allclose(expr(coal), expected)
        assert jnp.allclose(expr[1, 0](coal), expected[1, 0])
        assert jnp.allclose(expr.reshape(4)(coal), expected.reshape(4))
        assert jnp.allclose(expr.T(coal), expected.T)
        stacked = jf.jax_stack([expr, expr])
        assert stacked.shape == (2, 2, 2)
        assert jnp.allclose(stacked(coal), jnp.stack([expected, expected]))

    # Multi-game power broadcasting and shaping.
    ga = _table_game(2, (2, 1)) + 1.0
    gb = _table_game(2, (1, 3)) + 1.0
    gp = ga**gb
    assert isinstance(gp, MappedGame)
    assert gp.shape == (2, 3)
    assert jnp.allclose(gp(coal), jnp.power(ga(coal), gb(coal)))
    assert jnp.allclose(gp[1, 2](coal), jnp.power(ga(coal), gb(coal))[1, 2])
    assert jnp.allclose(gp.reshape(6)(coal), jnp.power(ga(coal), gb(coal)).reshape(6))

    # Valid pow() forms (including explicit modulo=None) and invalid modular pow().
    assert jnp.allclose(pow(g1, 2)(coal), (g1**2)(coal))
    assert jnp.allclose(pow(g1, 2, None)(coal), (g1**2)(coal))
    assert jnp.allclose(pow(2.0, g1)(coal), (2.0**g1)(coal))
    assert jnp.allclose(pow(2.0, g1, None)(coal), (2.0**g1)(coal))
    assert jnp.allclose(pow(g1, g2, None)(coal), (g1**g2)(coal))
    for bad_mod in (
        lambda: pow(g1, 2, 3),
        lambda: pow(2, g1, 3),
        lambda: pow(g1, g2, 3),
    ):
        with pytest.raises(TypeError):
            bad_mod()

    # Unsupported non-numeric / boolean operands are rejected.
    for bad in ("abc", True, jnp.array([True, False]), object()):
        with pytest.raises(TypeError):
            _ = g1**bad
        with pytest.raises(TypeError):
            _ = bad**g1

    # Mismatched players are rejected; broadcast-equivalent players are accepted.
    catalog = LatentPlayers(["a", "b", "c"])
    p_left = FinitePlayers(jnp.array([0, 1]), players=catalog)
    p_bad = FinitePlayers(jnp.array([0, 2]), players=catalog)
    p_equiv = FinitePlayers(jnp.array([[0, 1], [0, 1]]), players=catalog)
    t_left = DenseCoalitionGame(jnp.arange(4.0) + 1.0, p_left)
    t_bad = DenseCoalitionGame(jnp.ones(4), p_bad)
    t_equiv = DenseCoalitionGame(jnp.full((2, 4), 2.0), p_equiv)
    with pytest.raises(ValueError, match="compatible player sets"):
        _ = t_left**t_bad
    pow_equiv = t_left**t_equiv
    assert pow_equiv.shape == (2,)
    assert jnp.allclose(pow_equiv(coal), jnp.power(t_left(coal), t_equiv(coal)))


def test_power_dtypes_weak_promotion_and_jax_edge_cases():
    p = FinitePlayers(2)
    f32 = DenseCoalitionGame(jnp.array([-2.0, 0.0, 2.0, 3.0], dtype=jnp.float32), p)
    f16 = DenseCoalitionGame(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float16), p)
    u8 = _IntDenseCoalitionGame(jnp.array([0, 1, 2, 3], dtype=jnp.uint8), p)
    i8 = _IntDenseCoalitionGame(jnp.array([-2, 0, 2, 3], dtype=jnp.int8), p)
    i32 = _IntDenseCoalitionGame(jnp.array([0, 1, 2, 3], dtype=jnp.int32), p)
    i32_neg = _IntDenseCoalitionGame(jnp.array([-1, -2, -1, -2], dtype=jnp.int32), p)
    all_sets = subset_patterns(2)
    scalar_coal = jnp.array([True, False])

    # No spurious float64 promotion under x64 for weakly typed Python scalars or float32 games.
    with jax.enable_x64(new_val=True):
        f32_pos = DenseCoalitionGame(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32), p)
        for expr in (f32_pos**2, f32_pos**2.0, 2**f32_pos, 2.0**f32_pos, f32_pos**f32_pos):
            assert expr.dtype == jnp.float32
            assert expr(all_sets).dtype == jnp.float32

    # Direct JAX power semantics across 0, 1, negative, fractional, complex, and integer dtypes.
    cases = [
        (f32**0, jnp.power(f32(all_sets), 0)),
        (f32**1, jnp.power(f32(all_sets), 1)),
        (f32**2, jnp.power(f32(all_sets), 2)),
        (f32**2.0, jnp.power(f32(all_sets), 2.0)),
        (f16**-1, jnp.power(f16(all_sets), -1)),
        (f16**-1.5, jnp.power(f16(all_sets), -1.5)),
        (f16 ** jnp.int32(2), jnp.power(f16(all_sets), jnp.int32(2))),
        (u8**0, jnp.power(u8(all_sets), 0)),
        (u8**2, jnp.power(u8(all_sets), 2)),
        (i8**2, jnp.power(i8(all_sets), 2)),
        (i8**0.5, jnp.power(i8(all_sets), 0.5)),
        (2.0**i8, jnp.power(2.0, i8(all_sets))),
        (f16**u8, jnp.power(f16(all_sets), u8(all_sets))),
        (f32 ** (1.0 + 1j), jnp.power(f32(all_sets), 1.0 + 1j)),
    ]
    for expr, expected in cases:
        actual = expr(all_sets)
        assert expr.dtype == expected.dtype
        assert actual.dtype == expected.dtype
        assert jnp.allclose(actual, expected, equal_nan=True)

    # Known concrete integer scalars use lax.integer_pow semantics (preserve base dtype).
    for exp in (2, jnp.int32(2), np.int64(2), np.array(2, dtype=np.int32)):
        q = i8**exp
        expected_s = jnp.power(i8(scalar_coal), exp)
        assert q.dtype == jnp.int8
        assert q(scalar_coal).dtype == jnp.int8
        assert q(all_sets).dtype == jnp.int8
        assert jnp.array_equal(q(scalar_coal), expected_s)
        jitted_q = jax.jit(lambda g, c: g(c))(q, scalar_coal)
        assert jitted_q.dtype == jnp.int8
        assert jnp.array_equal(jitted_q, expected_s)

    # Concrete negative integer exponents on integer games raise TypeError eagerly and under JIT.
    for neg_exp in (-1, jnp.int32(-1)):
        neg_int_pow = i8**neg_exp
        assert neg_int_pow.dtype == jnp.int8
        with pytest.raises(TypeError, match="negative powers"):
            _ = neg_int_pow(all_sets)
        with pytest.raises(TypeError, match="negative powers"):
            _ = jax.jit(lambda q, c: q(c))(neg_int_pow, scalar_coal)
        with pytest.raises(TypeError, match="negative powers"):
            _ = jax.jit(lambda g, c, e=neg_exp: (g**e)(c))(i8, scalar_coal)

    # Truly dynamic integer exponents follow JAX's dynamic power promotion rules.
    jit_dyn_pos = jax.jit(lambda g, e, c: (g**e)(c))(i8, jnp.int32(2), scalar_coal)
    assert jit_dyn_pos.dtype == jnp.int32
    assert jnp.array_equal(jit_dyn_pos, jax.jit(jnp.power)(i8(scalar_coal), jnp.int32(2)))

    jit_dyn_neg = jax.jit(lambda g, e, c: (g**e)(c))(i8, jnp.int32(-1), scalar_coal)
    assert jit_dyn_neg.dtype == jnp.int32
    assert jnp.array_equal(jit_dyn_neg, jax.jit(jnp.power)(i8(scalar_coal), jnp.int32(-1)))

    # Sliced or JIT-returned dynamic _ScalarPowerGame stays on the uniform dynamic dtype path.
    sliced_dyn = (i8 ** jnp.array([2, 3], dtype=jnp.int32))[0]
    returned_dyn = jax.jit(lambda g, e: g**e)(i8, jnp.int32(2))
    for dyn_expr in (sliced_dyn, returned_dyn):
        assert dyn_expr.dtype == jnp.int32
        assert dyn_expr(scalar_coal).dtype == jnp.int32
        assert dyn_expr(all_sets).dtype == jnp.int32
        assert jax.jit(lambda q, c: q(c))(dyn_expr, scalar_coal).dtype == jnp.int32

    # Mixed integer game/game and reflected constant**game powers use a uniform dynamic dtype path.
    for mixed_expr, expected_fn in (
        (i8**i32, lambda c: jax.jit(jnp.power)(i8(c), i32(c))),
        (i8**i32_neg, lambda c: jax.jit(jnp.power)(i8(c), i32_neg(c))),
        (jnp.int8(2) ** i32, lambda c: jax.jit(jnp.power)(jnp.int8(2), i32(c))),
    ):
        assert mixed_expr.dtype == jnp.int32
        for coal_in in (scalar_coal, all_sets):
            expected = expected_fn(coal_in)
            eager_val = mixed_expr(coal_in)
            jit_val = jax.jit(lambda q, c: q(c))(mixed_expr, coal_in)
            assert eager_val.dtype == jnp.int32
            assert jit_val.dtype == jnp.int32
            assert jnp.array_equal(eager_val, expected)
            assert jnp.array_equal(jit_val, expected)
        expected_scalar = expected_fn(scalar_coal)
        with jax.disable_jit():
            no_jit_val = mixed_expr(scalar_coal)
            assert no_jit_val.dtype == jnp.int32
            assert jnp.array_equal(no_jit_val, expected_scalar)

    # Negative base to fractional power yields NaN and 0 ** 0 == 1, matching JAX.
    assert jnp.isnan((f32**0.5)(jnp.array([False, False])))
    assert jnp.array_equal((f32**0)(all_sets), jnp.ones(4, dtype=jnp.float32))


def test_power_additivity_propagation_and_general_power_contract(monkeypatch):
    p = FinitePlayers(3)
    am = _additive_moebius(jnp.array([1.0, -2.0, 3.0]), p) + 0.5
    sm0 = SparseMoebiusGame(
        values=jnp.array([4.0]),
        interactions=jnp.array([[0, 0, 0]], dtype=bool),
        players=p,
        additivity=0,
    )
    sm2 = SparseMoebiusGame(
        values=jnp.array([1.0, 2.0, -3.0]),
        interactions=jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=bool),
        players=p,
        additivity=2,
    )
    table = _table_game(3, ())
    all_sets = subset_patterns(3)

    calls = Mock(wraps=FinitePlayerGame.eval)

    def recording_eval(self, coalition):
        return calls(self, coalition)

    monkeypatch.setattr(FinitePlayerGame, "eval", recording_eval)

    pow0 = am**0
    pow1 = am**1
    pow2 = am**2
    pow_np = am ** np.uint8(2)
    pow_np0d = am ** np.array(2, dtype=np.int32)
    pow_jax = am ** jnp.array(2, dtype=jnp.int32)
    pow_capped = sm2**2
    pow_zero_bound = sm0**5
    pow_zero_players = (
        SparseMoebiusGame(
            values=jnp.array([3.0]),
            interactions=jnp.zeros((1, 0), dtype=bool),
            players=FinitePlayers(0),
            additivity=0,
        )
        ** 2
    )
    calls.assert_not_called()

    for expr, expected_bound in [
        (pow0, 0),
        (pow1, 1),
        (pow2, 2),
        (pow_np, 2),
        (pow_np0d, 2),
        (pow_jax, 2),
        (pow_capped, 3),
        (pow_zero_bound, 0),
        (pow_zero_players, 0),
    ]:
        assert isinstance(expr, ArithmeticGame)
        assert expr._known_additivity() == expected_bound
        assert expr.additivity() == expected_bound

    for expr in (pow0, pow1, pow2, pow_capped):
        exact = _exact_moebius_table(expr(all_sets), 3)
        assert jnp.allclose(expr.coefficients(all_sets), exact, atol=1e-5)
        assert jnp.allclose(expr.coefficients(order=0), exact[:1], atol=1e-5)
        assert jnp.allclose(expr.coefficients(order=1), exact[jnp.array([1, 2, 4])], atol=1e-5)
        assert jnp.allclose(expr.coefficients(order=2), exact[jnp.array([3, 5, 6])], atol=1e-5)
        assert jnp.allclose(expr.coefficients(order=3), exact[jnp.array([7])], atol=1e-5)

    assert jnp.allclose(pow_zero_players.coefficients(order=0), jnp.array([9.0]))

    calls.reset_mock()
    assert jnp.array_equal(pow0.coefficients(order=1), jnp.zeros(3))
    assert jnp.array_equal(pow2.coefficients(order=3), jnp.zeros(1))
    assert jnp.array_equal(pow2.coefficients(jnp.array([True, True, True])), jnp.array(0.0))
    calls.assert_not_called()

    composed = 2.0 * pow2 + 1.0
    assert composed.additivity() == 2
    assert jnp.allclose(
        composed.coefficients(all_sets), _exact_moebius_table(composed(all_sets), 3), atol=1e-5
    )

    general_cases = [
        table**0,
        table**2,
        am**-1,
        am**-2,
        am**2.0,
        am**0.5,
        am ** (2.0 + 0j),
        2**am,
        1**am,
        am**am,
        am ** jnp.array([2, 2]),
        am ** np.array([2]),
    ]
    for general in general_cases:
        assert isinstance(general, ArithmeticGame)
        assert general._known_additivity() is None

    base_add = _additive_moebius(jnp.array([2.0, 3.0, 4.0]), p)
    sq = base_add**2
    base_leaves = jax.tree.leaves(base_add)
    sq_leaves, sq_treedef = jax.tree.flatten(sq)
    assert len(sq_leaves) == len(base_leaves)
    replaced = [
        jnp.asarray(3) if x.shape == () and jnp.issubdtype(x.dtype, jnp.integer) else x
        for x in sq_leaves
    ]
    unflattened = jax.tree.unflatten(sq_treedef, replaced)
    full = p.full_coalition()
    assert unflattened.additivity() == 2
    assert jnp.allclose(unflattened(full), sq(full))
    assert jnp.allclose(unflattened.coefficients(full), 0.0)
    assert jnp.allclose((base_add**3).coefficients(full), 144.0)

    with jax.enable_x64(new_val=False):
        big_pos = am ** np.int64(2**31)
        assert big_pos.additivity() == p.n_players
        assert jnp.array_equal(big_pos(full), jnp.power(am(full), np.int64(2**31)))

        pre_truncated_neg = am ** jnp.asarray(np.int64(2**31))
        assert pre_truncated_neg._known_additivity() is None


def test_power_jit_dynamic_operands_and_structured_values():
    players = FinitePlayers(jnp.array([[0, 1], [0, 2]]), latent_n_players=3)
    t1 = DenseCoalitionGame(jnp.arange(8.0).reshape(2, 4) + 1.0, players)
    t2 = DenseCoalitionGame(jnp.full((2, 4), 2.0), players)
    am = _additive_moebius(jnp.array([[1.0, 2.0], [3.0, 4.0]]), players)
    coal = jnp.array([True, False])
    c_arr = jnp.array([2.0, 3.0])

    pow_k = am**2
    pow_neg = t1**-1
    pow_const = t1**c_arr
    pow_refl = c_arr**t1
    pow_games = t1**t2
    for game in (pow_k, pow_neg, pow_const, pow_refl, pow_games):
        leaves = jax.tree.leaves(game)
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)
        assert jnp.allclose(jax.jit(lambda g, c: g(c))(game, coal), game(coal))
        assert jnp.allclose(jax.jit(lambda g: g[1])(game)(coal), game[1](coal))
        assert jnp.allclose(
            jax.jit(lambda g: g.reshape(1, 2))(game)(coal), game(coal).reshape(1, 2)
        )
    assert len(jax.tree.leaves(pow_k)) == len(jax.tree.leaves(am))
    assert len(jax.tree.leaves(pow_neg)) == len(jax.tree.leaves(t1))
    assert pow_k[1].additivity() == 2 and pow_k[1].shape == ()
    assert pow_k.reshape(1, 2).additivity() == 2 and pow_k.reshape(1, 2).shape == (1, 2)
    stacked_int = jf.jax_stack([t1**2, t1**2])
    assert stacked_int.shape == (2, 2)
    assert jnp.allclose(stacked_int(coal), jnp.stack([(t1**2)(coal), (t1**2)(coal)]))

    mixed = jnp.array([[0, 0], [1, 0], [1, 1]], dtype=bool)
    assert jnp.allclose(
        jax.jit(lambda g, m: g.coefficients(m))(pow_k, mixed[:, None]),
        pow_k.coefficients(mixed[:, None]),
    )

    traced_known: list[int | None] = []

    @jax.jit
    def jit_dynamic_exp(g, exp, s):
        expr = g**exp
        traced_known.append(expr._known_additivity())
        return expr(s)

    assert jnp.allclose(jit_dynamic_exp(am, jnp.array(2), coal), jnp.power(am(coal), 2))
    assert jnp.allclose(jit_dynamic_exp(am, jnp.array(3), coal), jnp.power(am(coal), 3))
    assert traced_known == [None]

    jit_dynamic_base = jax.jit(lambda c, g, s: (c**g)(s))
    assert jnp.allclose(jit_dynamic_base(c_arr, t1, coal), jnp.power(c_arr, t1(coal)))
    assert jnp.allclose(
        jit_dynamic_base(jnp.array([4.0, 5.0]), t1, coal),
        jnp.power(jnp.array([4.0, 5.0]), t1(coal)),
    )

    captured_exp = jnp.array(2, dtype=jnp.int32)
    assert jnp.allclose(
        jax.jit(lambda g: (g**2).coefficients(order=2))(am), pow_k.coefficients(order=2)
    )
    assert jnp.allclose(
        jax.jit(lambda g: (g**captured_exp).coefficients(order=2))(am),
        pow_k.coefficients(order=2),
    )

    p1 = FinitePlayers(jnp.array([0, 2]), latent_n_players=4)
    p2 = FinitePlayers(jnp.array([0, 3]), latent_n_players=4)
    with pytest.raises(TypeError, match="traced player IDs"):
        jax.jit(lambda a, b: (a**b)(coal))(
            DenseCoalitionGame(jnp.ones(4), p1), DenseCoalitionGame(jnp.ones(4), p2)
        )

    p2_plain = FinitePlayers(2)
    vec_game = MappedGame(
        JaxBroadcastTuple((_table_game(2, (2,)) + 1.0,)),
        lambda vs: _VectorValue(jnp.stack([vs[0], vs[0] + 2.0], axis=-1)),
    )
    exp_batch = jnp.array([2.0, 3.0])
    res_pow = (vec_game**exp_batch)(coal)
    assert isinstance(res_pow, _VectorValue)
    assert jnp.allclose(res_pow.values, jnp.power(vec_game(coal).values, exp_batch[:, None]))
    res_rpow = (exp_batch**vec_game)(coal)
    assert isinstance(res_rpow, _VectorValue)
    assert jnp.allclose(res_rpow.values, jnp.power(exp_batch[:, None], vec_game(coal).values))
    res_gpow = (vec_game**vec_game)(coal)
    assert isinstance(res_gpow, _VectorValue)
    assert jnp.allclose(res_gpow.values, jnp.power(vec_game(coal).values, vec_game(coal).values))

    v_add = _VectorAdditiveGame(
        empty=_VectorValue(jnp.array([[1.0, 2.0], [3.0, 4.0]])),
        contribs=jnp.array([[[2.0, 3.0], [4.0, 5.0]], [[1.0, 2.0], [3.0, 1.0]]]),
        _players=p2_plain,
    )
    v_sq = v_add**2
    assert v_sq.additivity() == 2
    v_above = v_sq.coefficients(order=3)
    assert isinstance(v_above, _VectorValue)
    assert v_above.values.shape == (2, 0, 2)
    v_c2 = v_sq.coefficients(jnp.array([True, True]))
    assert isinstance(v_c2, _VectorValue)
    assert jnp.allclose(v_c2.values, 2.0 * v_add.contribs[:, 0, :] * v_add.contribs[:, 1, :])


def test_weak_dtype_metadata_propagates_across_table_arithmetic_and_mapped_game(monkeypatch):
    p = FinitePlayers(1)
    g = DenseCoalitionGame(jnp.full((2,), 2.0), p)  # weak float32
    h = DenseCoalitionGame(jnp.full((2,), 3.0, dtype=jnp.float16), p)  # strong float16
    s = p.full_coalition()
    assert g.values.weak_type
    assert not h.values.weak_type

    calls = Mock(wraps=DenseCoalitionGame.eval)

    def recording_eval(self, coalition):
        return calls(self, coalition)

    monkeypatch.setattr(DenseCoalitionGame, "eval", recording_eval)

    add_gh = g + h
    sub_gh = g - h
    mul_gh = g * h
    div_gh = g / h
    pow_gh = g**h
    mapped_gh = MappedGame(JaxBroadcastTuple((g, h)), lambda vs: vs[0] * vs[1])
    nested_add = ((g + 1) + 2) + h
    nested_neg = (-g) + h
    nested_mul = (g * g) + h
    nested_div = (g / g) + h
    nested_sdiv = (g / 2.0) + h
    nested_ipow = (g**2) + h
    nested_spow = (g**2.0) + h
    nested_rpow = (2.0**g) + h
    nested_gpow = (g**g) + h
    nested_mapped = MappedGame(JaxBroadcastTuple((g,)), lambda vs: vs[0] + 1.0) + h
    stacked_mapped = jf.jax_stack([MappedGame(JaxBroadcastTuple((g,)), lambda vs: vs[0])]) + h
    calls.assert_not_called()

    raw_g, raw_h = g(s), h(s)
    cases = [
        (add_gh, raw_g + raw_h),
        (sub_gh, raw_g - raw_h),
        (mul_gh, raw_g * raw_h),
        (div_gh, raw_g / raw_h),
        (pow_gh, raw_g**raw_h),
        (mapped_gh, raw_g * raw_h),
        (nested_add, ((raw_g + 1) + 2) + raw_h),
        (nested_neg, (-raw_g) + raw_h),
        (nested_mul, (raw_g * raw_g) + raw_h),
        (nested_div, (raw_g / raw_g) + raw_h),
        (nested_sdiv, (raw_g / 2.0) + raw_h),
        (nested_ipow, (raw_g**2) + raw_h),
        (nested_spow, (raw_g**2.0) + raw_h),
        (nested_rpow, (2.0**raw_g) + raw_h),
        (nested_gpow, (raw_g**raw_g) + raw_h),
        (nested_mapped, (raw_g + 1.0) + raw_h),
        (stacked_mapped, jnp.stack([raw_g]) + raw_h),
    ]
    for expr, expected in cases:
        assert expr.dtype == jnp.float16
        assert expected.dtype == jnp.float16
        actual = expr(s)
        assert actual.dtype == jnp.float16
        assert jnp.allclose(actual, expected)
        jit_actual = jax.jit(lambda game, coal: game(coal))(expr, s)
        assert jit_actual.dtype == jnp.float16
        assert jnp.allclose(jit_actual, expected)

    am_weak = _additive_moebius(jnp.full((1,), 2.0), p)
    assert am_weak.values.weak_type
    assert not am_weak(s).weak_type
    am_div_h = am_weak / h
    assert am_div_h.dtype == am_div_h(s).dtype == am_weak.values.dtype

    lying = MappedGame(
        JaxBroadcastTuple((g, h)), lambda vs: vs[0] * vs[1], output_dtype=jnp.float32
    )
    with pytest.raises(TypeError, match="expected float32"):
        lying(s)


def test_division_additivity_propagation_checks_narrow_computation_dtype():
    from shapiq.game._arithmetic_ops import _ScalarDivGame  # noqa: PLC0415

    p = FinitePlayers(2)
    am16 = _additive_moebius(jnp.array([1.0, 2.0], dtype=jnp.float16), p)
    empty = p.empty_coalition()
    full = p.full_coalition()

    for bad_const in (1e-8, 1e5, 1_000_000, jnp.asarray(1e-8), jnp.asarray(1e5)):
        expr = am16 / bad_const
        assert expr._known_additivity() is None
    assert jnp.isnan((am16 / 1e-8)(empty))
    assert jnp.isinf((am16 / 1e-8)(full))

    raw_div = _ScalarDivGame(JaxBroadcastTuple((am16,)), jnp.asarray(1e-8))
    assert raw_div._known_additivity() is None

    strong_ok = am16 / jnp.array(1e-8, dtype=jnp.float32)
    assert strong_ok.additivity() == 1
    assert strong_ok.dtype == jnp.float32
    assert jnp.allclose(strong_ok.contributions(), am16.contributions() / jnp.float32(1e-8))

    closed_underflow = jnp.asarray(1e-8)
    closed_valid = jnp.asarray(2.0)
    jit_known: list[int | None] = []

    @jax.jit
    def jit_check_underflow(s: jax.Array) -> jax.Array:
        expr = am16 / closed_underflow
        jit_known.append(expr._known_additivity())
        return expr(s)

    assert jnp.isnan(jit_check_underflow(empty))
    assert jit_known == [None]

    @jax.jit
    def jit_valid_contribs() -> jax.Array:
        return (am16 / closed_valid).contributions()

    assert jnp.allclose(jit_valid_contribs(), am16.contributions() / jnp.float16(2.0))


def test_jit_allows_closed_over_concrete_equal_player_ids_and_rejects_mismatched():
    g = DenseCoalitionGame(jnp.array([1.0, 2.0]), FinitePlayers(1))
    h = DenseCoalitionGame(jnp.array([3.0, 4.0]), FinitePlayers(1))
    coal = jnp.array([True])

    @jax.jit
    def jit_add(s: jax.Array) -> jax.Array:
        return (g + h)(s)

    @jax.jit
    def jit_mul(s: jax.Array) -> jax.Array:
        return (g * h)(s)

    assert jnp.allclose(jit_add(coal), jnp.array(6.0))
    assert jnp.allclose(jit_mul(coal), jnp.array(8.0))

    g_bad = DenseCoalitionGame(
        jnp.array([1.0, 2.0]), FinitePlayers(jnp.array([0]), latent_n_players=2)
    )
    h_bad = DenseCoalitionGame(
        jnp.array([3.0, 4.0]), FinitePlayers(jnp.array([1]), latent_n_players=2)
    )

    @jax.jit
    def jit_bad_add(s: jax.Array) -> jax.Array:
        return (g_bad + h_bad)(s)

    with pytest.raises(ValueError, match="compatible player sets"):
        jit_bad_add(coal)

    @jax.jit
    def jit_mixed(b: DenseCoalitionGame, s: jax.Array) -> jax.Array:
        return (g + b)(s)

    with pytest.raises(TypeError, match="traced player IDs"):
        jit_mixed(h, coal)
