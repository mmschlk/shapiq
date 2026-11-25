"""Tests for lazy ReductionGame, SumReductionGame, jax_sum, and FinitePlayerGame.sum."""

from dataclasses import dataclass, replace
from inspect import isabstract
from typing import Any, ClassVar, override
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flextype import (
    cast as flex_cast,
    castmethod,
    try_cast,
)
from jaxtyping import TypeCheckError
from ndlike import jax_functions as jf
from ndlike.protected_axis.jax import JaxAxisProtected
from ndlike.tuple.jax import JaxBroadcastTuple

from shapiq.coalition.coalition import pack_bool_coalition
from shapiq.game import (
    AdditiveGame,
    ArithmeticGame,
    FinitePlayerGame,
    KAdditiveGame,
    MappedGame,
    ReductionGame,
    SumReductionGame,
    TableGame,
)
from shapiq.game.discretederivative import subset_patterns
from shapiq.game.exact import ExactComputer
from shapiq.game.moebius import AdditiveMoebiusGame, MoebiusGame
from shapiq.players import FinitePlayers, LatentPlayers


def _table_game(
    n_players: int = 2,
    batch_shape: tuple[int, ...] = (2, 3),
    *,
    players: FinitePlayers[Any] | None = None,
    dtype: Any = jnp.float32,
) -> TableGame[Any]:
    size = 1 << n_players
    count = int(np.prod(batch_shape)) * size
    values = jnp.arange(count, dtype=dtype).reshape(*batch_shape, size)
    return TableGame(values, FinitePlayers(n_players) if players is None else players)


def _additive_moebius(
    values: jax.Array,
    players: FinitePlayers[Any],
    perm: list[int] | None = None,
) -> AdditiveMoebiusGame[Any, Any]:
    n = players.n_players
    order = list(range(n)) if perm is None else perm
    interactions = jnp.eye(n, dtype=bool)[jnp.array(order)]
    return AdditiveMoebiusGame(values=values, interactions=interactions, players=players)


def _exact_moebius_table(values: jax.Array, n_players: int) -> jax.Array:
    sets = subset_patterns(n_players)
    subset = ~jnp.any(sets[None, :, :] & ~sets[:, None, :], axis=-1)
    overlap = jnp.sum(sets[:, None, :] & sets[None, :, :], axis=-1)
    sign = (-1.0) ** (jnp.sum(sets, axis=-1)[:, None] - overlap)
    return values @ (subset * sign).T


@dataclass(frozen=True)
class _VectorValue(JaxAxisProtected["_VectorValue"]):
    values: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"values": 1}
    permitted_functions: ClassVar[set[Any]] = {
        jf.jax_add,
        jf.jax_multiply,
        jf.jax_power,
        jf.jax_sum,
        jf.jax_where,
    }


@dataclass(frozen=True)
class _StructuredPair(JaxAxisProtected["_StructuredPair"]):
    count: jax.Array
    vector: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"count": 0, "vector": 1}
    permitted_functions: ClassVar[set[Any]] = {
        jf.jax_add,
        jf.jax_multiply,
        jf.jax_sum,
        jf.jax_true_divide,
        jf.jax_where,
    }


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _IntTableGame(FinitePlayerGame[Any, jax.Array], JaxAxisProtected["_IntTableGame"]):
    values: jax.Array
    _players: FinitePlayers[Any]
    protected_axes: ClassVar[dict[str, int]] = {"values": 1, "_players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"_players"}

    @property
    @override
    def players(self) -> FinitePlayers[Any]:
        return self._players

    @override
    def _eval_bool(self, coalition: jax.Array) -> jax.Array:
        weights = 1 << jnp.arange(self.n_players, dtype=jnp.int32)
        idx = jnp.sum(coalition.astype(jnp.int32) * weights, axis=-1)
        shape = jnp.broadcast_shapes(self.shape, idx.shape)
        table = jnp.broadcast_to(self.values, (*shape, self.values.shape[-1]))
        return jnp.take_along_axis(table, jnp.broadcast_to(idx, shape)[..., None], axis=-1)[..., 0]

    @override
    def _eval_packed(self, coalition: jax.Array) -> jax.Array:
        return self._eval_bool(
            jnp.unpackbits(coalition, axis=-1, bitorder="little")[..., : self.n_players].astype(
                bool
            )
        )


@dataclass(frozen=True)
class _VectorAdditiveGame(AdditiveGame[Any, _VectorValue], JaxAxisProtected["_VectorAdditiveGame"]):
    empty: _VectorValue
    contribs: jax.Array
    _players: FinitePlayers[Any]
    protected_axes: ClassVar[dict[str, int]] = {"empty": 0, "contribs": 2, "_players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"_players"}

    @property
    @override
    def players(self) -> FinitePlayers[Any]:
        return self._players

    @override
    def contributions(self) -> _VectorValue:
        if self.n_players == 0 or self.additivity == 0:
            return _VectorValue(
                jnp.zeros(
                    (*self.shape, self.n_players, self.empty.values.shape[-1]),
                    dtype=self.empty.values.dtype,
                )
            )
        return _VectorValue(self.contribs)

    @override
    def _eval_bool(self, coalition: jax.Array) -> _VectorValue:
        batch = jnp.broadcast_shapes(self.shape, coalition.shape[:-1])
        empty_v = jnp.broadcast_to(self.empty.values, (*batch, self.empty.values.shape[-1]))
        contribs_v = jnp.broadcast_to(
            self.contribs, (*batch, self.n_players, self.empty.values.shape[-1])
        )
        mask = jnp.broadcast_to(coalition, (*batch, self.n_players))[..., None]
        return _VectorValue(empty_v + jnp.sum(jnp.where(mask, contribs_v, 0.0), axis=-2))

    @override
    def _eval_packed(self, coalition: jax.Array) -> _VectorValue:
        return self._eval_bool(
            jnp.unpackbits(coalition, axis=-1, bitorder="little")[..., : self.n_players].astype(
                bool
            )
        )


@dataclass(frozen=True)
class _PairAdditiveGame(AdditiveGame[Any, _StructuredPair], JaxAxisProtected["_PairAdditiveGame"]):
    empty: _StructuredPair
    contribs: _StructuredPair
    _players: FinitePlayers[Any]
    protected_axes: ClassVar[dict[str, int]] = {"empty": 0, "contribs": 1, "_players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"_players"}

    @property
    @override
    def players(self) -> FinitePlayers[Any]:
        return self._players

    @override
    def contributions(self) -> _StructuredPair:
        return self.contribs

    @override
    def _eval_bool(self, coalition: jax.Array) -> _StructuredPair:
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
            + jnp.sum(
                jnp.where(mask[..., None], v, jnp.zeros((), dtype=self.empty.vector.dtype)),
                axis=-2,
                dtype=self.empty.vector.dtype,
            ),
        )

    @override
    def _eval_packed(self, coalition: jax.Array) -> _StructuredPair:
        return self._eval_bool(
            jnp.unpackbits(coalition, axis=-1, bitorder="little")[..., : self.n_players].astype(
                bool
            )
        )


# --- Abstract ReductionGame contract and basic validation ---


def test_reduction_game_is_abstract_and_supports_custom_subclasses():
    assert isabstract(ReductionGame)
    assert ReductionGame.__abstractmethods__ == {"_reduce"}

    @dataclass(frozen=True, slots=True, weakref_slot=True)
    class _MaxReductionGame[P](ReductionGame[P, jax.Array]):
        scale: float

        @override
        def _reduce(self, values: jax.Array) -> jax.Array:
            return jnp.max(values, axis=-1) * self.scale

    g = _table_game(2, (2, 3, 4))
    r = _MaxReductionGame(g, 2.0, axis=(0, 2))
    coal = jnp.array([True, False])
    assert r.shape == (3,)
    assert r.n_reduced == 8
    assert r.game.shape == (3, 8)
    assert jnp.allclose(r(coal), jnp.max(g(coal), axis=(0, 2)) * 2.0)


def test_reduction_game_validates_inputs_axes_and_dtypes():
    g = _table_game(2, (2, 3))
    with pytest.raises((TypeError, TypeCheckError)):
        SumReductionGame(jnp.ones((2, 3)))  # type: ignore[arg-type]

    for bad_axis in (True, "0", 1.5, (True,), (0, 1.5)):
        with pytest.raises((TypeError, TypeCheckError)):
            SumReductionGame(g, axis=bad_axis)  # type: ignore[arg-type]

    for oob_axis in (2, -3, (0, 2)):
        with pytest.raises(ValueError, match="out of bounds"):
            SumReductionGame(g, axis=oob_axis)

    for dup_axis in ((0, 0), (1, -1)):
        with pytest.raises(ValueError, match="duplicate"):
            SumReductionGame(g, axis=dup_axis)

    with pytest.raises((TypeError, TypeCheckError)):
        SumReductionGame(g, keepdims=1)  # type: ignore[arg-type]

    for bad_dtype in (bool, jnp.bool_, "not_a_dtype"):
        with pytest.raises(TypeError, match="numeric dtype"):
            SumReductionGame(g, output_dtype=bad_dtype)

    # Scalar 0D game rejects default axis=-1, but accepts axis=None and axis=().
    g0 = _table_game(2, ())
    with pytest.raises(ValueError, match="out of bounds"):
        SumReductionGame(g0)
    assert SumReductionGame(g0, axis=None).shape == ()
    assert SumReductionGame(g0, axis=()).shape == ()


# --- Core jax_sum and FinitePlayerGame.sum semantics ---


@pytest.mark.parametrize(
    ("batch_shape", "axis", "keepdims"),
    [
        ((), None, False),
        ((), None, True),
        ((), (), False),
        ((4,), None, False),
        ((4,), 0, False),
        ((4,), -1, True),
        ((2, 3, 4), None, False),
        ((2, 3, 4), None, True),
        ((2, 3, 4), 1, False),
        ((2, 3, 4), -2, True),
        ((2, 3, 4), (0, 2), False),
        ((2, 3, 4), (2, 0), False),
        ((2, 3, 4), (0, 2), True),
        ((2, 3, 4), [0, -1], False),
        ((2, 3, 4), (), False),
        ((2, 3, 4), (), True),
        ((2, 0, 4), 1, False),
        ((2, 0, 4), 1, True),
        ((2, 0, 4), (0, 2), False),
    ],
)
def test_jax_sum_and_sum_method_match_jnp_sum_across_axes_and_shapes(batch_shape, axis, keepdims):
    g = _table_game(2, batch_shape)
    coal_scalar = jnp.array([True, False])
    coal_batch = jnp.array([[False, False], [True, False], [True, True]])
    expected_scalar = jnp.sum(g(coal_scalar), axis=axis, keepdims=keepdims)

    s_fn = jf.jax_sum(g, axis=axis, keepdims=keepdims)
    s_meth = g.sum(axis=axis, keepdims=keepdims)
    s_direct = SumReductionGame(g, axis=axis, keepdims=keepdims)

    for reduced in (s_fn, s_meth, s_direct):
        assert isinstance(reduced, SumReductionGame)
        assert reduced.shape == expected_scalar.shape
        assert reduced.dtype == expected_scalar.dtype
        actual = reduced(coal_scalar)
        assert actual.shape == expected_scalar.shape
        assert jnp.allclose(actual, expected_scalar)

        # Extra leading coalition batch dimensions broadcast cleanly without colliding with reduced axes.
        coal_lead = coal_batch.reshape((3, *(1,) * reduced.ndim, 2))
        if axis is None:
            lead_axis = tuple(range(1, 1 + g.ndim))
        else:
            raw_axes = (axis,) if isinstance(axis, int) else axis
            lead_axis = tuple(1 + (a + g.ndim if a < 0 else a) for a in raw_axes)
        expected_lead = jnp.sum(
            g(coal_batch.reshape((3, *(1,) * g.ndim, 2))),
            axis=lead_axis,
            keepdims=keepdims,
        )
        actual_lead = reduced(coal_lead)
        assert actual_lead.shape == expected_lead.shape
        assert jnp.allclose(actual_lead, expected_lead)


def test_reduction_game_default_axis_round_trips_and_forbids_mutations(monkeypatch):
    g = _table_game(2, (2, 3, 4))
    coal = jnp.array([True, True])
    packed = pack_bool_coalition(coal)
    expected = jnp.sum(g(coal), axis=(0, 2))

    calls = Mock(wraps=TableGame.eval)

    def recording_eval(self, coalition):
        return calls(self, coalition)

    monkeypatch.setattr(TableGame, "eval", recording_eval)

    r = g.sum(axis=(0, 2))
    r_roundtrip = SumReductionGame(r.game)
    sliced = r[1]
    reshaped = r.reshape(1, 3)
    _ = r.dtype
    calls.assert_not_called()

    assert r_roundtrip.shape == r.shape == (3,)
    assert jnp.allclose(r(coal), expected)
    assert calls.call_count == 1
    assert jnp.allclose(r_roundtrip(coal), expected)
    assert jnp.allclose(sliced(packed), expected[1])
    assert jnp.allclose(reshaped(packed), expected.reshape(1, 3))

    # Packed coalitions pass through without unpacking.
    def fail_unpack(*args, **kwargs):
        msg = "unpacked"
        raise AssertionError(msg)

    monkeypatch.setattr(jnp, "unpackbits", fail_unpack)
    assert jnp.allclose(r(packed), expected)

    # Indexed updates and representation-level fieldwise reductions remain disabled.
    with pytest.raises(TypeError, match="not permitted"):
        _ = r.at[0].set(r[0])
    assert jnp.allclose(r.at[1].get()(coal), expected[1])
    assert not r.permitted_functions
    with pytest.raises(TypeError):
        jf.jax_mean(r)

    # Invalid coalition batch shapes are rejected against the reduced batch shape.
    with pytest.raises((TypeCheckError, ValueError)):
        r(jnp.ones((2, 2), dtype=bool))
    assert r.is_valid(jnp.ones((3, 2), dtype=bool))
    assert not r.is_valid(jnp.ones((2, 2), dtype=bool))


def test_jax_sum_rejects_unsupported_arguments_and_non_game_primary_operand():
    g = _table_game(2, (2,))
    with pytest.raises(NotImplementedError, match="out"):
        jf.jax_sum(g, out=jnp.zeros(()))  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="initial"):
        jf.jax_sum(g, initial=1.0)
    with pytest.raises(NotImplementedError, match="where"):
        jf.jax_sum(g, where=jnp.array([True, False]))

    # When `a` is not a FinitePlayerGame, dispatch returns NotImplemented -> TypeError.
    with pytest.raises(TypeError):
        jf.jax_sum(jnp.ones(2), initial=g)
    with pytest.raises(TypeError):
        jf.jax_sum(jnp.ones(2), where=g)


# --- Player compatibility validation (broadcast, concrete, empty, JIT) ---


def test_player_compatibility_broadcast_concrete_and_incompatible_axes():
    catalog = LatentPlayers(["a", "b", "c", "d"])
    coal = jnp.array([True, False])

    # 1. Scalar players stay compact across all reductions.
    p_scalar = FinitePlayers(jnp.array([0, 2]), players=catalog)
    g_scalar = _table_game(2, (2, 3, 4), players=p_scalar)
    r_scalar = g_scalar.sum(axis=(0, 2))
    assert r_scalar.players is p_scalar
    assert r_scalar.players.shape == ()

    # 2. Broadcast players (size-1 on reduced axes, varying on surviving axes) stay compact.
    p_bcast = FinitePlayers(
        jnp.array([[[0, 1]], [[0, 2]], [[1, 3]]]), players=catalog
    )  # batch (3, 1)
    g_bcast = _table_game(2, (2, 3, 4), players=p_bcast)
    r_bcast = g_bcast.sum(axis=(0, 2))
    assert r_bcast.shape == (3,)
    assert r_bcast.players.shape == (3,)
    assert jnp.array_equal(r_bcast.players.ids, jnp.array([[0, 1], [0, 2], [1, 3]]))
    assert r_bcast.players.players is catalog
    assert jnp.allclose(r_bcast(coal), jnp.sum(g_bcast(coal), axis=(0, 2)))

    r_bcast_kd = g_bcast.sum(axis=(0, 2), keepdims=True)
    assert r_bcast_kd.shape == (1, 3, 1)
    assert r_bcast_kd.players.shape == (3, 1)
    assert jnp.array_equal(r_bcast_kd.players.ids, p_bcast.ids)

    # 3. Concrete equal players along reduced axis succeed even when different across surviving axis.
    ids_equal_rows = jnp.array(
        [
            [[0, 1], [0, 1], [0, 1]],
            [[1, 2], [1, 2], [1, 2]],
        ]
    )
    p_rows = FinitePlayers(ids_equal_rows, players=catalog)
    g_rows = _table_game(2, (2, 3), players=p_rows)
    r_rows = g_rows.sum(axis=1)
    assert r_rows.shape == (2,)
    assert r_rows.players.shape == (2,)
    assert jnp.array_equal(r_rows.players.ids, jnp.array([[0, 1], [1, 2]]))
    assert jnp.allclose(r_rows(coal), jnp.sum(g_rows(coal), axis=1))

    # Reducing axis 0 on g_rows mixes [0, 1] and [1, 2] -> rejected!
    with pytest.raises(ValueError, match="compatible player sets"):
        g_rows.sum(axis=0)
    with pytest.raises(ValueError, match="compatible player sets"):
        g_rows.sum(axis=None)

    # 4. Unequal players within one reduction group are rejected.
    ids_bad = jnp.array(
        [
            [[0, 1], [0, 1], [0, 1]],
            [[1, 2], [1, 3], [1, 2]],
        ]
    )
    g_bad = _table_game(2, (2, 3), players=FinitePlayers(ids_bad, players=catalog))
    with pytest.raises(ValueError, match="compatible player sets"):
        g_bad.sum(axis=1)


def test_player_compatibility_empty_axes_rebuild_and_zero_player_games():
    catalog = LatentPlayers(["a", "b", "c"])
    coal = jnp.array([True, False])

    # Zero-length reduced game axis with broadcast players (2, 1) succeeds and rebuilds cleanly!
    p_row = FinitePlayers(jnp.array([[[0, 1]], [[1, 2]]]), players=catalog)  # batch (2, 1)
    g_empty_red = _table_game(2, (2, 0), players=p_row)
    r_empty_red = g_empty_red.sum(axis=1)
    assert r_empty_red.shape == (2,)
    assert r_empty_red.players.shape == (2,)
    assert jnp.array_equal(r_empty_red.players.ids, jnp.array([[0, 1], [1, 2]]))
    assert jnp.array_equal(r_empty_red(coal), jnp.zeros(2, dtype=jnp.float32))

    # Rebuilding an unchanged empty reduction retains the surviving players.
    p_row3 = FinitePlayers(jnp.array([[[[0, 1]]], [[[1, 2]]]]), players=catalog)  # batch (2, 1, 1)
    r_empty3 = _table_game(2, (2, 0, 3), players=p_row3).sum(axis=(1, 2))
    for rebuilt in (replace(r_empty3), r_empty3.with_protected_values({"game": r_empty3.game})):
        assert rebuilt.shape == (2,)
        assert jnp.array_equal(rebuilt.players.ids, jnp.array([[0, 1], [1, 2]]))
        assert rebuilt.players.players is catalog

    # Zero-length reduced player axis (2, 0) with positive n_players and nonempty output (2,) cannot infer IDs.
    p_empty_red = FinitePlayers(jnp.zeros((2, 0, 2), dtype=jnp.int32), players=catalog)
    g_ambiguous = _table_game(2, (2, 0), players=p_empty_red)
    with pytest.raises(ValueError, match="zero-length player-set axis"):
        g_ambiguous.sum(axis=1)

    # Empty surviving batch with differing players along reduced axis succeeds (eager & JIT).
    p_diff = FinitePlayers(jnp.array([[0, 1], [0, 2]]), players=catalog)  # batch (2,)
    g_zero_lead = _table_game(2, (0, 2), players=p_diff)
    r_zero_lead = g_zero_lead.sum(axis=1)
    assert r_zero_lead.shape == (0,)
    assert r_zero_lead.players.shape == (0,)
    assert r_zero_lead.players.players is catalog
    assert r_zero_lead.players.latent_n_players == 3
    assert jax.jit(lambda g: g.sum(axis=1).full_value())(g_zero_lead).shape == (0,)

    # Zero-length reduced player axis with empty surviving batch (0, 0) or n_players=0 succeeds.
    p_empty_both = FinitePlayers(jnp.zeros((0, 0, 2), dtype=jnp.int32), players=catalog)
    r_empty_both = _table_game(2, (0, 0), players=p_empty_both).sum(axis=1)
    assert r_empty_both.shape == (0,)
    assert r_empty_both.players.shape == (0,)
    assert r_empty_both.players.players is catalog

    p_zero_players = FinitePlayers(jnp.zeros((2, 0, 0), dtype=jnp.int32), latent_n_players=3)
    r_zero_players = _table_game(0, (2, 0), players=p_zero_players).sum(axis=1)
    assert r_zero_players.shape == (2,)
    assert r_zero_players.n_players == 0
    assert jnp.array_equal(r_zero_players(jnp.zeros((0,), dtype=bool)), jnp.zeros(2))


def test_player_compatibility_under_jit_traced_and_closed_over():
    coal = jnp.array([True, False])

    # Broadcast players (size 1 along reduced axis) work inside JIT even with traced player IDs!
    p_bcast = FinitePlayers(jnp.array([[[0, 1]], [[1, 2]]]), latent_n_players=3)  # batch (2, 1)
    g_bcast = _table_game(2, (2, 3), players=p_bcast)
    assert jnp.allclose(
        jax.jit(lambda g, c: g.sum(axis=1)(c))(g_bcast, coal),
        jnp.sum(g_bcast(coal), axis=1),
    )
    # Also works with additive autocast inside JIT when game is rearranged (non-trailing reduced axis)!
    am_bcast = _additive_moebius(jnp.arange(12.0).reshape(3, 2, 2), p_bcast.T)  # players (1, 2)
    assert jnp.allclose(
        jax.jit(lambda g: g.sum(axis=0).contributions())(am_bcast),
        jnp.sum(am_bcast.contributions(), axis=0),
    )

    # Non-broadcast traced player IDs along reduced axis raise TypeError inside JIT.
    p_full = FinitePlayers(
        jnp.array([[[0, 1], [0, 1]], [[1, 2], [1, 2]]]), latent_n_players=3
    )  # batch (2, 2)
    g_full = _table_game(2, (2, 2), players=p_full)
    with pytest.raises(TypeError, match="traced player IDs"):
        jax.jit(lambda g, c: g.sum(axis=1)(c))(g_full, coal)

    # Closed-over concrete equal players succeed inside JIT; unequal raise ValueError.
    @jax.jit
    def jit_closed_equal(c: jax.Array) -> jax.Array:
        return g_full.sum(axis=1)(c)

    assert jnp.allclose(jit_closed_equal(coal), jnp.sum(g_full(coal), axis=1))
    p_bad = FinitePlayers(jnp.array([[[0, 1], [0, 2]], [[1, 2], [1, 2]]]), latent_n_players=3)
    g_bad = _table_game(2, (2, 2), players=p_bad)

    @jax.jit
    def jit_closed_bad(c: jax.Array) -> jax.Array:
        return g_bad.sum(axis=1)(c)

    with pytest.raises(ValueError, match="compatible player sets"):
        jit_closed_bad(coal)

    # Pre-constructed SumReductionGame with full players evaluates and shapes cleanly under JIT.
    r_pre = g_full.sum(axis=1)
    assert jnp.allclose(jax.jit(lambda r, c: r(c))(r_pre, coal), r_pre(coal))
    assert jnp.allclose(jax.jit(lambda r: r[1])(r_pre)(coal), r_pre[1](coal))
    assert jnp.allclose(jax.jit(lambda r: r.reshape(1, 2))(r_pre)(coal), r_pre(coal).reshape(1, 2))


# --- Dtypes and weak typing ---


def test_sum_dtypes_weak_typing_and_x64():
    p = FinitePlayers(2)
    coal = jnp.array([True, False])

    # Weak float32 game becomes strong float32 upon sum (matches jnp.sum), including public `weak_type`.
    g_weak = TableGame(jnp.full((2, 4), 2.0), p)
    g_f16 = TableGame(jnp.full((4,), 3.0, dtype=jnp.float16), p)
    assert g_weak.weak_type is True
    s_weak = g_weak.sum(axis=0)
    assert s_weak.dtype == jnp.float32
    assert s_weak.weak_type is False
    assert s_weak._value_spec().weak_type is False
    assert not s_weak(coal).weak_type
    assert (s_weak + g_f16).dtype == jnp.float32
    assert (s_weak + g_f16)(coal).dtype == jnp.float32

    # Narrow integer promotion and explicit output dtype match jnp.sum under x64=False and x64=True.
    for x64 in (False, True):
        with jax.enable_x64(new_val=x64):
            i8 = _IntTableGame(jnp.arange(8, dtype=jnp.int8).reshape(2, 4), p)
            u8 = _IntTableGame(jnp.arange(8, dtype=jnp.uint8).reshape(2, 4), p)
            f32 = _table_game(2, (2,), players=p, dtype=jnp.float32)
            for expr, expected in [
                (i8.sum(), jnp.sum(i8(coal))),
                (u8.sum(), jnp.sum(u8(coal))),
                (i8.sum(dtype=jnp.int16), jnp.sum(i8(coal), dtype=jnp.int16)),
                (f32.sum(dtype=jnp.float16), jnp.sum(f32(coal), dtype=jnp.float16)),
                (f32.sum(dtype=jnp.complex64), jnp.sum(f32(coal), dtype=jnp.complex64)),
                (i8.sum(axis=()), jnp.sum(i8(coal), axis=())),
            ]:
                assert expr.dtype == expected.dtype
                actual = expr(coal)
                assert actual.dtype == expected.dtype
                assert jnp.allclose(actual, expected)


# --- Additive and K-Additive autocasting ---


def test_sum_reduction_autocasts_additive_and_kadditive_without_eval(monkeypatch):
    p = FinitePlayers(3)
    am = _additive_moebius(jnp.arange(18.0, dtype=jnp.float32).reshape(2, 3, 3), p, perm=[2, 0, 1])
    sm1 = MoebiusGame(
        values=jnp.arange(18.0, dtype=jnp.float32).reshape(2, 3, 3),
        interactions=jnp.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=bool),
        players=p,
        additivity=1,
    )
    sm2 = MoebiusGame(
        values=jnp.arange(12.0, dtype=jnp.float32).reshape(2, 2, 3),
        interactions=jnp.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
        players=p,
        additivity=2,
    )
    all_sets = subset_patterns(3)
    expected_am_eval = jnp.sum(am(all_sets[:, None, None]), axis=2)
    expected_sm1_eval = jnp.sum(sm1(all_sets[:, None, None]), axis=2)
    expected_sm2_eval = jnp.sum(sm2(all_sets[:, None, None]), axis=2)

    def fail_eval(*args, **kwargs):
        msg = "eval must not be called for direct Möbius coefficient access"
        raise AssertionError(msg)

    with monkeypatch.context() as mp:
        mp.setattr(FinitePlayerGame, "eval", fail_eval)

        # AdditiveMoebiusGame -> AdditiveGame refinement.
        s_am = am.sum(axis=1)
        assert isinstance(s_am, SumReductionGame)
        assert isinstance(s_am, AdditiveGame)
        assert s_am.additivity == 1
        assert jnp.allclose(s_am.empty_value(), jnp.sum(am.empty_value(), axis=1))
        assert jnp.allclose(s_am.contributions(), jnp.sum(am.contributions(), axis=1))
        assert jnp.allclose(s_am.coefficients(order=0), jnp.zeros((2, 1)))
        assert jnp.allclose(s_am.coefficients(order=1), jnp.sum(am.contributions(), axis=1))
        assert jnp.allclose(s_am.coefficients(order=2), jnp.zeros((2, 3)))
        assert s_am.coefficients(order=4).shape == (2, 0)

        # MoebiusGame(additivity=1) zero-cost casts to AdditiveGame too!
        s_sm1 = sm1.sum(axis=1)
        assert isinstance(s_sm1, AdditiveGame)
        assert jnp.allclose(s_sm1.empty_value(), jnp.sum(sm1.empty_value(), axis=1))
        assert jnp.allclose(
            s_sm1.contributions(), jnp.sum(flex_cast(sm1, AdditiveGame).contributions(), axis=1)
        )

        # MoebiusGame(additivity=2) refines to KAdditiveGame (not AdditiveGame).
        s_sm2 = sm2.sum(axis=1)
        assert isinstance(s_sm2, KAdditiveGame)
        assert not isinstance(s_sm2, AdditiveGame)
        assert s_sm2.additivity == 2
        assert jnp.allclose(s_sm2.empty_value(), jnp.sum(sm2.empty_value(), axis=1))
        assert jnp.allclose(s_sm2.coefficients(order=1), jnp.sum(sm2.coefficients(order=1), axis=1))
        assert jnp.allclose(s_sm2.coefficients(order=2), jnp.sum(sm2.coefficients(order=2), axis=1))
        assert jnp.allclose(s_sm2.coefficients(order=3), jnp.zeros((2, 1)))

        # Mixed-order explicit interaction query (bool and packed with dirty padding bits).
        mixed = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=bool)
        packed_mixed = pack_bool_coalition(mixed) | jnp.uint8(0b11111000)
        assert jnp.allclose(
            s_sm2.coefficients(mixed[:, None]), s_sm2.coefficients(packed_mixed[:, None])
        )
        assert jnp.allclose(
            s_sm2.coefficients(mixed[:, None]),
            jnp.sum(sm2.coefficients(mixed[:, None, None]), axis=-1),
        )

        # Empty reduction (n_reduced == 0) has additivity 0 and shortcuts lazily.
        am_empty = _additive_moebius(jnp.zeros((2, 0, 3), dtype=jnp.float32), p)
        s_empty = am_empty.sum(axis=1)
        assert isinstance(s_empty, AdditiveGame)
        assert s_empty.additivity == 0
        assert jnp.array_equal(s_empty.empty_value(), jnp.zeros(2))
        assert jnp.array_equal(s_empty.contributions(), jnp.zeros((2, 3)))
        assert jnp.array_equal(s_empty.coefficients(jnp.array([True, False, False])), jnp.zeros(2))

        # Any dtype-changing cast (truncating, narrowing, or widening) declines automatic refinement.
        for dtype_change in (jnp.int32, jnp.float16, jnp.complex64):
            s_changed = am.sum(axis=1, dtype=dtype_change)
            assert type(s_changed) is SumReductionGame
            assert not isinstance(s_changed, KAdditiveGame)
            assert try_cast(s_changed, KAdditiveGame) is None
            assert try_cast(s_changed, AdditiveGame) is None

    # Verify exact Möbius coefficients match full-table inversion!
    assert jnp.allclose(s_am(all_sets[:, None]), expected_am_eval)
    assert jnp.allclose(s_sm1(all_sets[:, None]), expected_sm1_eval)
    assert jnp.allclose(s_sm2(all_sets[:, None]), expected_sm2_eval)
    assert jnp.allclose(
        s_sm2.coefficients(all_sets[:, None]).T,
        _exact_moebius_table(expected_sm2_eval.T, 3),
        atol=1e-5,
    )


def test_refined_reduction_dtype_invariance_rebuilds_and_subclasses():
    from shapiq.game.reduction import (  # noqa: PLC0415
        _AdditiveSumReductionGame,
        _KAdditiveSumReductionGame,
    )

    p = FinitePlayers(2)
    am = _additive_moebius(jnp.array([[0.6, 0.6]]), p)
    sm2 = MoebiusGame(
        values=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        interactions=jnp.array([[1, 0], [1, 1]], dtype=bool),
        players=p,
        additivity=2,
    )
    s_add = am.sum(axis=0)
    assert isinstance(s_add, AdditiveGame)

    # Rebuilding a refined reduction with dtype-changing output or non-additive child raises.
    with pytest.raises(ValueError, match="leaf dtype"):
        replace(s_add, output_dtype=jnp.int32)
    with pytest.raises(ValueError, match="leaf dtype"):
        replace(s_add, output_dtype=jnp.float16)

    table_child = _table_game(2, (1,), players=p)
    with pytest.raises(TypeError, match=r"AdditiveGame|KAdditiveGame"):
        replace(s_add, game=table_child)
    with pytest.raises(TypeError, match=r"AdditiveGame|KAdditiveGame"):
        s_add.with_protected_values({"game": table_child})

    # Narrowing and widening float casts decline refinement; same-dtype cast refines cleanly.
    am_large = _additive_moebius(jnp.array([[70000.0, -69000.0]], dtype=jnp.float32), p)
    s_narrow = am_large.sum(axis=0, dtype=jnp.float16)
    assert type(s_narrow) is SumReductionGame
    assert jnp.allclose(s_narrow.full_value(), jnp.float16(1000.0))

    am_overflow = _additive_moebius(jnp.array([[40000.0, 40000.0]], dtype=jnp.float16), p)
    s_wide = am_overflow.sum(axis=0, dtype=jnp.float32)
    assert type(s_wide) is SumReductionGame
    assert jnp.isinf(s_wide.full_value())

    am_f16 = _additive_moebius(jnp.array([[1.0, 2.0]], dtype=jnp.float16), p)
    s_same = am_f16.sum(axis=0, dtype=jnp.float16)
    assert isinstance(s_same, AdditiveGame)
    assert jnp.allclose(s_same.contributions(), jnp.array([1.0, 2.0], dtype=jnp.float16))

    # Metadata-only subclass keeps its type under autocast and supports explicit flex_cast.
    @dataclass(frozen=True, slots=True, weakref_slot=True)
    class TaggedSum(SumReductionGame[Any, Any]):
        tag: str = "sum"

    sub = TaggedSum(am, axis=0)
    assert type(sub) is TaggedSum
    assert isinstance(flex_cast(sub, AdditiveGame), AdditiveGame)
    assert jnp.allclose(flex_cast(sub, AdditiveGame)(p.full_coalition()), sub(p.full_coalition()))

    # Cooperative subclass providing its own AdditiveGame refinement is autocast directly.
    @dataclass(frozen=True, slots=True, weakref_slot=True)
    class _ScaledSum(SumReductionGame[Any, Any]):
        @override
        def _reduce(self, values: Any) -> Any:
            return 2.0 * super(_ScaledSum, self)._reduce(values)

        @castmethod(AdditiveGame)
        def _as_scaled_additive(self) -> AdditiveGame[Any, Any]:
            child = flex_cast(self.game, AdditiveGame)
            return _AdditiveScaledSum(game=child, _players=self._players)

    @dataclass(frozen=True, slots=True, weakref_slot=True)
    class _AdditiveScaledSum(_ScaledSum, AdditiveGame[Any, Any]):
        @override
        def contributions(self) -> jax.Array:
            return 2.0 * jnp.sum(self.game.contributions(), axis=-2)

    scaled_sub = _ScaledSum(am, axis=0)
    assert isinstance(scaled_sub, _AdditiveScaledSum)
    assert isinstance(scaled_sub, AdditiveGame)
    assert jnp.allclose(scaled_sub.full_value(), 2.4)
    assert jnp.allclose(scaled_sub.contributions(), jnp.array([1.2, 1.2]))

    # Custom zero-cost caster on child game is honored by autocast.
    @dataclass(frozen=True, slots=True, weakref_slot=True)
    class _CustomAdditiveTable[P](TableGame[P]):
        @castmethod(AdditiveGame)
        def _as_additive(self) -> AdditiveGame[P, jax.Array]:
            contribs = self.values[..., 1:] - self.values[..., :1]
            return _additive_moebius(contribs, self.players)

    ct = _CustomAdditiveTable(jnp.array([[0.0, 5.0], [0.0, 7.0]]), FinitePlayers(1))
    s_ct = ct.sum(axis=0)
    assert isinstance(s_ct, AdditiveGame)
    assert jnp.allclose(s_ct.contributions(), jnp.array([12.0]))

    # Direct construction of private refinements preserves exact class.
    assert type(_KAdditiveSumReductionGame(sm2, axis=0)) is _KAdditiveSumReductionGame
    assert type(_AdditiveSumReductionGame(am, axis=0)) is _AdditiveSumReductionGame


# --- Arbitrary game value shapes and structured values ---


def test_sum_supports_structured_values_arbitrary_shapes_and_eager_children():
    p = FinitePlayers(2)
    coal = jnp.array([True, False])

    # 1. MappedGame returning _VectorValue (protected value axis).
    vec_game = MappedGame(
        JaxBroadcastTuple((_table_game(2, (2, 3), players=p),)),
        lambda vs: _VectorValue(jnp.stack([vs[0], vs[0] + 10.0, -vs[0]], axis=-1)),
    )
    s_vec = vec_game.sum(axis=1)
    res_vec = s_vec(coal)
    assert isinstance(res_vec, _VectorValue)
    assert res_vec.shape == (2,)
    assert res_vec.values.shape == (2, 3)
    assert jnp.allclose(res_vec.values, jnp.sum(vec_game(coal).values, axis=1))

    # 1b. Game-valued MappedGame returning TableGame: summing returns SumReductionGame, not TableGame.
    inner_p = FinitePlayers(1)
    game_valued = MappedGame(
        JaxBroadcastTuple((_table_game(2, (2, 3), players=p),)),
        lambda vs: TableGame(jnp.stack([jnp.zeros_like(vs[0]), vs[0]], axis=-1), inner_p),
    )
    eval_inner = game_valued(coal)
    assert isinstance(eval_inner, TableGame)
    res_game = game_valued.sum(axis=1)(coal)
    assert isinstance(res_game, SumReductionGame)
    assert not isinstance(res_game, TableGame)
    assert res_game.shape == (2,)
    assert jnp.allclose(res_game.full_value(), jnp.sum(eval_inner.full_value(), axis=1))

    # 2. Structured AdditiveGame (_VectorValue) preserves type across sum, contributions, and zeros.
    v_empty = _VectorValue(jnp.arange(12.0).reshape(2, 3, 2))
    v_contribs = jnp.arange(24.0).reshape(2, 3, 2, 2)
    v_add = _VectorAdditiveGame(empty=v_empty, contribs=v_contribs, _players=p)
    sv = v_add.sum(axis=1)
    assert isinstance(sv, AdditiveGame)
    assert isinstance(sv.empty_value(), _VectorValue)
    assert jnp.allclose(sv.empty_value().values, jnp.sum(v_empty.values, axis=1))
    assert isinstance(sv.contributions(), _VectorValue)
    assert jnp.allclose(sv.contributions().values, jnp.sum(v_contribs, axis=1))
    above = sv.coefficients(order=2)
    assert isinstance(above, _VectorValue)
    assert above.values.shape == (2, 1, 2)
    assert jnp.array_equal(above.values, jnp.zeros((2, 1, 2)))

    # 3. Mixed-dtype structured pair: same leaf dtypes refine; secondary-field changes decline refinement.
    with jax.enable_x64(new_val=False):
        pair_same = _PairAdditiveGame(
            empty=_StructuredPair(
                count=jnp.ones((2, 3), dtype=jnp.float32),
                vector=jnp.ones((2, 3, 2), dtype=jnp.int32),
            ),
            contribs=_StructuredPair(
                count=jnp.ones((2, 3, 2), dtype=jnp.float32),
                vector=jnp.ones((2, 3, 2, 2), dtype=jnp.int32),
            ),
            _players=p,
        )
        sp = pair_same.sum(axis=1)
        assert isinstance(sp, AdditiveGame)
        for out in (
            sp(coal),
            sp.empty_value(),
            sp.contributions(),
            sp.coefficients(order=0),
            sp.coefficients(order=1),
            sp.coefficients(order=2),
            sp.coefficients(jnp.array([True, True])),
        ):
            assert isinstance(out, _StructuredPair)
            assert out.count.dtype == jnp.float32
            assert out.vector.dtype == jnp.int32

    pair_mixed = _PairAdditiveGame(
        empty=_StructuredPair(
            count=jnp.zeros((1,), dtype=jnp.float16),
            vector=jnp.zeros((1, 1), dtype=jnp.int16),
        ),
        contribs=_StructuredPair(
            count=jnp.zeros((1, 2), dtype=jnp.float16),
            vector=jnp.array([[[2049], [-2048]]], dtype=jnp.int16),
        ),
        _players=p,
    )
    s_pair_f16 = pair_mixed.sum(axis=0, dtype=jnp.float16)
    assert type(s_pair_f16) is SumReductionGame
    assert not isinstance(s_pair_f16, KAdditiveGame)
    assert jnp.array_equal(s_pair_f16.full_value().vector, jnp.array([1.0], dtype=jnp.float16))

    # 4. Eager-only AdditiveGame child whose `_eval_bool` calls `np.asarray` still constructs & evaluates `.sum()`.
    @dataclass(frozen=True)
    class _EagerAdditiveGame(
        AdditiveGame[Any, _VectorValue], JaxAxisProtected["_EagerAdditiveGame"]
    ):
        contribs: jax.Array
        _players: FinitePlayers[Any]
        protected_axes: ClassVar[dict[str, int]] = {"contribs": 2, "_players": 0}
        broadcasted_batch: ClassVar[set[str]] = {"_players"}

        @property
        @override
        def players(self) -> FinitePlayers[Any]:
            return self._players

        @override
        def contributions(self) -> _VectorValue:
            return _VectorValue(self.contribs)

        @override
        def _eval_bool(self, coalition: jax.Array) -> _VectorValue:
            mask = np.asarray(coalition)
            batch = np.broadcast_shapes(self.shape, mask.shape[:-1])
            mask_b = jnp.asarray(np.broadcast_to(mask, (*batch, self.n_players)))[..., None]
            contribs_b = jnp.broadcast_to(
                self.contribs, (*batch, self.n_players, self.contribs.shape[-1])
            )
            return _VectorValue(jnp.sum(jnp.where(mask_b, contribs_b, 0.0), axis=-2))

        @override
        def _eval_packed(self, coalition: jax.Array) -> _VectorValue:
            return self._eval_bool(
                jnp.unpackbits(coalition, axis=-1, bitorder="little")[..., : self.n_players].astype(
                    bool
                )
            )

    eager_g = _EagerAdditiveGame(
        contribs=jnp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=jnp.float32),
        _players=p,
    )
    s_eager = eager_g.sum(axis=0)
    assert isinstance(s_eager, SumReductionGame)
    assert jnp.array_equal(s_eager.full_value().values, jnp.array([4.0, 6.0], dtype=jnp.float32))


# --- Composition with arithmetic, ndlike shaping, TableGame fast cast, JIT/grad/vmap ---


def test_composition_with_arithmetic_nested_sums_and_ndlike_shaping():
    p = FinitePlayers(3)
    p1 = FinitePlayers(1)
    am = _additive_moebius(jnp.arange(36.0).reshape(2, 3, 2, 3), p)
    t1 = _table_game(3, (2, 3), players=p)
    t2 = _table_game(3, (2,), players=p) + 1.0
    coal = jnp.array([True, False, True])

    # Nested reductions preserve AdditiveGame and values.
    nested_am = am.sum(axis=0).sum(axis=1)
    assert isinstance(nested_am, AdditiveGame)
    assert nested_am.shape == (3,)
    assert jnp.allclose(nested_am(coal), jnp.sum(am(coal), axis=(0, 2)))
    assert jnp.allclose(nested_am.contributions(), jnp.sum(am.contributions(), axis=(0, 2)))

    # Sum composed with affine, product, division, power.
    combo = 2.0 * (am.sum(axis=(0, 2)) + 1.5) / 3.0
    assert isinstance(combo, AdditiveGame)
    assert jnp.allclose(combo(coal), 2.0 * (jnp.sum(am(coal), axis=(0, 2)) + 1.5) / 3.0)
    assert jnp.allclose(combo.contributions(), 2.0 * jnp.sum(am.contributions(), axis=(0, 2)) / 3.0)

    prod_sum = (am * am).sum(axis=(0, 2))
    assert isinstance(prod_sum, KAdditiveGame)
    assert prod_sum.additivity == 2
    assert jnp.allclose(prod_sum(coal), jnp.sum((am * am)(coal), axis=(0, 2)))

    arith_sum = (t1.sum(axis=1) + t2) ** 2
    assert isinstance(arith_sum, ArithmeticGame)
    assert jnp.allclose(arith_sum(coal), (jnp.sum(t1(coal), axis=1) + t2(coal)) ** 2)

    # Full ndlike shaping suite on SumReductionGame.
    r = _table_game(2, (2, 3, 4)).sum(axis=1)  # shape (2, 4)
    base = r(jnp.array([True, False]))
    assert jnp.allclose(r[1](jnp.array([True, False])), base[1])
    assert jnp.allclose(r[:, 2](jnp.array([True, False])), base[:, 2])
    assert jnp.allclose(r.reshape(4, 2)(jnp.array([True, False])), base.reshape(4, 2))
    assert jnp.allclose(r.T(jnp.array([True, False])), base.T)
    assert jnp.allclose(r.mT(jnp.array([True, False])), base.mT)
    assert jnp.allclose(jf.jax_expand_dims(r, 0)(jnp.array([True, False])), base[None, ...])
    assert jnp.allclose(
        jf.jax_moveaxis(r, 0, 1)(jnp.array([True, False])), jnp.moveaxis(base, 0, 1)
    )
    assert jnp.allclose(
        jf.jax_swapaxes(r, 0, 1)(jnp.array([True, False])), jnp.swapaxes(base, 0, 1)
    )
    stacked = jf.jax_stack([r, r])
    assert stacked.shape == (2, 2, 4)
    assert jnp.allclose(stacked(jnp.array([True, False])), jnp.stack([base, base]))
    concat = jf.jax_concatenate([r, r], axis=0)
    assert concat.shape == (4, 4)
    assert jnp.allclose(concat(jnp.array([True, False])), jnp.concatenate([base, base], axis=0))

    # LCCA stacking: _AdditiveSumReductionGame + _KAdditiveSumReductionGame -> _KAdditiveSumReductionGame.
    am2 = _additive_moebius(jnp.arange(12.0).reshape(2, 2, 3), p)
    mg2 = MoebiusGame(
        values=jnp.arange(12.0).reshape(2, 2, 3),
        interactions=jnp.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=bool),
        players=p,
        additivity=2,
    )
    stacked_k = jf.jax_stack([am2.sum(axis=1), mg2.sum(axis=1)])
    assert isinstance(stacked_k, KAdditiveGame)
    assert not isinstance(stacked_k, AdditiveGame)
    assert stacked_k.additivity == 2
    assert stacked_k.shape == (2, 2)

    # Stacking reductions with differing child dtypes or weak_type is rejected.
    g_f16 = TableGame(jnp.array([[0.0, 2048.0], [0.0, 1.0]], dtype=jnp.float16), p1)
    g_f32 = TableGame(jnp.array([[0.0, 2048.0], [0.0, 1.0]], dtype=jnp.float32), p1)
    with pytest.raises(ValueError, match="differing value dtypes"):
        jf.jax_stack([g_f16.sum(), g_f32.sum()])

    g_weak_f32 = TableGame(jnp.full((1, 2), 2048.0), p1)
    g_strong_f32 = TableGame(jnp.full((1, 2), 2048.0, dtype=jnp.float32), p1)
    with pytest.raises(ValueError, match="differing value dtypes"):
        jf.jax_stack([g_weak_f32.sum(), g_strong_f32.sum()])


def test_table_game_fast_cast_and_pytree_grad_vmap(monkeypatch):
    p = FinitePlayers(2)
    t1 = _table_game(2, (2, 3), players=p)
    t2 = _table_game(2, (2,), players=p)
    all_sets = subset_patterns(2)

    s_plain = t1.sum(axis=1)
    s_arith_in = (2.0 * t1 + 1.0).sum(axis=1)
    s_arith_out = t1.sum(axis=1) + t2

    def fail_eval(*args, **kwargs):
        msg = "Fast TableGame cast must not call eval()"
        raise AssertionError(msg)

    with monkeypatch.context() as mp:
        mp.setattr(FinitePlayerGame, "eval", fail_eval)
        for expr in (s_plain, s_arith_in, s_arith_out):
            casted = TableGame.from_game(expr)
            assert isinstance(casted, TableGame)
            assert casted.shape == (2,)
            assert casted.players is p
            comp = ExactComputer.from_game(expr)
            assert jnp.array_equal(comp.table.values, casted.values)

    for expr in (s_plain, s_arith_in, s_arith_out):
        casted = TableGame.from_game(expr)
        assert jnp.allclose(casted(all_sets[:, None]), expr(all_sets[:, None]))

    # Clean pytree leaves (no stray masks/scalars), grad, and vmap.
    leaves_plain = jax.tree.leaves(s_plain)
    assert len(leaves_plain) == len(jax.tree.leaves(t1)) + len(jax.tree.leaves(p))

    coal = jnp.array([True, False])
    grad_vals = jax.grad(lambda vals: TableGame(vals, p).sum()(coal))(t1.values)
    expected_grad = jax.grad(lambda vals: jnp.sum(TableGame(vals, p)(coal)))(t1.values)
    assert jnp.allclose(grad_vals, expected_grad)

    vmapped = jax.vmap(lambda vals: TableGame(vals, p).sum(axis=0)(coal))(t1.values)
    assert jnp.allclose(vmapped, jnp.sum(t1(coal), axis=1))
