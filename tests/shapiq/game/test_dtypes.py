"""Value dtype ownership, native storage casts and lazy casts."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flextype import try_cast
from ndlike import jax_functions as jf
from ndlike.protected_axis.jax import JaxAxisProtected
from ndlike.tuple.jax import JaxBroadcastTuple

from shapiq.coalition.coalition import pack_bool_coalition
from shapiq.game import (
    AdditiveGame,
    CastGame,
    KAdditiveGame,
    MappedGame,
    TableGame,
    table as table_module,
)
from shapiq.game.discretederivative import DiscreteDerivative
from shapiq.game.fourier import FourierGame
from shapiq.game.moebius import AdditiveMoebiusGame, MoebiusGame
from shapiq.players import FinitePlayers
from shapiq.value import VectorValue


@pytest.mark.parametrize("game_cls", [TableGame, MoebiusGame, AdditiveMoebiusGame, FourierGame])
def test_native_astype_casts_only_stored_values(game_cls):
    players = FinitePlayers(1)
    values = jnp.array([2.0, 3.0])
    interactions = jnp.array([[False], [True]])
    if game_cls is TableGame:
        game = TableGame(values, players)
    else:
        game = game_cls(values=values, players=players, interactions=interactions, additivity=1)
    converted = jax.jit(lambda g: g.astype(jnp.float16))(game)
    assert type(converted) is game_cls
    assert converted.dtype == converted.full_value().dtype == jnp.float16
    assert not converted.weak_type
    assert game.astype(None) is game
    assert game.astype(game.dtype) is game
    copied = game.astype(None, copy=True)
    assert copied is not game
    assert copied.values is not game.values
    assert converted.players.ids.dtype == players.ids.dtype
    if game_cls is not TableGame:
        assert converted.interactions.dtype == interactions.dtype


@pytest.mark.parametrize("game_cls", [TableGame, AdditiveMoebiusGame])
@pytest.mark.parametrize("dtype", [None, jnp.float16])
def test_native_astype_moves_all_dynamic_fields(game_cls, dtype):
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two CPU devices")
    cpu0, cpu1 = devices[:2]
    players = FinitePlayers(1)
    if game_cls is TableGame:
        game = TableGame(jnp.array([0.0, 3.0]), players)
    else:
        game = AdditiveMoebiusGame(jnp.array([3.0]), players, jnp.array([[True]]))
    game = game.to_device(cpu0)
    moved = game.astype(dtype, device=cpu1)
    assert moved.values.device == cpu1
    assert moved.players.ids.device == cpu1
    if game_cls is AdditiveMoebiusGame:
        assert moved.interactions.device == cpu1
    assert moved.full_value().device == cpu1
    assert moved.full_value().item() == 3.0
    assert moved.dtype == (game.dtype if dtype is None else dtype)


def test_structured_basis_cast_uses_value_astype():
    players = FinitePlayers(1)
    interactions = players.full_coalition()[None]
    values = VectorValue(jnp.array([[2.0, 3.0]]))
    game = AdditiveMoebiusGame(values=values, interactions=interactions, players=players)
    converted = game.astype(jnp.bfloat16)

    assert isinstance(converted.values, VectorValue)
    assert converted.players is players
    assert converted.interactions is interactions
    assert converted.dtype == converted.full_value().dtype == jnp.bfloat16


def test_cast_game_keeps_native_packed_evaluation_and_uses_fast_table_cast(monkeypatch):
    players = FinitePlayers(1)
    table = TableGame(jnp.full((2,), 3.0), players)
    weak_game = table + 1
    assert weak_game.weak_type
    converted = weak_game.astype(weak_game.dtype)
    assert isinstance(converted, CastGame)
    assert converted.game is weak_game
    assert not converted.weak_type
    assert converted.astype(None) is converted
    assert converted.astype(converted.dtype) is converted
    assert converted.astype(None, copy=True) is not converted

    def unexpected_bool(*_):
        pytest.fail("Packed evaluation must forward packed coalitions.")

    with monkeypatch.context() as patch:
        patch.setattr(TableGame, "_eval_bool", unexpected_bool)
        value = jax.jit(lambda g, c: g(c))(converted, pack_bool_coalition(players.full_coalition()))
        assert value == 4
        assert not value.weak_type

    fast = try_cast(converted, TableGame, max_cost=1.0)
    assert fast is not None
    assert fast.value.players is players
    assert jnp.array_equal(fast.value.values, jnp.full((2,), 4.0))
    assert not fast.value.weak_type
    assert try_cast(converted.astype(jnp.int8), TableGame) is None


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.complex64])
def test_cast_game_refines_kadditive_and_additive(dtype):
    players = FinitePlayers(2)
    interactions = jnp.array([[False, False], [True, False], [False, True], [True, True]])
    kn = MoebiusGame(jnp.array([1.0, 2.0, 3.0, 4.0]), players, interactions, additivity=2)
    additive = MoebiusGame(kn.values[:-1], players, interactions[:-1], additivity=1)

    for child, capability in ((kn, KAdditiveGame), (additive, AdditiveGame)):
        converted = CastGame(child, dtype)
        assert isinstance(converted, capability)
        assert converted.additivity == child.additivity
        assert converted.empty_value().dtype == dtype
        assert converted.empty_value() == 1
        assert converted.coefficients(order=3).dtype == dtype
        assert converted.coefficients(order=3).shape == (0,)
        assert jnp.array_equal(converted.coefficients(order=1), jnp.array([2, 3], dtype=dtype))
        assert jnp.array_equal(
            converted.coefficients(interactions), child.coefficients(interactions).astype(dtype)
        )
    assert isinstance(converted, AdditiveGame)
    assert jnp.array_equal(converted.contributions(), jnp.array([2, 3], dtype=dtype))

    zero_player = AdditiveMoebiusGame(kn.values[:1], FinitePlayers(0), interactions[:1, :0])
    converted = CastGame(zero_player, dtype)
    assert isinstance(converted, AdditiveGame)
    assert converted.additivity == 0
    assert converted.contributions().shape == (0,)


@pytest.mark.parametrize("dtype", [jnp.int8, jnp.bool_])
def test_non_linear_casts_do_not_refine(dtype):
    players = FinitePlayers(1)
    child = AdditiveMoebiusGame(jnp.array([0.5]), players, jnp.array([[True]]))
    converted = CastGame(child, dtype)
    assert type(converted) is CastGame
    assert try_cast(converted, KAdditiveGame, max_cost=0.0) is None
    assert try_cast(converted, AdditiveGame, max_cost=0.0) is None


def test_cast_game_preserves_intermediate_casts():
    table = TableGame(jnp.array([0.5, 2.75]), FinitePlayers(1))
    integer = (table + 0).astype(jnp.int8)
    restored = integer.astype(jnp.float32)
    assert restored.astype("float32") is restored
    assert restored.game is integer
    assert restored.full_value() == 2


def test_cast_game_places_values_coefficients_and_fast_tables_on_target_device():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two CPU devices")
    cpu0, cpu1 = devices[:2]
    players = FinitePlayers(1)
    values = jnp.array([2.0, 3.0])
    table = TableGame(values, players).to_device(cpu0)
    stored = CastGame(table, jnp.float16, device=cpu1)
    assert stored.full_value().device == cpu1
    assert stored.astype(jnp.float16).game is stored
    result = try_cast(stored, TableGame, max_cost=1.0)
    assert result is not None
    assert result.value.players.ids.device == result.value.full_value().device == cpu1

    child = AdditiveMoebiusGame(values, players, jnp.array([[False], [True]])).to_device(cpu0)
    refined = CastGame(child, jnp.float16, device=cpu1)
    assert isinstance(refined, AdditiveGame)
    assert refined.empty_value().device == cpu1
    assert refined.contributions().device == cpu1
    assert refined.coefficients(order=1).device == cpu1
    assert refined.coefficients(players.full_coalition()).device == cpu1


def test_cast_game_normalizes_64bit_target_when_disabled():
    with jax.enable_x64(new_val=False):
        game = CastGame(TableGame(jnp.array([1.0]), FinitePlayers(0)), jnp.float64)
        assert game.target_dtype == game.dtype == game.full_value().dtype == jnp.float32


@pytest.mark.parametrize("dtype", [np.timedelta64, np.longdouble, "invalid"])
def test_cast_game_rejects_invalid_jax_target_dtype(dtype):
    if dtype is np.longdouble and np.dtype(dtype) == np.dtype(np.float64):
        pytest.skip("NumPy longdouble aliases float64 on this platform")
    table = TableGame(jnp.array([1.0]), FinitePlayers(0))
    with pytest.raises(TypeError, match=r"target_dtype .* supported by JAX"):
        CastGame(table, dtype)


@pytest.mark.parametrize("game_cls", [MoebiusGame, FourierGame])
def test_cast_game_fast_table_path_declines_expensive_child(game_cls, monkeypatch):
    child = game_cls(jnp.array([1.0]), FinitePlayers(1), jnp.array([[True]]), additivity=1)
    assert try_cast(child, TableGame, max_cost=5.0) is not None

    def unexpected(*_):
        pytest.fail("The cast must decline before building a child table.")

    monkeypatch.setattr(import_module(game_cls.__module__), "_require_table_values", unexpected)
    assert try_cast(CastGame(child, jnp.float16), TableGame, max_cost=1.0) is None


def test_table_materialization_checks_dtype_before_allocating_coalitions(monkeypatch):
    table = TableGame(jnp.array([1.0]), FinitePlayers(0))

    def unexpected(*_):
        pytest.fail("The unsupported value dtype must be rejected before allocating the powerset.")

    monkeypatch.setattr(table_module, "subset_patterns", unexpected)
    assert try_cast(CastGame(table, jnp.int8), TableGame) is None


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.int8, jnp.bool_, None])
def test_derivative_spec_matches_signed_subset_sum(dtype):
    players = FinitePlayers(1)
    table = TableGame(jnp.arange(2.0) if dtype is not None else jnp.full((2,), 2.0), players)
    if dtype in (jnp.float16, jnp.bfloat16):
        table = table.astype(dtype)
    child = CastGame(table, dtype) if dtype in (jnp.int8, jnp.bool_) else table
    game = DiscreteDerivative(child, players.full_coalition(), 1)
    value = jax.jit(lambda g: g.empty_value())(game)
    assert value == (0 if dtype is None else 1)
    assert (
        value.dtype == game.dtype == (jnp.asarray(1).dtype if dtype == jnp.bool_ else child.dtype)
    )
    assert not value.weak_type
    assert game._value_spec() == jax.ShapeDtypeStruct((), value.dtype)
    converted = game.astype(jnp.float32, copy=True)
    assert isinstance(converted, CastGame)
    assert converted.game.interaction is not game.interaction
    assert jnp.array_equal(converted.game.interaction, game.interaction)
    assert converted.empty_value().dtype == jnp.float32


def test_derivative_strengthens_weak_vector_values():
    table = TableGame(jnp.full((2,), 1.0), FinitePlayers(1))
    child = MappedGame(
        JaxBroadcastTuple((table,)),
        lambda vs: VectorValue(jnp.stack([vs[0], vs[0]], axis=-1)),
    )
    game = DiscreteDerivative(child, table.players.full_coalition(), 1)
    value = game.empty_value()
    assert child.weak_type
    assert isinstance(value, VectorValue)
    assert not value.weak_type
    assert game._value_spec() == jax.ShapeDtypeStruct((), value.dtype)
    assert game.dtype == value.dtype


def test_reduction_cast_changes_the_result_dtype():
    players = FinitePlayers(1)
    game = TableGame(jnp.array([[0.0, 1024.0], [0.0, 0.25]]), players).sum()
    converted = game.astype(jnp.float16)
    assert isinstance(converted, CastGame)
    assert converted.game is game
    assert converted.players is game.players
    assert converted.full_value() == game.full_value().astype(jnp.float16)


def test_placement_only_cast_preserves_additive_view_and_table_colocation():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires at least two logical CPU devices")
    cpu0, cpu1 = devices[:2]
    players = FinitePlayers(1).to_device(cpu0)
    interactions = jax.device_put(jnp.array([[True]]), cpu0)
    values = jax.device_put(jnp.array([2.5], dtype=jnp.float32), cpu0)
    child = AdditiveMoebiusGame(values, players, interactions)

    placed = CastGame(child, None, device=cpu1)
    assert isinstance(placed, AdditiveGame)
    assert placed.contributions().devices() == {cpu1}

    table = TableGame.from_game(placed)
    assert table.values.devices() == {cpu1}
    assert table.players.devices() == {cpu1}
    assert table.full_value() == jnp.float32(2.5)


def test_discrete_derivative_structured_int8_matches_declared_spec():
    @dataclass(frozen=True, slots=True)
    class _Int8Vec(JaxAxisProtected["_Int8Vec"]):
        values: jax.Array
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}
        permitted_functions: ClassVar[set[Any]] = {jf.jax_multiply, jf.jax_sum}

    table = TableGame(jnp.array([1.0, 3.0], dtype=jnp.float32), FinitePlayers(1))
    child = MappedGame(
        JaxBroadcastTuple((table,)),
        lambda vs: _Int8Vec(jnp.stack([vs[0], vs[0]], axis=-1).astype(jnp.int8)),
        output_dtype=jnp.int8,
    )
    game = DiscreteDerivative(child, table.players.full_coalition(), 1)
    value = game.empty_value()
    assert isinstance(value, _Int8Vec)
    assert game.dtype == jnp.int8
    assert value.dtype == jnp.int8
