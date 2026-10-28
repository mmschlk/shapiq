"""Alternative native evaluators and finite-coalition game dispatch."""

import _py_abc
from abc import abstractmethod
from dataclasses import dataclass
from inspect import isabstract
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Float, Shaped, TypeCheckError
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.coalition.coalition import pack_bool_coalition
from shapiq.game import (
    BoolFinitePlayerGame,
    FinitePlayerGame,
    PackedFinitePlayerGame,
    finite as finite_game,
)
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked


class _BooleanGame(BoolFinitePlayerGame[Any, jax.Array]):
    def _eval_bool(self, coalition):
        return self.values[..., 0] + jnp.sum(coalition, axis=-1)


class _PackedGame(PackedFinitePlayerGame[Any, jax.Array]):
    def _eval_packed(self, coalition):
        remainder = self.players.n_players % 8
        if remainder:
            coalition = coalition.at[..., -1].set(
                coalition[..., -1] & jnp.uint8((1 << remainder) - 1)
            )
        return self.values[..., 0] + jnp.sum(jax.lax.population_count(coalition), axis=-1)


class _BothGame(FinitePlayerGame[Any, jax.Array]):
    _eval_bool = _BooleanGame._eval_bool
    _eval_packed = _PackedGame._eval_packed


class _PureABCMeta(_py_abc.ABCMeta, type(FinitePlayerGame)):
    """Run the pure-Python ABC calculation after the normal metaclass hooks."""


def _array_game_class(base, *, metaclass=None):
    if metaclass is None:
        metaclass = type(base)

    # Declare representation fields on the final class, as required by ndlike.
    @typechecked
    @dataclass
    class ArrayGame(base, JaxAxisProtected[jax.Array], metaclass=metaclass):
        values: Float[jax.Array, "*batch 1"]
        _players: Shaped[FinitePlayers, "#*batch"]
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

        @classmethod
        def _pytree_child_names(cls):
            return ("values", "_players")

        @property
        def players(self) -> FinitePlayers:
            return self._players

    return ArrayGame


@pytest.fixture(scope="module", params=[type(FinitePlayerGame), _PureABCMeta])
def game_metaclass(request):
    return request.param


@pytest.fixture(scope="module", params=[_BooleanGame, _PackedGame, _BothGame])
def game_class(request, game_metaclass):
    return _array_game_class(request.param, metaclass=game_metaclass)


@pytest.mark.parametrize(
    ("base", "required"),
    [
        (FinitePlayerGame, {"_eval_bool", "_eval_packed"}),
        (BoolFinitePlayerGame, {"_eval_bool"}),
        (PackedFinitePlayerGame, {"_eval_packed"}),
    ],
)
def test_native_hooks_are_ordinary_abstract_requirements(base, required, game_metaclass):
    game_class = _array_game_class(base, metaclass=game_metaclass)
    assert isabstract(base)
    assert isabstract(game_class)
    assert game_class.__abstractmethods__ == required
    with pytest.raises(TypeError, match="|".join(sorted(required))):
        game_class(jnp.zeros(1), FinitePlayers(3))


def test_implementing_the_chosen_native_hooks_allows_concrete_games(game_class):
    assert not isabstract(game_class)
    assert not game_class.__abstractmethods__
    game_class(jnp.zeros(1), FinitePlayers(3))


def test_game_pytrees_include_player_ids_without_expanding_shared_players(game_class):
    metadata = ["a", "b", "c", "d"]
    players = FinitePlayers(jnp.array([1, 3]), players=metadata)
    game = game_class(jnp.zeros((2, 3, 1)), players)
    leaves = jax.tree.leaves(game)
    assert len(leaves) == 2
    assert leaves[0] is game.values
    assert leaves[1] is players.ids
    restored = jax.jit(lambda candidate: candidate)(game)
    assert restored.shape == (2, 3)
    assert restored.players.shape == ()
    assert restored.players.players is players.players
    assert restored.players.latent_n_players == 4


@pytest.mark.parametrize("base", [_BooleanGame, _PackedGame, _BothGame])
def test_inherited_evaluation_hooks_count(base, game_metaclass):
    class Inherited(base):
        pass

    game_class = _array_game_class(Inherited, metaclass=game_metaclass)
    assert not isabstract(game_class)
    game = game_class(jnp.zeros(1), FinitePlayers(3))
    assert float(game(jnp.array([True, False, True]))) == 2


def test_inheriting_only_a_conversion_default_stays_abstract(game_metaclass):
    class Intermediate(BoolFinitePlayerGame):
        pass

    class Descendant(Intermediate):
        pass

    game_class = _array_game_class(Descendant, metaclass=game_metaclass)
    assert isabstract(game_class)
    with pytest.raises(TypeError, match="_eval_bool"):
        game_class(jnp.zeros(1), FinitePlayers(3))


def test_restoring_an_abstract_base_hook_restores_abstractness(game_metaclass):
    class Reset(_BooleanGame):
        _eval_bool = FinitePlayerGame._eval_bool

    assert isabstract(_array_game_class(Reset, metaclass=game_metaclass))


def test_overriding_the_other_format_does_not_satisfy_the_native_requirement(game_metaclass):
    class MissingBoolean(BoolFinitePlayerGame):
        _eval_packed = _PackedGame._eval_packed

    class MissingPacked(PackedFinitePlayerGame):
        _eval_bool = _BooleanGame._eval_bool

    for base, required in [(MissingBoolean, "_eval_bool"), (MissingPacked, "_eval_packed")]:
        game_class = _array_game_class(base, metaclass=game_metaclass)
        assert game_class.__abstractmethods__ == {required}
        with pytest.raises(TypeError, match=required):
            game_class(jnp.zeros(1), FinitePlayers(3))


def test_the_common_base_requires_both_native_hooks(game_metaclass):
    class MissingPacked(FinitePlayerGame):
        _eval_bool = _BooleanGame._eval_bool

    class MissingBoolean(FinitePlayerGame):
        _eval_packed = _PackedGame._eval_packed

    for base, required in [(MissingBoolean, "_eval_bool"), (MissingPacked, "_eval_packed")]:
        game_class = _array_game_class(base, metaclass=game_metaclass)
        assert game_class.__abstractmethods__ == {required}


def test_an_abstract_hook_does_not_satisfy_the_requirement(game_metaclass):
    class AbstractBoolean(_BooleanGame):
        @abstractmethod
        def _eval_bool(self, coalition):
            pass

    game_class = _array_game_class(AbstractBoolean, metaclass=game_metaclass)
    assert game_class.__abstractmethods__ == {"_eval_bool"}


def test_other_abstract_requirements_are_not_cleared(game_metaclass):
    class ExtraRequirement(_BooleanGame):
        @abstractmethod
        def extra(self):
            pass

    game_class = _array_game_class(ExtraRequirement, metaclass=game_metaclass)
    assert game_class.__abstractmethods__ == {"extra"}


def test_explicitly_abstract_hook_remains_required_even_with_other_native_hook(game_metaclass):
    class AbstractBoolean(_BothGame):
        @abstractmethod
        def _eval_bool(self, coalition):
            pass

    game_class = _array_game_class(AbstractBoolean, metaclass=game_metaclass)
    assert game_class.__abstractmethods__ == {"_eval_bool"}


def test_overriding_only_eval_does_not_supply_a_native_hook(game_metaclass):
    class EvalOnly(FinitePlayerGame):
        def eval(self, coalition):
            return jnp.array(0)

    assert isabstract(_array_game_class(EvalOnly, metaclass=game_metaclass))


def test_reset_without_a_dataclass_decorator_restores_abstractness(game_class):
    # In particular, no dataclass/update_abstractmethods call may repair this class.
    class Reset(game_class):
        values: Float[jax.Array, "*batch 1"]
        _players: Shaped[FinitePlayers, "#*batch"]
        _eval_bool = FinitePlayerGame._eval_bool
        _eval_packed = FinitePlayerGame._eval_packed

    assert isabstract(Reset)
    assert Reset.__abstractmethods__ == {"_eval_bool", "_eval_packed"}
    with pytest.raises(TypeError, match="_eval_bool"):
        Reset(jnp.zeros(1), FinitePlayers(3))


def test_explicit_abstract_default_alias_is_not_replaced(game_metaclass):
    class ExplicitAbstract(_PackedGame):
        _eval_bool = FinitePlayerGame._eval_bool

    game_class = _array_game_class(ExplicitAbstract, metaclass=game_metaclass)
    assert game_class.__abstractmethods__ == {"_eval_bool"}


def test_both_native_hooks_can_be_implemented_directly(game_metaclass):
    game_class = _array_game_class(_BothGame, metaclass=game_metaclass)
    assert game_class._eval_bool is _BooleanGame._eval_bool
    assert game_class._eval_packed is _PackedGame._eval_packed


def test_abstract_hooks_follow_normal_mro_priority(game_metaclass):
    class AbstractPacked(FinitePlayerGame):
        @abstractmethod
        def _eval_packed(self, coalition):
            pass

    class Combined(AbstractPacked, _BooleanGame):
        pass

    game_class = _array_game_class(Combined, metaclass=game_metaclass)
    assert game_class.__abstractmethods__ == {"_eval_packed"}


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("n_players", [0, 1, 7, 8, 9, 17])
def test_evaluation_broadcasts_and_converts_only_when_needed(
    game_class, packed, compiled, n_players
):
    players = FinitePlayers(2 * jnp.arange(n_players), latent_n_players=2 * n_players + 3)
    game = game_class(jnp.array([10.0, 20.0]).reshape((2, 1, 1)), players)
    membership = np.arange(3 * n_players).reshape((3, n_players)) % 2 == 0
    coalition = jnp.asarray(membership)
    if packed:
        coalition = pack_bool_coalition(coalition)
        if n_players % 8:
            padding = jnp.uint8(255 ^ ((1 << (n_players % 8)) - 1))
            coalition = coalition.at[..., -1].set(coalition[..., -1] | padding)

    def evaluate(g, c):
        return g(c)

    if compiled:
        evaluate = jax.jit(evaluate)
    expected = np.array([10.0, 20.0])[:, None] + membership.sum(axis=-1)[None, :]
    result = evaluate(game, coalition)
    assert result.shape == (2, 3)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_evaluation_handles_empty_batches(game_class, packed, compiled):
    game = game_class(jnp.zeros((0, 1, 1)), FinitePlayers(2))
    coalition = jnp.ones((3, 2), dtype=bool)
    if packed:
        coalition = pack_bool_coalition(coalition)

    def evaluate(g, c):
        return g(c)

    if compiled:
        evaluate = jax.jit(evaluate)
    assert evaluate(game, coalition).shape == (0, 3)


@pytest.mark.parametrize("packed", [False, True])
def test_two_native_hooks_do_not_convert_input(monkeypatch, packed):
    game = _array_game_class(_BothGame)(jnp.zeros(1), FinitePlayers(3))
    coalition = jnp.array([True, False, True])
    if packed:
        coalition = pack_bool_coalition(coalition)

    def unexpected_conversion(*args, **kwargs):
        msg = "A native representation must be evaluated without conversion"
        raise AssertionError(msg)

    monkeypatch.setattr(finite_game, "pack_bool_coalition", unexpected_conversion)
    monkeypatch.setattr(finite_game, "unpack_bool_coalition", unexpected_conversion)
    assert float(game(coalition)) == 2


def test_boolean_fallback_packs_and_clears_padding(monkeypatch):
    game = _array_game_class(_PackedGame)(jnp.zeros(1), FinitePlayers(9))
    received = []

    def record(coalition):
        received.append(coalition)
        return jnp.array(9.0)

    monkeypatch.setattr(game, "_eval_packed", record)
    assert float(game(jnp.ones(9, dtype=bool))) == 9
    assert len(received) == 1
    assert received[0].dtype == jnp.uint8
    assert received[0].tolist() == [255, 1]


def test_packed_fallback_unpacks_local_count_and_ignores_padding(monkeypatch):
    game = _array_game_class(_BooleanGame)(jnp.zeros(1), FinitePlayers(9, latent_n_players=100))
    received = []

    def record(coalition):
        received.append(coalition)
        return jnp.array(9.0)

    monkeypatch.setattr(game, "_eval_bool", record)
    assert float(game(jnp.array([255, 255], dtype=jnp.uint8))) == 9
    assert len(received) == 1
    assert received[0].dtype == jnp.bool_
    assert received[0].tolist() == [True] * 9


@typechecked
@dataclass
class _StructuredValue(JaxAxisProtected[jax.Array]):
    count: Shaped[jax.Array, "*batch"]
    vector: Shaped[jax.Array, "*batch 2"]
    protected_axes: ClassVar[dict[str, int]] = {"count": 0, "vector": 1}


@pytest.mark.parametrize("compiled", [False, True])
def test_evaluation_preserves_shaped_structured_values(compiled):
    class StructuredGame(BoolFinitePlayerGame[Any, _StructuredValue]):
        def _eval_bool(self, coalition):
            count = self.values[..., 0] + jnp.sum(coalition, axis=-1)
            return _StructuredValue(count, jnp.stack((count, count + 1), axis=-1))

    game = _array_game_class(StructuredGame)(jnp.zeros((2, 1, 1)), FinitePlayers(3))

    def evaluate(g, c):
        return g(c)

    if compiled:
        evaluate = jax.jit(evaluate)
    result = evaluate(game, jnp.full((3, 1), 5, dtype=jnp.uint8))
    assert isinstance(result, _StructuredValue)
    assert result.shape == (2, 3)
    assert bool(jnp.all(result.count == 2))
    assert result.vector.shape == (2, 3, 2)
    assert bool(jnp.all(result.vector == jnp.array([2, 3])))


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("result_shape", [(), (1,), (2, 1), (2, 3, 4)])
def test_evaluation_rejects_wrong_logical_value_shape(packed, compiled, result_shape):
    class WrongShapeGame(_BothGame):
        def _eval_bool(self, coalition):
            return jnp.zeros(result_shape)

        _eval_packed = _eval_bool

    game = _array_game_class(WrongShapeGame)(jnp.zeros((2, 1, 1)), FinitePlayers(3))
    coalition = jnp.ones((3, 3), dtype=bool)
    if packed:
        coalition = pack_bool_coalition(coalition)

    def evaluate(g, c):
        return g(c)

    if compiled:
        evaluate = jax.jit(evaluate)
    with pytest.raises(TypeCheckError, match="return value"):
        evaluate(game, coalition)


@pytest.mark.parametrize("result", [{"count": jnp.array(2)}, 2, [2]])
def test_evaluation_rejects_unshaped_values(result):
    class UnshapedGame(_BooleanGame):
        def _eval_bool(self, coalition):
            return result

    game = _array_game_class(UnshapedGame)(jnp.zeros(1), FinitePlayers(3))
    with pytest.raises(TypeCheckError, match="return value"):
        game(jnp.ones(3, dtype=bool))


@pytest.mark.parametrize(
    "coalition",
    [
        jnp.ones(3),
        jnp.ones(4, dtype=bool),
        jnp.ones(2, dtype=jnp.uint8),
        jnp.ones((), dtype=bool),
        [True, False, True],
    ],
)
def test_evaluation_rejects_invalid_storage(game_class, coalition):
    game = game_class(jnp.zeros(1), FinitePlayers(3))
    assert not game.is_valid(coalition)
    with pytest.raises(TypeCheckError):
        game(coalition)


@pytest.mark.parametrize("packed", [False, True])
def test_validation_uses_game_batch_not_only_shared_players(game_class, packed):
    game = game_class(jnp.zeros((2, 1)), FinitePlayers(3))
    coalition = jnp.ones((3, 3), dtype=bool)
    if packed:
        coalition = pack_bool_coalition(coalition)
    assert game.players.is_valid(coalition)
    assert not game.is_valid(coalition)
    with pytest.raises(ValueError, match="Inadmissible coalition batch"):
        game(coalition)
