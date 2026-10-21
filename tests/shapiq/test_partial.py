"""Behavioral checks for the staged lifting sketch."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from shapiq.game import Game, PGame
from shapiq.partial import Partial, defer, delay, partial
from shapiq.transformation import lifted


def test_binding_executes_one_stage_and_keeps_keyword_signature():
    calls = []

    def outer(*, scale: int) -> Partial[..., int]:
        calls.append("outer")
        return partial(lambda *, offset: scale + offset, result_type=int)

    family = partial(outer)
    with pytest.raises(TypeError):
        family(2)
    assert calls == []
    remaining = family.bind(scale=2)
    assert calls == ["outer"]
    assert remaining(offset=3) == 5
    assert family(scale=4)(offset=3) == 7


def test_delay_and_map_execute_only_on_reduction():
    calls = []

    def make() -> int:
        calls.append("make")
        return 4

    pending = delay(make).map(lambda n: defer(n + 1), result_type=Partial[..., int])
    assert calls == []
    result = pending()
    assert isinstance(result, Partial)
    assert calls == ["make"]
    assert result() == 5


def test_partial_target_replaces_only_the_matched_stage():
    calls = []

    @lifted
    def relabel(stage: Partial[..., int]) -> Partial[..., str]:
        calls.append("relabel")
        return partial(lambda *, label: f"{label}: {stage(amount=2)}", result_type=str)

    inner = partial(lambda *, amount: amount + 1, result_type=int)
    tower = defer(defer(inner))
    result = relabel(tower)
    assert calls == []
    next_stage = result()
    assert calls == []
    replaced = next_stage()
    assert calls == ["relabel"]
    assert replaced(label="value") == "value: 3"
    assert relabel(inner)(label="direct") == "direct: 3"


def test_new_partial_layers_are_inserted_intact_at_arbitrary_depth():
    calls = []

    @lifted
    def introduce(value: int) -> Partial[..., Partial[..., int]]:
        calls.append(value)
        return defer(defer(value + 1))

    tower = 7
    for _ in range(12):
        tower = defer(tower)
    result = introduce(tower)
    for _ in range(12):
        assert calls == []
        result = result()
    assert calls == [7]
    assert result()() == 8
    assert calls == [7]


def test_unreachable_target_rejected_without_executing_input():
    calls = []

    def source() -> str:
        calls.append("source")
        return "wrong leaf"

    @lifted
    def increment(value: int) -> int:
        return value + 1

    with pytest.raises(TypeError, match="Cannot reach"):
        increment(defer(delay(source)))
    assert calls == []


def test_map_of_lifted_handler_preserves_the_remaining_stack():
    @lifted
    def increment(value: int) -> int:
        return value + 1

    result = defer(defer(2)).map(increment)
    assert isinstance(result(), Partial)
    assert result()() == 3
    # A further transformation can still discover the nested target.
    assert increment(result)()() == 4


def test_game_facade_remains_a_target_after_a_previous_lift():
    def expensive() -> Game:
        pytest.fail("A partial-aware handler must receive the game stage intact.")

    @lifted
    def identity(game: Game) -> Game:
        return game

    @lifted
    def inspect_stage(game: PGame[..., Game]) -> int:
        assert isinstance(game, PGame)
        return 42

    mapped = identity(defer(defer(delay(expensive))))
    assert inspect_stage(mapped)()() == 42


@dataclass
class Batch:
    shape: tuple[int, ...]


def test_shape_refinement_survives_nested_lifting():
    @lifted(output_shape=lambda shape: shape)
    def identity(value: Batch) -> Batch:
        return value

    def select(*, size: int) -> Partial[..., Batch]:
        return delay(lambda: Batch((size,)), result_type=Batch, shape=(size,))

    family = partial(select, shape=(None,))
    graph = identity(defer(family))
    assert graph.shape == (None,)
    selected = graph()(size=3)
    assert selected.shape == (3,)
    assert selected().shape == (3,)
    assert family.map(identity).shape == (None,)


def test_known_shape_is_preserved_without_mutating_a_shared_stage():
    inner = delay(lambda: Batch((3,)), result_type=Batch)
    outer = delay(lambda: inner, result_type=Partial[..., Batch], shape=(3,))
    refined = outer()
    assert inner.shape is None
    assert refined.shape == (3,)
    assert refined().shape == (3,)

    wrong = delay(lambda: Batch((4,)), result_type=Batch, shape=(3,))
    with pytest.raises(ValueError, match="contradicts"):
        wrong()


def test_declared_result_structure_is_checked_on_binding():
    wrong = delay(lambda: "oops", result_type=int)
    with pytest.raises(TypeError, match="does not satisfy"):
        wrong()
