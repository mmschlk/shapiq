"""The minimal functional flow through family-specific partial stages."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_type, cast

import pytest

from shapiq.partial import Partial

if TYPE_CHECKING:
    from shapiq.transformation import Transformation


@dataclass(frozen=True)
class Scale[T](Partial[T]):
    function: Callable[[float], T]
    label: str = "scale"

    def __call__(self, *, scale: float) -> T:
        return self.function(scale)

    def map[U](self, function: Callable[[T], U]) -> "Scale[U]":
        return Scale(lambda scale: function(self(scale=scale)), label=self.label)


@dataclass(frozen=True)
class Offset[T](Partial[T]):
    function: Callable[[int], T]
    label: str = "offset"

    def __call__(self, *, offset: int) -> T:
        return self.function(offset)

    def map[U](self, function: Callable[[T], U]) -> "Offset[U]":
        return Offset(lambda offset: function(self(offset=offset)), label=self.label)


@dataclass(frozen=True)
class Thunk[T](Partial[T]):
    function: Callable[[], T]

    def __call__(self) -> T:
        return self.function()

    def map[U](self, function: Callable[[T], U]) -> "Thunk[U]":
        return Thunk(lambda: function(self()))


class CallOnly(Partial[int]):
    def __call__(self) -> int:
        return 1


class MapOnly(Partial[int]):
    def map[U](self, function: Callable[[int], U]) -> Thunk[U]:
        return Thunk(lambda: function(1))


@pytest.mark.parametrize("family", [Partial, CallOnly, MapOnly])
def test_both_operations_are_required(family):
    with pytest.raises(TypeError, match="abstract"):
        family()


def test_call_executes_one_stage_and_preserves_keyword_signatures():
    calls = []

    def outer(scale: float) -> Offset[float]:
        calls.append("scale")

        def inner(offset: int) -> float:
            calls.append("offset")
            return scale + offset

        return Offset(inner)

    family = Scale(outer)
    assert_type(family, Scale[Offset[float]])
    # Deliberately exercise invalid calls at the dynamic API boundary.
    call = cast("Callable[..., object]", family)
    with pytest.raises(TypeError):
        call(2.0)
    with pytest.raises(TypeError):
        call(offset=3)
    assert calls == []

    remaining = family(scale=2.0)
    assert_type(remaining, Offset[float])
    assert calls == ["scale"]
    assert remaining(offset=3) == 5.0
    assert calls == ["scale", "offset"]
    assert family(scale=4.0)(offset=3) == 7.0


def test_map_is_deferred_and_repackages_the_family_without_mutation():
    calls = []

    def source(offset: int) -> int:
        calls.append("source")
        return offset + 1

    def render(value: int) -> str:
        calls.append("render")
        return str(value)

    original = Offset(source, label="kept by this family")
    mapped = original.map(render)
    assert_type(mapped, Offset[str])
    assert isinstance(mapped, Offset)
    assert mapped is not original
    assert mapped.label == original.label
    assert calls == []

    assert mapped(offset=2) == "3"
    assert calls == ["source", "render"]
    assert original(offset=4) == 5
    assert mapped(offset=6) == "7"
    assert calls == ["source", "render", "source", "source", "render"]


def test_explicit_nested_mapping_preserves_each_family_and_stage():
    calls = []

    def make_offset(scale: float) -> Offset[float]:
        calls.append("scale")

        def compute(offset: int) -> float:
            calls.append("compute")
            return scale + offset

        return Offset(compute)

    def render(value: float) -> str:
        calls.append("render")
        return str(value)

    transformation: Transformation[float, str] = render
    source = Scale(make_offset)
    result = source.map(lambda stage: stage.map(transformation))
    assert_type(result, Scale[Offset[str]])
    assert calls == []

    remaining = result(scale=2.0)
    assert_type(remaining, Offset[str])
    assert calls == ["scale"]
    assert remaining(offset=3) == "5.0"
    assert calls == ["scale", "compute", "render"]


def test_partial_aware_handler_receives_exactly_the_selected_stage():
    calls = []

    def expensive(offset: int) -> int:
        pytest.fail("The handler must receive the partial without executing it.")

    inner = Offset(expensive)
    source = Scale(lambda scale: inner)

    def replace(stage: Offset[int]) -> Thunk[str]:
        assert stage is inner
        calls.append("replace")
        return Thunk(lambda: "replacement")

    result = source.map(replace)
    assert_type(result, Scale[Thunk[str]])
    assert calls == []
    remaining = result(scale=2.0)
    assert_type(remaining, Thunk[str])
    assert calls == ["replace"]
    assert remaining() == "replacement"


def test_callback_result_adds_a_stage_without_flattening_it():
    calls = []

    def introduce(value: int) -> Scale[str]:
        calls.append("introduce")
        return Scale(lambda scale: str(value * scale))

    result = Offset(lambda offset: offset + 1).map(introduce)
    assert_type(result, Offset[Scale[str]])
    assert calls == []

    remaining = result(offset=2)
    assert_type(remaining, Scale[str])
    assert calls == ["introduce"]
    assert remaining(scale=4.0) == "12.0"


def test_zero_parameter_stage_defers_work_until_called():
    calls = []

    def source() -> int:
        calls.append("source")
        return 4

    result = Thunk(source).map(lambda value: Thunk(lambda: value + 1))
    assert_type(result, Thunk[Thunk[int]])
    assert calls == []
    remaining = result()
    assert calls == ["source"]
    assert remaining() == 5


def test_map_obeys_identity_and_composition():
    source = Offset(lambda offset: offset + 1)

    def double(value: int) -> int:
        return 2 * value

    def render(value: int) -> str:
        return f"value: {value}"

    assert source.map(lambda value: value)(offset=3) == source(offset=3)
    composed = source.map(double).map(render)
    assert_type(composed, Offset[str])
    assert composed(offset=3) == source.map(lambda value: render(double(value)))(offset=3)


def test_transformations_are_ordinary_functions_without_automatic_traversal():
    def increment(value: int) -> int:
        return value + 1

    function: Transformation[int, int] = increment
    assert function is increment
    assert function(2) == 3

    stage = Offset(lambda offset: offset)
    call = cast("Callable[..., object]", function)
    with pytest.raises(TypeError):
        call(stage)
    assert stage.map(function)(offset=2) == 3
