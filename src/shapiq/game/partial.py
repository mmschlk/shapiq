"""Small executable sketch of staged computation and game projections."""

from __future__ import annotations

from copy import copy
from types import GenericAlias
from typing import TYPE_CHECKING, Any, cast, get_args, get_origin, get_type_hints

from .game import Game

if TYPE_CHECKING:
    from collections.abc import Callable

type PartialShape = tuple[int | None, ...] | None


def result_contract(function: Callable, source: object | None = None) -> object:
    """Read a callable's declared result without running it."""
    infer_result = getattr(function, "infer_result", None)
    if source is not None and callable(infer_result):
        return infer_result(source)
    if isinstance(function, Partial) or hasattr(function, "result_type"):
        return function.result_type
    if isinstance(function, type):
        return function
    return get_type_hints(function).get("return", object)


def matches(actual: object, target: object) -> bool:
    """Match nominal types and nested partial result types.

    Parameter signatures remain attached to the callables and their static
    types. Runtime routing in this sketch uses the nested result structure.
    """
    actual_origin = get_origin(actual) or actual
    target_origin = get_origin(target) or target
    if not isinstance(actual_origin, type) or not isinstance(target_origin, type):
        msg = "Lifting currently supports nominal classes and Partial[..., T] contracts."
        raise TypeError(msg)
    if not issubclass(actual_origin, target_origin):
        return False
    if issubclass(target_origin, Partial) and get_args(target):
        return bool(get_args(actual)) and matches(get_args(actual)[-1], get_args(target)[-1])
    return True


def contract(value: object) -> object:
    """Describe a value, retaining each partial's declared result structure."""
    if isinstance(value, Partial):
        return GenericAlias(type(value), (Ellipsis, value.result_type))
    return type(value)


def refine_shape(expected: PartialShape, actual: PartialShape) -> PartialShape:
    """Combine compatible shape knowledge, rejecting contradictory dimensions."""
    if expected is None:
        return actual
    if actual is None:
        return expected
    if len(expected) != len(actual) or any(
        a is not None and b is not None and a != b for a, b in zip(expected, actual, strict=True)
    ):
        msg = f"Result shape {actual} contradicts declared shape {expected}."
        raise ValueError(msg)
    return tuple(b if a is None else a for a, b in zip(expected, actual, strict=True))


class Partial[**P, T]:
    """A reusable suspended stage with a declared result and batch metadata."""

    def __init__(
        self,
        function: Callable[P, T],
        *,
        result_type: object | None = None,
        shape: PartialShape = None,
    ) -> None:
        """Store the stage and its metadata without running the callable."""
        self._function = function
        self.result_type = result_contract(function) if result_type is None else result_type
        self.shape = shape

    def bind(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Execute exactly this stage and return its declared result."""
        result = self._function(*args, **kwargs)
        if not matches(contract(result), self.result_type):
            msg = f"Result {contract(result)} does not satisfy {self.result_type}."
            raise TypeError(msg)
        shape = refine_shape(self.shape, getattr(result, "shape", None))
        if isinstance(result, Partial) and shape != result.shape:
            # Refine a fresh wrapper; the original family remains reusable.
            result = copy(result)
            result.shape = shape
        return result

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Supply this stage's inputs; shorthand for ``bind``."""
        return self.bind(*args, **kwargs)

    def map[U](
        self,
        function: Callable[[T], U],
        *,
        result_type: object | None = None,
        shape: PartialShape = None,
    ) -> Partial[P, U]:
        """Postcompose one stage, preserving a callback's entire result."""

        def run(*args: P.args, **kwargs: P.kwargs) -> U:
            return function(self.bind(*args, **kwargs))

        output_shape = getattr(function, "output_shape", None)
        if shape is None and callable(output_shape):
            shape = output_shape(self.shape)
        return partial(
            run,
            result_type=result_contract(function, self.result_type)
            if result_type is None
            else result_type,
            shape=shape,
        )


class PGame[**P, T](Partial[P, T]):
    """A partial whose eventual result is a game batch."""

    def eval(self, coalition: object) -> Partial[P, Any]:
        """Record coalition evaluation through all surrounding game stages."""
        from shapiq.transformation import lifted  # noqa: PLC0415

        @lifted
        def evaluate(game: Game) -> object:
            return game.eval(coalition)

        return evaluate(cast("Partial[P, Layers[Game]]", self))


def partial_class(result_type: object) -> type[Partial]:
    """Select a stage facade using only its eventual result type."""
    leaf = result_type
    while isinstance(origin := get_origin(leaf) or leaf, type) and issubclass(origin, Partial):
        if not get_args(leaf):
            break
        leaf = get_args(leaf)[-1]
    return PGame if matches(leaf, Game) else Partial


def partial[**P, T](
    function: Callable[P, T],
    *,
    result_type: object | None = None,
    shape: PartialShape = None,
) -> Partial[P, T]:
    """Wrap a stage, selecting the game facade from its result contract."""
    result_type = result_contract(function) if result_type is None else result_type
    return partial_class(result_type)(function, result_type=result_type, shape=shape)


def delay[T](
    function: Callable[[], T],
    *,
    result_type: object | None = None,
    shape: PartialShape = None,
) -> Partial[[], T]:
    """Suspend a zero-parameter computation until explicitly called."""
    return partial(function, result_type=result_type, shape=shape)


def defer[T](value: T) -> Partial[[], T]:
    """Add a zero-parameter stage around an existing value."""
    return delay(lambda: value, result_type=contract(value), shape=getattr(value, "shape", None))


type Layers[T] = T | Partial[..., Layers[T]]
type LiftedGame[G: Game | PGame] = G | PGame[..., LiftedGame[G]]
