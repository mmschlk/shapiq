"""Target-directed lifting by mapping over surrounding partial stages."""

from __future__ import annotations

from inspect import signature
from types import GenericAlias
from typing import TYPE_CHECKING, Any, Protocol, get_args, get_origin, get_type_hints, overload

from shapiq.game.partial import Layers, Partial, PartialShape, contract, matches, partial_class

if TYPE_CHECKING:
    from collections.abc import Callable


def unknown_shape(_shape: PartialShape) -> PartialShape:
    """Conservatively describe an operation without a shape rule."""
    return None


class Transformation[A, B]:
    """A unary handler that preserves the stages surrounding its input type."""

    def __init__(
        self,
        function: Callable[[A], B],
        *,
        output_shape: Callable[[PartialShape], PartialShape] = unknown_shape,
    ) -> None:
        """Read a unary handler's input and result contracts."""
        parameters = list(signature(function).parameters)
        hints = get_type_hints(function)
        if len(parameters) != 1 or parameters[0] not in hints or "return" not in hints:
            msg = "A lifted handler needs one annotated input and an annotated result."
            raise TypeError(msg)
        self._function = function
        self.input_type = hints[parameters[0]]
        self.result_type = hints["return"]
        self.output_shape = output_shape

    def infer_result(self, source: object) -> object:
        """Replace the target in a declared stack, using only type metadata."""
        if matches(source, self.input_type):
            return self.result_type
        origin = get_origin(source) or source
        if isinstance(origin, type) and issubclass(origin, Partial) and get_args(source):
            result = self.infer_result(get_args(source)[-1])
            return GenericAlias(partial_class(result), (Ellipsis, result))
        msg = f"Cannot reach {self.input_type} through {source}."
        raise TypeError(msg)

    @overload
    def __call__(self, value: A, /) -> B: ...

    @overload
    def __call__[**P](self, value: Partial[P, A], /) -> Partial[P, B]: ...

    @overload
    def __call__[**P](self, value: Partial[P, Layers[A]], /) -> Partial[P, Layers[B]]: ...

    def __call__(self, value: Any, /) -> Any:
        """Apply at the declared target, mapping over each surrounding stage."""
        if matches(contract(value), self.input_type):
            return self._function(value)
        if isinstance(value, Partial):
            result_type = self.infer_result(value.result_type)
            return value.map(self, result_type=result_type, shape=self.output_shape(value.shape))
        msg = f"Expected {self.input_type}, got {type(value)}."
        raise TypeError(msg)


class _Decorator(Protocol):
    def __call__[A, B](self, function: Callable[[A], B], /) -> Transformation[A, B]: ...


@overload
def lifted[A, B](function: Callable[[A], B], /) -> Transformation[A, B]: ...


@overload
def lifted(*, output_shape: Callable[[PartialShape], PartialShape]) -> _Decorator: ...


def lifted(
    function: Callable | None = None,
    /,
    *,
    output_shape: Callable[[PartialShape], PartialShape] = unknown_shape,
) -> Transformation | _Decorator:
    """Decorate a handler for transparent, target-directed partial lifting."""

    def decorate[A, B](handler: Callable[[A], B]) -> Transformation[A, B]:
        return Transformation(handler, output_shape=output_shape)

    return decorate if function is None else decorate(function)
