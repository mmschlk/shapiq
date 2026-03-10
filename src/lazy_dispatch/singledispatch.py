"""A lazy version of functools.singledispatch."""

from __future__ import annotations

from functools import reduce, singledispatch, update_wrapper
import operator
from typing import TYPE_CHECKING, Any, get_args, overload

from lazy_dispatch.isinstance import (
    LazyType,
    _find_closest_string_type,
    _is_union_type,
    _split_lazy_type,
    lazy_issubclass,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import UnionType

type RegistrationFunction = Callable[[type], Any]


def is_valid_dispatch_type(cls: LazyType) -> bool:
    """Check if cls is a valid dispatch type."""
    if isinstance(cls, (type, str)):
        return True
    if isinstance(cls, tuple):
        return all(is_valid_dispatch_type(c) for c in cls)
    if _is_union_type(cls):
        return all(is_valid_dispatch_type(c) for c in get_args(cls))
    return False


def first_argument[T](x: T, *args: Any, **kwargs: Any) -> T:  # noqa: ANN401, ARG001
    """Return the first argument."""
    return x


class lazydispatch[A: Callable, Out]:  # noqa: N801
    """A lazy version of functools.singledispatch that also works with string types."""

    def __new__(
        cls,
        func: A | None = None,
        *,
        dispatch_on: Callable = first_argument,
    ) -> lazydispatch[A, Out] | Callable[[A], lazydispatch[A, Out]]:
        """Create a new lazy_singledispatch or return a decorator."""
        if func is None:

            def decorator(func: A) -> lazydispatch[A, Out]:
                return cls(func, dispatch_on=dispatch_on)

            return decorator

        return super().__new__(cls)

    def __init__(
        self,
        func: A | None = None,
        *,
        dispatch_on: Callable = first_argument,
    ) -> None:
        """Initialize the lazy_singledispatch instance."""
        if func is None:
            msg = "func must be provided"
            raise ValueError(msg)
        update_wrapper(self, func, updated=())

        self._singledispatcher = singledispatch(func)
        self.funcname = getattr(func, "__name__", "singledispatch function")
        self.string_registry: dict[str, Callable] = {}
        self.delayed_registration_registry: dict[str | type, RegistrationFunction] = {}
        self.dispatch_on = dispatch_on

    def dispatch(self, cls: type, *, delayed_register: bool = True) -> Callable[..., Out]:
        """Find the best available function for the given type or string."""
        delayed_registration_registry = self.delayed_registration_registry
        string_registry = self.string_registry
        if delayed_register and len(delayed_registration_registry) > 0:
            active_registrations: dict[str | type, RegistrationFunction] = {
                t: registration_func
                for t, registration_func in delayed_registration_registry.items()
                if lazy_issubclass(cls, t)
            }

            for t, registration_func in active_registrations.items():
                registration_func(cls)
                del delayed_registration_registry[t]

        if len(string_registry) > 0:
            closest = _find_closest_string_type(cls, self.string_registry)
            if closest is not None:
                real_type, string_type = closest
                registration_func = string_registry.pop(string_type)
                self.eager_register(real_type, registration_func)

        return self._singledispatcher.dispatch(cls)

    def eager_register(self, cls: type | UnionType | Callable, func: Callable | None = None) -> Callable:
        """Eagerly register a new implementation for the given type or union type."""
        return self._singledispatcher.register(cls, func)  # type: ignore[arg-type]

    def register(self, cls: LazyType | Callable, func: Callable | None = None) -> Callable:
        """Register a new implementation for the given type or string."""
        if is_valid_dispatch_type(cls):  # type: ignore[arg-type]
            if func is None:
                return lambda f: self.register(cls, f)
        else:
            if func is not None:
                msg = f"Invalid first argument to `register()`. {cls!r} is not a class, string, tuple or union type."
                raise TypeError(msg)
            ann = getattr(cls, "__annotations__", {})
            if not ann:
                msg = (
                    f"Invalid first argument to `register()`: {cls!r}. "
                    f"Use either `@register(some_class)` or plain `@register` "
                    f"on an annotated function."
                )
                raise TypeError(msg)
            func = cls  # type: ignore[assignment]

            argname, cls = next(iter(func.__annotations__.items()))
            if not is_valid_dispatch_type(cls):
                if _is_union_type(cls) or isinstance(cls, tuple):
                    msg = f"Invalid annotation for {argname!r}. {cls!r} not all arguments are classes or strings."
                    raise TypeError(msg)
                msg = f"Invalid annotation for {argname!r}. {cls!r} is not a class or string."
                raise TypeError(msg)

        types, strings = _split_lazy_type(cls)  # type: ignore[arg-type]

        if len(types) > 0:
            # Use reduce with operator.or_ to dynamically create a Union (PEP 604 style) and avoid private API usage.
            union_type = reduce(operator.or_, types)

            self.eager_register(union_type, func)

        if len(strings) > 0:
            for s in strings:
                self.string_registry[s] = func  # type: ignore[assignment]

        return func  # type: ignore[return-value]

    @overload
    def delayed_register(self, cls: LazyType) -> Callable[[RegistrationFunction], RegistrationFunction]: ...

    @overload
    def delayed_register(self, cls: RegistrationFunction) -> RegistrationFunction: ...

    @overload
    def delayed_register(self, cls: LazyType, func: RegistrationFunction) -> RegistrationFunction: ...

    def delayed_register(
        self,
        cls: LazyType | RegistrationFunction,
        func: RegistrationFunction | None = None,
    ) -> RegistrationFunction | Callable[[RegistrationFunction], RegistrationFunction]:
        """Register a delayed registration function."""
        if is_valid_dispatch_type(cls):  # type: ignore[arg-type]
            if func is None:
                return lambda f: self.delayed_register(cls, f)  # type: ignore[arg-type]
        else:
            if func is not None:
                msg = (
                    f"Invalid first argument to `delayed_register()`. "
                    f"{cls!r} is not a class, string, tuple or union type."
                )
                raise TypeError(msg)
            ann = getattr(cls, "__annotations__", {})
            if not ann:
                msg = (
                    f"Invalid first argument to `delayed_register()`: {cls!r}. "
                    f"Use either `@delayed_register(some_class)` or plain `@delayed_register` "
                    f"on an annotated function."
                )
                raise TypeError(msg)
            func = cls  # type: ignore[assignment]

            argname, cls = next(iter(func.__annotations__.items()))
            if not is_valid_dispatch_type(cls):
                if _is_union_type(cls) or isinstance(cls, tuple):
                    msg = f"Invalid annotation for {argname!r}. {cls!r} not all arguments are classes or strings."
                    raise TypeError(msg)
                msg = f"Invalid annotation for {argname!r}. {cls!r} is not a class or string."
                raise TypeError(msg)

        types, strings = _split_lazy_type(cls)  # type: ignore[arg-type]

        for t in types:
            self.delayed_registration_registry[t] = func  # type: ignore[assignment]

        for s in strings:
            self.delayed_registration_registry[s] = func  # type: ignore[assignment]

        return func  # type: ignore[return-value]

    def __call__(self, *args: Any, **kwargs: Any) -> Out:  # noqa: ANN401
        """Call the appropriate registered function based on the type of the first argument."""
        if not args:
            msg = f"{self.funcname} requires at least 1 positional argument"
            raise TypeError(msg)
        return self.dispatch(self.dispatch_on(*args, **kwargs).__class__)(*args, **kwargs)
