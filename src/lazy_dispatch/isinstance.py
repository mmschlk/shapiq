"""A lazy version of isinstance."""

from __future__ import annotations

from types import UnionType
from typing import Any, Union, get_args, get_origin

type LazyType = type | str | UnionType | tuple[LazyType, ...]


def _is_union_type(cls: Any) -> bool:  # noqa: ANN401
    return get_origin(cls) in {Union, UnionType}


def _split_lazy_type(lazy_type: LazyType) -> tuple[set[type], set[str]]:
    """Split classinfo into a set of types and a set of strings."""
    if isinstance(lazy_type, str):
        return set(), {t.strip() for t in lazy_type.split("|")}
    if isinstance(lazy_type, type):
        return {lazy_type}, set()
    if isinstance(lazy_type, tuple):
        types: set[type] = set()
        strings: set[str] = set()
        for item in lazy_type:
            t, s = _split_lazy_type(item)
            types.update(t)
            strings.update(s)
        return types, strings
    if _is_union_type(lazy_type):
        types = set()
        strings = set()
        for arg in get_args(lazy_type):
            t, s = _split_lazy_type(arg)
            types.update(t)
            strings.update(s)
        return types, strings

    msg = f"Invalid classinfo: {lazy_type!r}"
    raise TypeError(msg)


def _find_matching_string_type(cls: type, string_types: set[str] | dict[str, Any]) -> str | None:
    """Check if the type's name matches any of the strings."""
    module = cls.__module__
    qualname = cls.__qualname__

    if module == "builtins":
        for s in string_types:
            if qualname == s:
                return s

    for s in string_types:
        if f"{module}.{qualname}" == s:
            return s

    return None


def _find_closest_string_type(cls: type, string_types: set[str] | dict[str, Any]) -> tuple[type, str] | None:
    """Check if any type in the MRO matches any of the strings."""
    mro = cls.__mro__
    for super_cls in mro:
        matching_type = _find_matching_string_type(super_cls, string_types)
        if matching_type is not None:
            return super_cls, matching_type
    return None


def lazy_isinstance(obj: object, class_or_tuple: LazyType, /) -> bool:
    """A lazy version of isinstance."""
    types, strings = _split_lazy_type(class_or_tuple)
    if len(types) > 0 and isinstance(obj, tuple(types)):
        return True

    if len(strings) > 0:
        return _find_closest_string_type(type(obj), strings) is not None

    return False


def lazy_issubclass(cls: type, class_or_tuple: LazyType, /) -> bool:
    """A lazy version of issubclass."""
    types, strings = _split_lazy_type(class_or_tuple)
    if len(types) > 0 and issubclass(cls, tuple(types)):
        return True

    if len(strings) > 0:
        return _find_closest_string_type(cls, strings) is not None

    return False
