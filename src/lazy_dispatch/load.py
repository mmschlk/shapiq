"""Lazy import utilities."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType


def lazy_import(name: str, package: str | None = None, register: bool = False) -> ModuleType:
    """Lazily import a module."""
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.find_spec(name, package=package)

    if spec is None or spec.loader is None:
        msg = f"Module {name} not found"
        raise ImportError(msg)

    loader: importlib.util.LazyLoader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    loader.exec_module(module)
    return module


@overload
def lazy_callable(
    module: ModuleType | str,
    attrs: str,
    package: str | None = None,
    register: bool = False,
) -> Callable: ...


@overload
def lazy_callable(
    module: ModuleType | str,
    attrs: Iterable[str],
    package: str | None = None,
    register: bool = False,
) -> list[Callable]: ...


def lazy_callable(
    module: ModuleType | str,
    attrs: str | Iterable[str],
    package: str | None = None,
    register: bool = False,
) -> Callable | list[Callable]:
    """Lazily get a callable attribute from a module or module name."""
    if isinstance(module, str):
        module = lazy_import(module, package=package, register=register)

    if isinstance(attrs, str):

        def fn(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            return getattr(module, attrs)(*args, **kwargs)

        return fn

    return [lazy_callable(module, attr) for attr in attrs]
