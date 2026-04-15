"""Tests that every concrete public subclass is registered in its module's __all__.

These guard against adding a new subclass without listing it in __all__.
"""

from __future__ import annotations

import importlib
import inspect

import pytest


def _find_concrete_subclasses(module: object, base: type) -> set[str]:
    """Return names of all concrete, public subclasses of base visible in module."""
    return {
        name
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, base)
        and obj is not base
        and not inspect.isabstract(obj)
        and not name.startswith("_")
    }


@pytest.mark.parametrize(
    ("module_path", "base_path"),
    [
        ("shapiq.approximator", "shapiq.approximator.base:Approximator"),
        ("shapiq.explainer", "shapiq.explainer.base:Explainer"),
        ("shapiq.imputer", "shapiq.imputer.base:Imputer"),
    ],
    ids=["approximator", "explainer", "imputer"],
)
def test_all_concrete_subclasses_in_all(module_path: str, base_path: str) -> None:
    """Every concrete public subclass must appear in its module's __all__."""
    module = importlib.import_module(module_path)
    base_module_path, base_class_name = base_path.split(":")
    base: type = getattr(importlib.import_module(base_module_path), base_class_name)

    concrete = _find_concrete_subclasses(module, base)
    exported = set(module.__all__)
    missing = concrete - exported

    pkg_init = f"src/shapiq/{module_path.split('.')[-1]}/__init__.py"
    assert not missing, (
        f"Concrete subclasses not listed in {module_path}.__all__: {missing}. "
        f"Add them to {pkg_init}."
    )
