"""Tests that every concrete public subclass is registered in its module's ``__all__``.

These tests act as a guard: if someone adds a new concrete subclass (e.g. a new
Approximator) without listing it in the corresponding ``__all__``, the test
fails and prompts the author to update the public API declaration.
"""

from __future__ import annotations

import importlib
import inspect

import pytest


def _find_concrete_subclasses(module: object, base: type) -> set[str]:
    """Return names of all concrete, public subclasses of *base* visible in *module*.

    Note: only classes that have been explicitly imported into *module*'s namespace
    are detected. A class that lives in a sub-module but is not re-exported at the
    package level will be silently missed.

    Args:
        module: The Python module to inspect.
        base: The base class to check against.

    Returns:
        A set of class names that are concrete (non-abstract), non-private
        subclasses of *base* found in *module*, excluding *base* itself.
    """
    return {
        name
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, base)
        and obj is not base
        and not inspect.isabstract(obj)
        and not name.startswith("_")
    }


@pytest.mark.parametrize(
    ("module_path", "base_path", "label"),
    [
        (
            "shapiq.approximator",
            "shapiq.approximator.base:Approximator",
            "Approximator",
        ),
        (
            "shapiq.explainer",
            "shapiq.explainer.base:Explainer",
            "Explainer",
        ),
        (
            "shapiq.imputer",
            "shapiq.imputer.base:Imputer",
            "Imputer",
        ),
    ],
    ids=["approximator", "explainer", "imputer"],
)
def test_all_concrete_subclasses_in_all(module_path: str, base_path: str, label: str) -> None:
    """Every concrete public subclass must appear in its module's ``__all__``.

    This prevents new classes from silently escaping the public API and
    therefore the auto-generated documentation.

    Args:
        module_path: Dotted import path of the public package (e.g. ``shapiq.approximator``).
        base_path: Colon-separated ``module:ClassName`` for the abstract base class.
        label: Human-readable name used in the assertion message.
    """
    module = importlib.import_module(module_path)
    base_module_path, base_class_name = base_path.split(":")
    base: type = getattr(importlib.import_module(base_module_path), base_class_name)

    concrete = _find_concrete_subclasses(module, base)
    exported = set(module.__all__)
    missing = concrete - exported

    pkg_init = f"src/shapiq/{module_path.split('.')[-1]}/__init__.py"
    assert not missing, (
        f"Concrete {label} subclasses not listed in {module_path}.__all__: "
        f"{missing}. Add them to {pkg_init}."
    )
