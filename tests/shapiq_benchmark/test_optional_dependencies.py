"""Tests for shapiq_benchmark's optional-dependency wiring.

The benchmark model backends (``optuna``, ``tabpfn``, ``lightgbm``, ``xgboost``)
are optional and imported lazily so that a plain ``pip install shapiq`` still
ships and imports ``shapiq_benchmark``. These tests pin that behavior down:
the package imports without the backends, missing backends raise a helpful
error pointing at the ``benchmark`` extra, and the always-available
scikit-learn models keep working.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

import shapiq_benchmark
from shapiq_benchmark._optional import require
from shapiq_benchmark.setup import (
    _OPTIONAL_MODEL_BACKENDS,
    _SUPPORTED_MODEL_TASKS,
    _resolve_model_builder,
)

OPTIONAL_MODELS = ["xgboost", "lightgbm", "tabpfn"]


def test_package_imports() -> None:
    """The package itself imports even without the optional backends installed."""
    assert shapiq_benchmark.__name__ == "shapiq_benchmark"


def test_require_returns_module_when_present() -> None:
    """``require`` returns the imported module for an installed package."""
    assert require("numpy") is np


def test_require_raises_helpful_error_when_missing() -> None:
    """``require`` raises an ImportError that points at the benchmark extra."""
    with pytest.raises(ImportError, match=r"shapiq\[benchmark\]"):
        require("a_module_that_is_definitely_not_installed_xyz")


def test_resolve_sklearn_builder_needs_no_optional_backend() -> None:
    """scikit-learn models resolve directly, without importing an optional backend."""
    from sklearn.tree import DecisionTreeClassifier

    assert _resolve_model_builder("decision_tree", "classification") is DecisionTreeClassifier


@pytest.mark.parametrize("model", OPTIONAL_MODELS)
def test_missing_optional_backend_raises_helpful_error(monkeypatch, model) -> None:
    """Resolving an optional model with its backend absent errors clearly.

    Setting ``sys.modules[name] = None`` makes ``import name`` raise, faithfully
    simulating a bare ``pip install shapiq`` even when the backend happens to be
    installed in the test environment.
    """
    package = _OPTIONAL_MODEL_BACKENDS[model][0]
    monkeypatch.setitem(sys.modules, package, None)
    with pytest.raises(ImportError, match=r"shapiq\[benchmark\]"):
        _resolve_model_builder(model, "classification")


@pytest.mark.parametrize("model", OPTIONAL_MODELS)
def test_optional_backend_resolves_when_present(model) -> None:
    """When the backend is installed, resolving returns a usable (callable) builder."""
    package = _OPTIONAL_MODEL_BACKENDS[model][0]
    pytest.importorskip(package)
    for task in ("classification", "regression"):
        assert callable(_resolve_model_builder(model, task))


def test_supported_model_set_is_backend_independent() -> None:
    """All model names are advertised regardless of which backends are installed."""
    names = {name for name, _task in _SUPPORTED_MODEL_TASKS}
    assert {"decision_tree", "random_forest", "mlp", *OPTIONAL_MODELS} <= names


def test_import_warns_when_a_backend_is_missing(monkeypatch) -> None:
    """Importing the package emits an ImportWarning listing the missing backends."""
    # make xgboost look absent, then re-run __init__ via reload to catch the warning
    monkeypatch.setitem(sys.modules, "xgboost", None)
    with pytest.warns(ImportWarning, match=r"shapiq\[benchmark\]"):
        importlib.reload(shapiq_benchmark)
    # reload again with the backend restored so test ordering is unaffected
    monkeypatch.undo()
    importlib.reload(shapiq_benchmark)
