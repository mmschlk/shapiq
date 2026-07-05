"""Lazy imports for shapiq_benchmark's optional model backends."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_INSTALL_HINT = "Install the benchmark extras with: pip install 'shapiq[benchmark]'"


def require(package: str) -> ModuleType:
    """Import ``package``, raising a helpful error pointing at the extra if it is missing."""
    try:
        return importlib.import_module(package)
    except ImportError as err:
        msg = (
            f"'{package}' is required for this shapiq_benchmark feature but is not "
            f"installed. {_INSTALL_HINT}"
        )
        raise ImportError(msg) from err
