"""Optional-dependency handling for shapiq_benchmark.

The benchmark model backends (``optuna``, ``tabpfn``, ``lightgbm``,
``xgboost``) are optional. They are imported lazily through :func:`require`
so that importing ``shapiq_benchmark`` (and its benchmark-type modules) does
not require them; the extras are only needed when the corresponding model is
actually built. Install them via ``pip install 'shapiq[benchmark]'``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_INSTALL_HINT = "Install the benchmark extras with: pip install 'shapiq[benchmark]'"


def require(package: str) -> ModuleType:
    """Import an optional benchmark backend, raising a helpful error if missing.

    Args:
        package: Import name of the optional dependency (e.g. ``"xgboost"``).

    Returns:
        The imported module.

    Raises:
        ImportError: If the package is not installed. The message points to the
            ``benchmark`` extra so users know how to get it.
    """
    try:
        return importlib.import_module(package)
    except ImportError as err:
        msg = (
            f"'{package}' is required for this shapiq_benchmark feature but is not "
            f"installed. {_INSTALL_HINT}"
        )
        raise ImportError(msg) from err
