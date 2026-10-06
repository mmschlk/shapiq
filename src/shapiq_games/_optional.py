"""Lazy imports for the optional dependencies of :mod:`shapiq_games`."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_INSTALL_HINT = "Install the optional game dependencies with: pip install 'shapiq[games]'"


def require(package: str, *, purpose: str | None = None) -> ModuleType:
    """Import ``package`` or raise an ``ImportError`` that names the missing dependency.

    Args:
        package: The importable name of the package (e.g. ``"openml"``).
        purpose: An optional short description of what needs the package, used in the message.

    Returns:
        The imported module.

    Raises:
        ImportError: If the package is not installed.
    """
    try:
        return importlib.import_module(package)
    except ImportError as error:
        needed_for = f" for {purpose}" if purpose else ""
        msg = f"'{package}' is required{needed_for} but is not installed. {_INSTALL_HINT}"
        raise ImportError(msg) from error
