"""Lazy imports for the optional dependencies of :mod:`shapiq_games` and :mod:`shapiq_benchmark`."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def require(package: str, *, purpose: str | None = None, extra: str = "games") -> ModuleType:
    """Import ``package`` or raise an ``ImportError`` that names the missing dependency.

    Args:
        package: The importable name of the package (e.g. ``"openml"``).
        purpose: An optional short description of what needs the package, used in the message.
        extra: The extra of ``shapiq`` that installs the package, ``"games"`` (default) or
            ``"benchmark"``.

    Returns:
        The imported module.

    Raises:
        ImportError: If the package is not installed.
    """
    try:
        return importlib.import_module(package)
    except ImportError as error:
        needed_for = f" for {purpose}" if purpose else ""
        msg = (
            f"'{package}' is required{needed_for} but is not installed. Install it with: "
            f"pip install 'shapiq[{extra}]'"
        )
        raise ImportError(msg) from error
