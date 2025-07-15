"""Utility functions for checking module imports and class instances."""

from __future__ import annotations

import sys
from importlib import import_module


def safe_isinstance(
    obj: object,
    class_path_str: str | list[str] | tuple[str],
) -> bool:
    """Safely checks if an object is an instance of a class.

    Acts as a safe version of isinstance without having to explicitly import packages which may
    not exist in the user's environment. Checks if obj is an instance of type specified by
    class_path_str.

    Note:
        This function was directly taken from the `shap` repository.

    Args:
        obj: Some object you want to test against
        class_path_str: A string or list of strings specifying full class paths Example:
            `sklearn.ensemble.RandomForestRegressor`

    Returns:
            True if isinstance is true and the package exists, False otherwise

    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list | tuple):
        class_path_strs = list(class_path_str)
    else:
        class_path_strs = [""]

    # try each module path in order
    for _class_path_str in class_path_strs:
        if "." not in _class_path_str:
            msg = "class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'"
            raise ValueError(msg)

        # Splits on last occurrence of "."
        module_name, class_name = _class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        # Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False


def check_import_module(name: str, functionality: str | None = None) -> None:
    """Check if the optional dependency is available."""
    try:
        import_module(name)
    except ImportError as error:
        message = f"Missing optional dependency '{name}'. Install '{name}'"
        if functionality:
            message += f" for {functionality}"
        message += f". Use pip or conda to install '{name}'."
        raise ImportError(message) from error
