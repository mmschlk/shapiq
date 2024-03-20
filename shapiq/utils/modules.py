import importlib
import sys
from types import ModuleType
from typing import Any, Optional, Union


def safe_isinstance(obj: Any, class_path_str: Union[str, list[str], tuple[str]]) -> bool:
    """
    Acts as a safe version of isinstance without having to explicitly import packages which may not
    exist in the user's environment. Checks if obj is an instance of type specified by
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
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = [""]

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError(
                "class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'"
            )

        # Splits on last occurrence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

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


def try_import(name: str, package: Optional[str] = None) -> Optional[ModuleType]:
    """
    Try to import a module and return None if it fails.

    Note:
        Solution adapted from [stack overflow](https://stackoverflow.com/a/53241197).

    Args:
        name: The name of the module to import.
        package: The package to import the module from.

    Returns:
        The imported module or None if the import fails.
    """
    try:
        return importlib.import_module(name, package=package)
    except ImportError:
        return None
