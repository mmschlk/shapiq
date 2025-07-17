"""A module containing all deprecated features / behaviors in the shapiq package.

Usage:
    To add a new deprecated feature or behavior, create an instance of the `DeprecatedFeature` and
    append it to the `DEPRECATED_FEATURES` list. The `call` attribute should be a callable that
    triggers the deprecation warning when executed. The `deprecated_in` and `removed_in` attributes
    should specify the version in which the feature was deprecated and the version in which it will
    be removed, respectively.

"""

from __future__ import annotations

import importlib
import pathlib
import pkgutil
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest


class DeprecatedFeature(NamedTuple):
    """A named tuple to represent a deprecated feature."""

    name: str
    deprecated_in: str
    removed_in: str
    call: Callable[[pytest.FixtureRequest], None]


DEPRECATED_FEATURES: list[DeprecatedFeature] = []


def register_deprecated(name: str, deprecated_in: str, removed_in: str):
    def decorator(func: Callable[[pytest.FixtureRequest], None]):
        DEPRECATED_FEATURES.append(DeprecatedFeature(name, deprecated_in, removed_in, func))
        return func

    return decorator


@register_deprecated(
    name="ExampleFeature(NeverRemoveThis)", deprecated_in="1.0.0", removed_in="9.9.9"
)
def example_always_warn(requests: pytest.FixtureRequest):
    """An example feature that always raises a deprecation warning using a fixture.

    This "behavior" is just an example of how to handle deprecations in the codebase. This
    "feature" needs a fixture (in this case an example game) which is passed via the `request`
    fixture. The warning which is raised contains the same version information as the
    `DeprecatedFeature` instance in the `DEPRECATED_FEATURES` list.

    Note:
        Do not delete this feature, as it is a good example of how to handle deprecations.

    """
    from shapiq.utils.errors import raise_deprecation_warning

    raise_deprecation_warning(
        "This is an example of a deprecated feature that always raises a warning.",
        deprecated_in="1.0.0",
        removed_in="9.9.9",
    )
    _ = requests.getfixturevalue("iv_7_all")  # This line grabs the fixture as an example of usage


# auto-import all deprecated modules in the current package
def _auto_import_deprecated_modules():
    current_path = pathlib.Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(current_path)]):
        name = module_info.name
        if name not in {"__init__", "features"}:
            importlib.import_module(f"{__package__}.{name}")


_auto_import_deprecated_modules()
