"""This module contains tests for deprecations."""

from __future__ import annotations

import inspect
import re
import warnings
from importlib.metadata import version

import pytest
from packaging.version import parse

from .features import DEPRECATED_FEATURES, DeprecatedFeature


def feature_should_be_removed(feature: DeprecatedFeature) -> None:
    """Fails if the feature is still accessible after its scheduled removal date."""
    source_file = inspect.getsourcefile(feature.call)
    source_line = inspect.getsourcelines(feature.call)[1]
    if parse(version("shapiq")) >= parse(feature.removed_in):
        pytest.fail(
            f"{feature.name} was scheduled for removal in {feature.removed_in} "
            f"but is still accessible. Remove the deprecated behavior and this test.\n"
            f"Feature registered at: {source_file}:{source_line}"
        )


def feature_raises_deprecation_warning(
    feature: DeprecatedFeature, request: pytest.FixtureRequest
) -> None:
    """Fails if the feature does not raise a deprecation warning."""
    expected_msg = (
        f"deprecated in version {feature.deprecated_in} and will be removed in version "
        f"{feature.removed_in}."
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        feature.call(request)

    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]

    # check if any deprecation warning matches the expected regex
    if not any(re.search(expected_msg, str(w.message)) for w in deprecation_warnings):
        formatted = "\n".join(f"{w.category.__name__}: {w.message}" for w in deprecation_warnings)
        source_file = inspect.getsourcefile(feature.call)
        source_line = inspect.getsourcelines(feature.call)[1]
        pytest.fail(
            f"No matching DeprecationWarning for feature '{feature.name}'.\n"
            f"Expected regex: {expected_msg}\n"
            f"Warnings captured: {formatted or 'None'}\n"
            f"Feature registered at: {source_file}:{source_line}\n"
        )


@pytest.mark.parametrize("feature", DEPRECATED_FEATURES, ids=lambda f: f.name)
def test_deprecated_features(feature: DeprecatedFeature, request: pytest.FixtureRequest):
    """Tests the deprecated initialization with path_to_values."""
    # check if the feature should already be removed
    feature_should_be_removed(feature)

    # check if the feature raises a correct deprecation warning
    feature_raises_deprecation_warning(feature, request)
