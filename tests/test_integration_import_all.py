"""This integration test module contains all tests regarding the importability of the package.
(I.e. it checks whether all modules can be imported without errors.)"""

import importlib
import pkgutil
import sys
import pytest

import shapiq
import approximator as approximator
import explainer as explainer
import games as games
import utils as utils
import plot as plot


@pytest.mark.parametrize(
    "package",
    [
        shapiq,
        approximator,
        explainer,
        games,
        utils,
        plot,
    ],
)
def test_import_package(package):
    """Tests whether the package can be imported."""
    assert package.__name__ in sys.modules


@pytest.mark.parametrize(
    "package",
    [
        shapiq,
        approximator,
        explainer,
        games,
        utils,
        plot,
    ],
)
def test_import_submodules(package):
    """Tests whether all submodules of the package can be imported."""
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module_name)


def test_approximator_imports():
    """Tests whether all modules can be imported manually."""

    from shapiq import (
        approximator,
    )

    from shapiq.approximator import (
        PermutationSamplingSII,
        PermutationSamplingSTI,
        RegressionFSI,
        ShapIQ,
    )

    from shapiq import ShapIQ, PermutationSamplingSII, PermutationSamplingSTI, RegressionFSI

    assert True
