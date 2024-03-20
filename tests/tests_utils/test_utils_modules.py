"""This test module contains tests for utils.modules."""
import pytest

from shapiq.utils import safe_isinstance, try_import
from sklearn.tree import DecisionTreeRegressor


def test_safe_isinstance():
    """Test the safe_isinstance function."""
    model = DecisionTreeRegressor()

    assert safe_isinstance(model, "sklearn.tree.DecisionTreeRegressor")
    assert safe_isinstance(
        model, ["sklearn.tree.DecisionTreeClassifier", "sklearn.tree.DecisionTreeRegressor"]
    )
    assert safe_isinstance(model, ("sklearn.tree.DecisionTreeRegressor",))
    with pytest.raises(ValueError):
        safe_isinstance(model, "DecisionTreeRegressor")
    with pytest.raises(ValueError):
        safe_isinstance(model, None)
    assert not safe_isinstance(model, "my.made.up.module")
    assert not safe_isinstance(model, ["sklearn.ensemble.DecisionTreeRegressor"])


def test_try_import():
    """Tests the try_import function."""

    # Test with a package that exists
    try_import("DecisionTreeClassifier", package="sklearn.tree")
    # check if the module is imported in the current environment namespace

    # Test with a package that does not exist
    try_import("DecisionTreeClassifier", package="my.made.up.module")
