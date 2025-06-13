"""This test module contains tests for utils.modules."""

from __future__ import annotations

import pytest
from sklearn.tree import DecisionTreeRegressor

from shapiq.utils import check_import_module, safe_isinstance


def test_safe_isinstance():
    """Test the safe_isinstance function."""
    model = DecisionTreeRegressor()

    assert safe_isinstance(model, "sklearn.tree.DecisionTreeRegressor")
    assert safe_isinstance(
        model,
        ["sklearn.tree.DecisionTreeClassifier", "sklearn.tree.DecisionTreeRegressor"],
    )
    assert safe_isinstance(model, ("sklearn.tree.DecisionTreeRegressor",))
    with pytest.raises(ValueError):
        safe_isinstance(model, "DecisionTreeRegressor")
    with pytest.raises(ValueError):
        safe_isinstance(model, None)
    assert not safe_isinstance(model, "my.made.up.module")
    assert not safe_isinstance(model, ["sklearn.ensemble.DecisionTreeRegressor"])


def test_check_import_module():
    """Test check_import_module function."""
    check_import_module("sklearn")
    with pytest.raises(ImportError):
        check_import_module("my.made.up.module", functionality="something")
