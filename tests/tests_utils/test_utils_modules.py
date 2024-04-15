"""This test module contains tests for utils.modules."""

import pytest
from sklearn.tree import DecisionTreeRegressor

from shapiq.utils import safe_isinstance


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
