"""This test module tests base functionality of imputers."""

from __future__ import annotations

import numpy as np
import pytest
from tests.utils import get_concrete_class

from shapiq.games.imputer.base import Imputer


def test_abstract_imputer():
    """Tests if the attributes and properties of imputers are set correctly."""

    def model(x):
        return x

    data = np.asarray([[1, 2, 3], [4, 5, 6]])
    imputer = get_concrete_class(Imputer)(model, data)
    assert imputer.model == model
    assert np.all(imputer.data == data)
    assert imputer.n_features == 3
    assert imputer._cat_features == []
    assert imputer.random_state is None
    assert imputer._rng is not None

    with pytest.raises(NotImplementedError):
        imputer(np.array([[True, False, True]]))
