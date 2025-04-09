"""This test module contains the tests for all datasets."""

from __future__ import annotations

import numpy as np
from pandas import DataFrame, Series

from shapiq.datasets import load_adult_census, load_bike_sharing, load_california_housing


def test_load_bike():
    x_data, y_data = load_bike_sharing()
    assert isinstance(x_data, DataFrame)
    assert isinstance(y_data, Series)

    x_data, y_data = load_bike_sharing(to_numpy=True)
    assert isinstance(x_data, np.ndarray)
    assert isinstance(y_data, np.ndarray)


def test_load_adult_census():
    x_data, y_data = load_adult_census()
    assert isinstance(x_data, DataFrame)
    assert isinstance(y_data, Series)

    x_data, y_data = load_adult_census(to_numpy=True)
    assert isinstance(x_data, np.ndarray)
    assert isinstance(y_data, np.ndarray)


def test_load_california_housing():
    x_data, y_data = load_california_housing()
    assert isinstance(x_data, DataFrame)
    assert isinstance(y_data, Series)

    x_data, y_data = load_california_housing(to_numpy=True)
    assert isinstance(x_data, np.ndarray)
    assert isinstance(y_data, np.ndarray)
