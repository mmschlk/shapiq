"""This test module contains the tests for all datasets."""

from pandas import DataFrame, Series

from shapiq.datasets import load_bike_sharing, load_adult_census, load_california_housing


def test_load_bike():
    x_data, y_data = load_bike_sharing()
    # test if data is a pandas dataframe and series
    assert isinstance(x_data, DataFrame)
    assert isinstance(y_data, Series)


def test_load_adult_census():
    x_data, y_data = load_adult_census()
    # test if data is a pandas dataframe and series
    assert isinstance(x_data, DataFrame)
    assert isinstance(y_data, Series)


def test_load_california_housing():
    x_data, y_data = load_california_housing()
    # test if data is a pandas dataframe and series
    assert isinstance(x_data, DataFrame)
    assert isinstance(y_data, Series)
