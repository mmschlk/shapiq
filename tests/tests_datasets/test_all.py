"""This test module contains the tests for all datasets."""

from pandas import DataFrame

from shapiq.datasets import load_bike


def test_load_bike():
    data = load_bike()
    # test if data is a pandas dataframe
    assert isinstance(data, DataFrame)
