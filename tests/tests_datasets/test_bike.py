"""This test module contains the tests for the bike dataset."""
from shapiq import load_bike


def test_load_bike():
    data = load_bike()
    # test if data is a pandas dataframe
    assert isinstance(data, type(data))
