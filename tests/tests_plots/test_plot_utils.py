"""This module contains tests for the plot.utils module."""

from shapiq.plot.utils import abbreviate_feature_names


def test_abbreviate():
    """Tests the abbreviate_feature_names function."""

    # test with all cases
    feature_names = [
        # seperators
        "feature_A",
        "feature-B",
        "feature.C",
        "feature D",
        # seperators with extra dot at the end should not be included
        "feature E.",
        # capital letters
        "CapitalLetters",
        # normal base case
        "longlowercase",
    ]
    expected = ["FA", "FB", "FC", "FD", "FE", "CL", "lon."]
    assert abbreviate_feature_names(feature_names) == expected
