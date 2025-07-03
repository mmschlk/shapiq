"""This test module tests all plotting utilities."""

from __future__ import annotations

from shapiq.plot.utils import abbreviate_feature_names, format_labels, format_value


def test_format_value():
    """Test the format_value function."""
    assert format_value(1.0) == "1"
    assert format_value(1.234) == "1.23"
    assert format_value(-1.234) == "\u22121.23"
    assert format_value("1.234") == "1.234"


def test_format_labels():
    """Test the format_labels function."""
    feature_mapping = {0: "A", 1: "B", 2: "C"}
    assert format_labels(feature_mapping, (0, 1)) == "A x B"
    assert format_labels(feature_mapping, (0,)) == "A"
    assert format_labels(feature_mapping, ()) == "Base Value"
    assert format_labels(feature_mapping, (0, 1, 2)) == "A x B x C"


def test_abbreviate_feature_names():
    """Tests the abbreviate_feature_names function."""
    # check for splitting characters
    feature_names = ["feature-0", "feature_1", "feature 2", "feature.3"]
    assert abbreviate_feature_names(feature_names) == ["F0", "F1", "F2", "F3"]

    # check for long names
    feature_names = ["longfeaturenamethatisnotshort", "stilllong"]
    assert abbreviate_feature_names(feature_names) == ["lon.", "sti."]

    # check for abbreviation with capital letters
    feature_names = ["LongFeatureName", "Short"]
    assert abbreviate_feature_names(feature_names) == ["LFN", "Sho."]
