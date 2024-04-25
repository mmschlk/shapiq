"""Tests the indices configuration module."""

from shapiq.indices import ALL_AVAILABLE_CONCEPTS


def test_configuration():
    """Tests if the fields in the configuration are correct."""

    all_indices_checked = set()

    for index, index_info in ALL_AVAILABLE_CONCEPTS.items():
        assert index_info["name"] != ""
        assert index_info["source"] != ""
        assert index_info["generalizes"] != ""
        all_indices_checked.add(index)

    assert all_indices_checked == set(ALL_AVAILABLE_CONCEPTS.keys())
