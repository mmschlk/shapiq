"""Tests for the io utils module."""

from __future__ import annotations

import datetime
from importlib.metadata import version

import pytest

from shapiq import KernelSHAP as ApproximatorForTest
from shapiq.utils.saving import (
    dict_to_lookup_and_values,
    lookup_and_values_to_dict,
    make_file_metadata,
    safe_str_to_tuple,
    safe_tuple_to_str,
)


class TestInteractionConversion:
    """Tests converting interaction lookup and values to a dictionary and back."""

    def test_interactions_to_dict(self, iv_7_all):
        """Test converting interaction lookup and values to a dictionary."""
        interaction_lookup, interaction_values = iv_7_all.interaction_lookup, iv_7_all.values
        interaction_dict = lookup_and_values_to_dict(interaction_lookup, interaction_values)

        # Check if the dictionary has the correct keys and values
        for tup in interaction_lookup:
            assert safe_tuple_to_str(tup) in interaction_dict
            assert (
                interaction_dict[safe_tuple_to_str(tup)]
                == interaction_values[interaction_lookup[tup]]
            )

    def test_dict_to_interactions(self, iv_7_all):
        """Test converting a dictionary of interaction values back to lookup and values."""
        interaction_lookup, interaction_values = iv_7_all.interaction_lookup, iv_7_all.values
        interaction_dict = lookup_and_values_to_dict(interaction_lookup, interaction_values)

        new_lookup, new_values = dict_to_lookup_and_values(interaction_dict)

        # Check if the new lookup matches the original
        assert new_lookup == interaction_lookup

        # Check if the new values match the original
        for interaction in interaction_lookup:
            assert (
                new_values[interaction_lookup[interaction]]
                == interaction_values[interaction_lookup[interaction]]
            )


class TestTupleConversion:
    """Tests for tuple conversion functions."""

    data_to_test = (
        ((1, 2, 3), "1,2,3"),
        ((1, 2, 3, 4), "1,2,3,4"),
        ((10, 20), "10,20"),
        ((0, 0, 0), "0,0,0"),  # testing with same integers
        ((3, 2, 1), "3,2,1"),  # tuples are not sorted
        ((-1, -2, -3), "-1,-2,-3"),  # negative integers work (not important, but good to test)
        ((100,), "100"),  # single element tuple
        ((), "Empty"),  # Empty tuple case
    )

    @pytest.mark.parametrize("tup, expected_str", data_to_test)
    def test_safe_tuple_to_str(self, tup, expected_str):
        """Test converting a tuple of integers to a string."""
        assert safe_tuple_to_str(tup) == expected_str

    @pytest.mark.parametrize("tup, string_data", data_to_test)
    def test_safe_str_to_tuple(self, tup, string_data):
        """Test converting a string representation of integers back to a tuple."""
        assert safe_str_to_tuple(string_data) == tup


class TestMetadataBlock:
    """Tests for metadata block creation."""

    def test_meta_data_defaults(self, iv_7_all):
        """Tests creating a metadata block with default values."""
        metadata = make_file_metadata(iv_7_all)
        assert metadata["version"] == version("shapiq")
        assert metadata["data_type"] is None
        assert metadata["created_from"] is None
        assert metadata["description"] is None
        assert metadata["parameters"] == {}

    def test_meta_data_time(self, iv_7_all):
        """Tests if the timestamp is created correctly."""
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat() + "Z"
        metadata = make_file_metadata(iv_7_all)
        now_metadata = metadata["timestamp"]
        assert now_metadata.startswith(now[:20])  # check if timestamp is almost equal to now

    @pytest.mark.parametrize(
        "object_fixture, expected_name",
        [
            ("iv_7_all", "InteractionValues"),
            ("cooking_game_pre_computed", "CookingGame"),
        ],
    )
    def test_make_metadata_block(self, object_fixture, expected_name, request):
        """Test creating a metadata block."""

        object_to_store = request.getfixturevalue(object_fixture)

        desc = "This is a test description."
        parameters = {"param1": "value1", "param2": 42}

        metadata = make_file_metadata(
            object_to_store,
            desc=desc,
            parameters=parameters,
            data_type="game",
            created_from=ApproximatorForTest,
        )

        # check version is correct
        assert isinstance(metadata["version"], str)  # Version should be a string
        assert metadata["version"] == version("shapiq")

        # check if rest is correct
        object_name = metadata["object_name"]
        assert object_name == expected_name
        assert metadata["data_type"] == "game"
        assert metadata["description"] == desc
        assert metadata["created_from"] == repr(ApproximatorForTest)
        assert metadata["parameters"] == parameters
