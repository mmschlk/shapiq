"""Tests the indices configuration module."""

from __future__ import annotations

from typing import get_args

from shapiq.game_theory.indices import ALL_AVAILABLE_CONCEPTS
from shapiq.typing import IndexType


def test_configuration():
    """Tests if the fields in the configuration are correct."""
    all_indices_checked = set()

    for index, index_info in ALL_AVAILABLE_CONCEPTS.items():
        assert index_info["name"] != ""
        assert index_info["source"] != ""
        assert index_info["generalizes"] != ""
        all_indices_checked.add(index)

    assert all_indices_checked == set(ALL_AVAILABLE_CONCEPTS.keys())


def test_all_concepts_in_index_type():
    """Checks if all indices in ALL_AVAILABLE_CONCEPTS are in IndexType."""
    index_type_args = set(get_args(IndexType))
    all_indices = set(ALL_AVAILABLE_CONCEPTS.keys())

    assert index_type_args == all_indices, (
        f"IndexType does not contain all indices from ALL_AVAILABLE_CONCEPTS. "
        f"Missing indices: {all_indices - index_type_args}."
    )
