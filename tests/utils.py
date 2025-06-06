"""This module contains utility functions for testing purposes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from shapiq.game_theory import index_generalizes_bv, index_generalizes_sv

if TYPE_CHECKING:
    from shapiq.utils.custom_types import IndexType


def get_concrete_class(abclass):
    """Class decorator to create a concrete class from an abstract class.

    The function is used to test abstract classes and their methods.
    Directly taken from https://stackoverflow.com/a/37574495.

    Args:
        abclass: The abstract class to create a concrete class from.

    Returns:
        The concrete class.

    """

    class concreteCls(abclass):
        pass

    concreteCls.__abstractmethods__ = frozenset()
    return type("DummyConcrete" + abclass.__name__, (concreteCls,), {})


def get_expected_index_or_skip(index: IndexType, order: int) -> IndexType:
    """Get the expected index based on the order and index."""
    expected_index = index
    if order == 1:
        expected_index = "BV" if index_generalizes_bv(index) else expected_index
        expected_index = "SV" if index_generalizes_sv(index) else expected_index

    # skip tests for indices that are not possible
    if expected_index in ["BV", "SV"] and order > 1:
        pytest.skip("Skipping test for BV and SV indices with order > 1.")

    return expected_index
