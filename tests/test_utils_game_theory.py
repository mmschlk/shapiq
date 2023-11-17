"""This test module contains the test cases for the utils game theory module."""

import pytest

from utils.game_theory import powerset


@pytest.mark.parametrize(
    "iterable, min_size, max_size, expected",
    [
        ([1, 2, 3], 0, None, [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]),
        ([1, 2, 3], 1, None, [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]),
        ([1, 2, 3], 0, 2, [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]),
    ],
)
def test_powerset(iterable, min_size, max_size, expected):
    """Tests the powerset function."""
    assert list(powerset(iterable, min_size, max_size)) == expected
