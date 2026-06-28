"""This test module contains the test cases for the utils sets module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import binom

from shapiq.utils import (
    count_interactions,
    generate_interaction_lookup,
    generate_interaction_lookup_from_coalitions,
    get_explicit_subsets,
    log_binom,
    pair_subset_sizes,
    powerset,
    split_subsets_budget,
    transform_array_to_coalitions,
    transform_coalitions_to_array,
)


@pytest.mark.parametrize(
    ("iterable", "min_size", "max_size", "expected"),
    [
        ([1, 2, 3], 0, None, [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]),
        ([1, 2, 3], 1, None, [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]),
        ([1, 2, 3], 0, 2, [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]),
        (
            ["A", "B", "C"],
            0,
            None,
            [(), ("A",), ("B",), ("C",), ("A", "B"), ("A", "C"), ("B", "C"), ("A", "B", "C")],
        ),
    ],
)
def test_powerset(iterable, min_size, max_size, expected):
    """Tests the powerset function."""
    assert list(powerset(iterable, min_size, max_size)) == expected


@pytest.mark.parametrize(
    ("order", "n", "expected"),
    [
        (1, 5, ([(1, 4), (2, 3)], None)),
        (2, 5, ([(2, 3)], None)),
        (3, 5, ([], None)),
        (1, 6, ([(1, 5), (2, 4)], 3)),
        (2, 6, ([(2, 4)], 3)),
        (3, 6, ([], 3)),
    ],
)
def test_pairing(order, n, expected):
    """Tests the get_paired_subsets function."""
    assert pair_subset_sizes(order, n) == expected


@pytest.mark.parametrize(
    ("order", "n", "budget", "q", "expected"),
    [
        (1, 6, 100, [0, 1, 1, 1, 1, 1, 0], ([1, 5, 2, 4, 3], [], 38)),
        (1, 6, 60, [0, 1, 1, 1, 1, 1, 0], ([1, 5, 2, 4], [3], 18)),
        (1, 6, 100, [0, 0, 0, 0, 0, 0, 0], ([], [1, 2, 3, 4, 5], 100)),
    ],
)
def test_split_subsets_budget(budget, order, n, q, expected):
    """Tests the split_subsets_budget function."""
    sampling_weights = np.asarray(q, dtype=float)
    assert split_subsets_budget(order, n, budget, sampling_weights) == expected
    assert (
        split_subsets_budget(order=order, n=n, budget=budget, sampling_weights=sampling_weights)
        == expected
    )


@pytest.mark.parametrize(
    ("n", "subset_sizes", "expected"),
    [
        (3, [1, 2], [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]),
        (3, [1, 2, 3], [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]),
    ],
)
def test_get_explicit_subsets(n, subset_sizes, expected):
    """Tests the get_explicit_subsets function."""

    def check_correctness(subsets, expected):
        assert len(subsets) == len(expected)
        expected_array = np.zeros((len(expected), n), dtype=bool)
        for i, subset in enumerate(expected):
            expected_array[i, subset] = True
        assert np.all(subsets == expected_array)

    explicit_subsets = get_explicit_subsets(n, subset_sizes)  # without parameter names
    check_correctness(explicit_subsets, expected)
    explicit_subsets = get_explicit_subsets(n=n, subset_sizes=subset_sizes)  # with parameter names
    check_correctness(explicit_subsets, expected)


@pytest.mark.parametrize(
    ("n", "min_order", "max_order", "expected"),
    [
        (3, 1, 1, {(0,): 0, (1,): 1, (2,): 2}),
        (3, 2, 2, {(0, 1): 0, (0, 2): 1, (1, 2): 2}),
        (3, 3, 3, {(0, 1, 2): 0}),
        (3, 1, 2, {(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5}),
        (3, 1, 3, {(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5, (0, 1, 2): 6}),
        (["A", "B", "C"], 1, 1, {("A",): 0, ("B",): 1, ("C",): 2}),
        ({1, 5, 8}, 1, 2, {(1,): 0, (5,): 1, (8,): 2, (1, 5): 3, (1, 8): 4, (5, 8): 5}),
    ],
)
def test_generate_interaction_lookup(n, min_order, max_order, expected):
    """Tests the generate_interaction_lookup function."""
    assert generate_interaction_lookup(n, min_order, max_order) == expected


@pytest.mark.parametrize(
    ("coalitions", "expected"),
    [
        (
            np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]]),
            {(0, 2): 0, (1, 2): 1, (0, 1): 2, (2,): 3},
        ),
        (
            np.array([[1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            {(0, 1, 2): 0, (1,): 1, (0,): 2, (2,): 3},
        ),
        (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            {(0,): 0, (1,): 1, (2,): 2},
        ),
        (
            np.array([[1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 0]]),
            {(0, 1, 3): 0, (2, 3): 1, (0, 2): 2},
        ),
        (
            np.array([[0, 0, 0], [1, 1, 1]]),
            {(): 0, (0, 1, 2): 1},
        ),
    ],
)
def test_generate_interaction_lookup_from_coalitions(coalitions, expected):
    """Tests the generate_interaction_lookup_from_coalitions function."""
    result = generate_interaction_lookup_from_coalitions(coalitions)
    assert result == expected


@pytest.mark.parametrize(
    ("coalitions", "n_player", "expected"),
    [
        (
            [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)],
            None,
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]),
        ),
        (
            [(0, 1), (1, 2), (0, 2)],
            3,
            np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
        ),
        (
            [(0, 1), (1, 2), (0, 2)],
            4,
            np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]]),
        ),
    ],
)
def test_transform_coalitions_to_array(coalitions, n_player, expected):
    """Tests the transform_coalitions_to_array function."""
    assert np.all(transform_coalitions_to_array(coalitions, n_player) == expected)


@pytest.mark.parametrize(
    ("coalitions", "expected"),
    [
        (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]),
            [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)],
        ),
        (
            np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
            [(0, 1), (1, 2), (0, 2)],
        ),
        (
            np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]]),
            [(0, 1), (1, 2), (0, 2)],
        ),
        (
            np.array([[False, False, False], [True, True, True]]),
            [(), (0, 1, 2)],
        ),
    ],
)
def test_transform_array_to_coalitions(coalitions, expected):
    """Tests the transform_array_to_coalitions function."""
    assert transform_array_to_coalitions(coalitions) == expected


@pytest.mark.parametrize(
    ("n", "max_order", "min_order", "expected"),
    [
        (3, None, 0, 8),
        (3, 1, 1, 3),
        (3, 2, 1, 6),
        (3, 2, 0, 7),
        (3, 3, 1, 7),
        (3, 3, 2, 4),
    ],
)
def test_count_interactions(n, max_order, min_order, expected):
    """Tests the count_interactions function."""
    count = count_interactions(n, max_order, min_order)
    assert count == expected
    assert isinstance(count, int)


@pytest.mark.parametrize("n", [0, 1, 5, 20])
def test_log_binom_matches_scipy_binom(n):
    """log_binom equals log(scipy.binom) across the full valid range of k."""
    k = np.arange(0, n + 1)
    expected = np.log(binom(n, k))
    np.testing.assert_allclose(log_binom(n, k), expected, rtol=1e-12, atol=1e-12)


def test_log_binom_scalar_returns_float():
    """A scalar k yields a Python float; log(binom(n, 0)) == log(binom(n, n)) == 0."""
    result = log_binom(10, 3)
    assert isinstance(result, float)
    assert result == pytest.approx(float(np.log(binom(10, 3))))
    assert log_binom(10, 0) == pytest.approx(0.0)
    assert log_binom(10, 10) == pytest.approx(0.0)


def test_log_binom_array_returns_array():
    """An array k yields a numpy array of matching shape."""
    result = log_binom(8, np.array([0, 4, 8]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


@pytest.mark.parametrize("k", [-1, -5, 6, 100])
def test_log_binom_out_of_range_is_neg_inf(k):
    """k outside [0, n] means binom(n, k) == 0, i.e. log_binom == -inf."""
    assert log_binom(5, k) == -np.inf


def test_log_binom_out_of_range_array():
    """Out-of-range entries in an array k are individually set to -inf."""
    result = log_binom(5, np.array([-1, 2, 6]))
    assert result[0] == -np.inf
    assert result[2] == -np.inf
    assert np.isfinite(result[1])


def test_log_binom_stays_finite_where_binom_overflows():
    """For large n the central coefficient overflows binom but log_binom stays finite."""
    n = 2000
    assert np.isinf(binom(n, n // 2))  # scipy.binom overflows to inf
    central = log_binom(n, n // 2)
    assert np.isfinite(central)
    # log(binom(n, n/2)) ~= n*log(2) - 0.5*log(pi*n/2) (Stirling); within 1% is plenty.
    approx = n * np.log(2) - 0.5 * np.log(np.pi * n / 2)
    assert central == pytest.approx(approx, rel=1e-2)
