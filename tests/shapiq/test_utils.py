"""Unit tests for ``shapiq.utils`` helpers (sets, modules, datasets, errors)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from shapiq.utils import (
    check_import_module,
    count_interactions,
    generate_interaction_lookup,
    generate_interaction_lookup_from_coalitions,
    get_explicit_subsets,
    pair_subset_sizes,
    powerset,
    raise_deprecation_warning,
    safe_isinstance,
    shuffle_data,
    split_subsets_budget,
    transform_array_to_coalitions,
    transform_coalitions_to_array,
)

# ===================================================================
# sets.py
# ===================================================================


class TestPowerset:
    def test_default_includes_empty_and_full(self):
        ps = list(powerset([1, 2, 3]))
        assert ps[0] == ()
        assert (1, 2, 3) in ps
        assert len(ps) == 2**3

    def test_min_size_excludes_smaller_subsets(self):
        ps = list(powerset([1, 2, 3], min_size=1))
        assert () not in ps
        assert all(len(s) >= 1 for s in ps)

    def test_max_size_limits_upper_bound(self):
        ps = list(powerset([1, 2, 3], max_size=2))
        assert all(len(s) <= 2 for s in ps)
        assert (1, 2, 3) not in ps

    def test_max_size_exceeds_length(self):
        """max_size > len(iterable) should be clamped, not explode."""
        ps = list(powerset([1, 2], max_size=10))
        assert len(ps) == 2**2


class TestPairSubsetSizes:
    def test_even_n_order_1(self):
        paired, unpaired = pair_subset_sizes(order=1, n=5)
        assert paired == [(1, 4), (2, 3)]
        assert unpaired is None

    def test_odd_n_has_unpaired(self):
        paired, unpaired = pair_subset_sizes(order=1, n=6)
        assert paired == [(1, 5), (2, 4)]
        assert unpaired == 3


class TestSplitSubsetsBudget:
    def test_full_budget_computes_all(self):
        complete, incomplete, remaining = split_subsets_budget(
            order=1, n=6, budget=100, sampling_weights=np.ones(6)
        )
        assert set(complete) == {1, 2, 3, 4, 5}
        assert incomplete == []
        assert remaining >= 0

    def test_zero_weights_nothing_explicit(self):
        complete, incomplete, remaining = split_subsets_budget(
            order=1, n=6, budget=100, sampling_weights=np.zeros(6)
        )
        assert complete == []
        assert incomplete == [1, 2, 3, 4, 5]
        assert remaining == 100


class TestGetExplicitSubsets:
    def test_shape_and_counts(self):
        subsets = get_explicit_subsets(n=4, subset_sizes=[1, 2])
        assert subsets.dtype == bool
        # C(4,1) + C(4,2) = 4 + 6 = 10 rows, 4 columns
        assert subsets.shape == (10, 4)
        # each of the 4 singleton rows has exactly 1 True
        assert (subsets[:4].sum(axis=1) == 1).all()
        # each of the 6 pair rows has exactly 2 True
        assert (subsets[4:].sum(axis=1) == 2).all()


class TestInteractionLookup:
    def test_from_integer_players(self):
        lookup = generate_interaction_lookup(3, 1, 2)
        assert lookup == {(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5}

    def test_from_named_players(self):
        lookup = generate_interaction_lookup(["A", "B", "C"], 1, 1)
        assert lookup == {("A",): 0, ("B",): 1, ("C",): 2}

    def test_from_coalitions(self):
        coalitions = np.array(
            [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]],
            dtype=bool,
        )
        lookup = generate_interaction_lookup_from_coalitions(coalitions)
        assert lookup == {(0, 2): 0, (1, 2): 1, (0, 1): 2, (2,): 3}


class TestCoalitionTransforms:
    def test_tuples_to_array_inferred_n_players(self):
        arr = transform_coalitions_to_array([(0, 1), (1, 2), (0, 2)])
        assert arr.shape == (3, 3)
        assert arr.dtype == bool

    def test_tuples_to_array_with_padding(self):
        arr = transform_coalitions_to_array([(0, 1)], n_players=4)
        assert arr.shape == (1, 4)
        assert arr[0].tolist() == [True, True, False, False]

    def test_roundtrip(self):
        original = [(0, 2), (1,), (0, 1, 2)]
        arr = transform_coalitions_to_array(original, n_players=3)
        recovered = list(transform_array_to_coalitions(arr))
        assert recovered == original


class TestCountInteractions:
    def test_full_powerset(self):
        # powerset of 4 players = 2^4 including the empty set
        assert count_interactions(n=4) == 2**4

    def test_bounded_orders(self):
        # orders 1..2 on 4 players: C(4,1) + C(4,2) = 4 + 6 = 10
        assert count_interactions(n=4, max_order=2, min_order=1) == 10


# ===================================================================
# modules.py
# ===================================================================


class TestModules:
    def test_safe_isinstance_true(self):
        # numpy is always imported in the test environment
        arr = np.array([1, 2, 3])
        assert safe_isinstance(arr, "numpy.ndarray") is True

    def test_safe_isinstance_false_for_other_type(self):
        assert safe_isinstance("not an array", "numpy.ndarray") is False

    def test_safe_isinstance_unimported_module_returns_false(self):
        # A fully-qualified path from a module that's almost certainly not imported.
        assert safe_isinstance(object(), "zzz_definitely_not_a_module.SomeClass") is False

    def test_safe_isinstance_invalid_path_raises(self):
        with pytest.raises(ValueError):
            safe_isinstance(object(), "not_a_dotted_path")

    def test_check_import_module_present(self):
        # numpy is present — should not raise
        check_import_module("numpy")

    def test_check_import_module_missing_raises(self):
        with pytest.raises(ImportError, match="Missing optional dependency"):
            check_import_module("zzz_definitely_not_a_module")


# ===================================================================
# datasets.py
# ===================================================================


class TestShuffleData:
    def test_shuffle_preserves_pairing(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))
        y = np.arange(20)
        X_s, y_s = shuffle_data(X, y, random_state=42)

        # same shapes
        assert X_s.shape == X.shape
        assert y_s.shape == y.shape
        # each row still corresponds to its label
        for i, label in enumerate(y_s):
            assert np.array_equal(X_s[i], X[label])

    def test_shuffle_reproducible(self):
        X = np.arange(30).reshape(10, 3)
        y = np.arange(10)
        a_X, a_y = shuffle_data(X.copy(), y.copy(), random_state=123)
        b_X, b_y = shuffle_data(X.copy(), y.copy(), random_state=123)
        assert np.array_equal(a_X, b_X)
        assert np.array_equal(a_y, b_y)


# ===================================================================
# errors.py
# ===================================================================


class TestRaiseDeprecationWarning:
    def test_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raise_deprecation_warning("foo is deprecated", deprecated_in="1.0", removed_in="2.0")

        assert len(caught) == 1
        assert issubclass(caught[0].category, DeprecationWarning)
        assert "1.0" in str(caught[0].message)
        assert "2.0" in str(caught[0].message)
