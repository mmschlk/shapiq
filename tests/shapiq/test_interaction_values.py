"""Tests for the InteractionValues data structure."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues, aggregate_interaction_values
from shapiq.utils import powerset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_iv(
    n_players: int = 5,
    max_order: int = 2,
    min_order: int = 1,
    index: str = "k-SII",
    baseline: float = 1.0,
) -> InteractionValues:
    """Create a deterministic InteractionValues for testing."""
    interaction_lookup = {}
    values = []
    for i, interaction in enumerate(
        powerset(range(n_players), min_size=min_order, max_size=max_order)
    ):
        interaction_lookup[interaction] = i
        values.append(float(i) * 0.1)
    return InteractionValues(
        values=np.array(values),
        index=index,
        n_players=n_players,
        min_order=min_order,
        max_order=max_order,
        interaction_lookup=interaction_lookup,
        baseline_value=baseline,
        estimated=True,
        estimation_budget=100,
    )


# ===================================================================
# Creation & basic access
# ===================================================================


class TestCreation:
    def test_basic_properties(self):
        iv = _make_iv()
        assert iv.n_players == 5
        assert iv.max_order == 2
        assert iv.min_order == 1
        assert iv.index == "k-SII"
        assert iv.baseline_value == 1.0
        assert iv.estimated is True
        assert iv.estimation_budget == 100

    def test_getitem_single_interaction(self):
        iv = _make_iv()
        val = iv[(0,)]
        assert isinstance(val, float)

    def test_getitem_missing_returns_zero(self):
        iv = _make_iv(max_order=1)
        assert iv[(0, 1)] == 0.0

    def test_empty_interaction_is_baseline(self):
        iv = _make_iv(min_order=0)
        assert iv[()] == pytest.approx(iv.baseline_value)

    @pytest.mark.parametrize(
        ("index", "should_warn"),
        [("k-SII", False), ("SII", False), ("NOT_VALID", True)],
    )
    def test_invalid_index_warns(self, index, should_warn):
        if should_warn:
            with pytest.warns(UserWarning):
                _make_iv(index=index)
        else:
            _make_iv(index=index)  # should not warn


# ===================================================================
# Order extraction
# ===================================================================


class TestOrderExtraction:
    def test_get_n_order_values_shape(self):
        iv = _make_iv(n_players=5, max_order=2, min_order=1)
        order_1 = iv.get_n_order_values(1)
        assert order_1.shape == (5,)

    def test_get_n_order(self):
        iv = _make_iv(n_players=5, max_order=2, min_order=1)
        iv_order1 = iv.get_n_order(order=1)
        assert iv_order1.max_order == 1
        assert iv_order1.min_order == 1


# ===================================================================
# Serialization
# ===================================================================


class TestSerialization:
    def test_json_roundtrip(self, tmp_path):
        iv = _make_iv()
        path = tmp_path / "test_iv.json"
        iv.to_json_file(path)
        loaded = InteractionValues.from_json_file(path)

        assert loaded.n_players == iv.n_players
        assert loaded.index == iv.index
        assert np.allclose(loaded.values, iv.values)
        assert loaded.baseline_value == pytest.approx(iv.baseline_value)


# ===================================================================
# Aggregation
# ===================================================================


class TestAggregation:
    def test_aggregate_mean(self):
        iv1 = _make_iv()
        iv2_values = iv1.values.copy()
        iv2_values[:] = 1.0
        iv2 = InteractionValues(
            values=iv2_values,
            index=iv1.index,
            n_players=iv1.n_players,
            min_order=iv1.min_order,
            max_order=iv1.max_order,
            interaction_lookup=dict(iv1.interaction_lookup),
            baseline_value=iv1.baseline_value,
        )
        result = aggregate_interaction_values([iv1, iv2])
        assert isinstance(result, InteractionValues)
        assert result.n_players == iv1.n_players


# ===================================================================
# Copy behavior
# ===================================================================


class TestCopy:
    def test_deepcopy_independent(self):
        from copy import deepcopy

        iv = _make_iv()
        iv_copy = deepcopy(iv)
        iv_copy.values[0] = 999.0
        assert iv.values[0] != 999.0
