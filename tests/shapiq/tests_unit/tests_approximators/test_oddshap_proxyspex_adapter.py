"""Unit tests for the ProxySPEX adapter used by OddSHAP."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pytest

from shapiq.approximator.regression._oddshap_proxyspex_adapter import (
    _tree_to_fourier,
    lgboost_to_fourier,
    top_k_interactions,
)


# -----------------------------------------------------------------------------
# top_k_interactions — pure-Python selection logic
# -----------------------------------------------------------------------------


def test_top_k_returns_odd_only_by_default():
    coeffs = {
        (): 10.0,        # even (cardinality 0)
        (0,): 5.0,       # odd (cardinality 1)
        (1,): 3.0,       # odd
        (0, 1): 99.0,    # even (cardinality 2) — should be excluded
        (0, 1, 2): 1.0,  # odd
    }
    result = top_k_interactions(coeffs, k=10, odd=True)
    assert (0, 1) not in result
    assert () not in result
    assert set(result.keys()) == {(0,), (1,), (0, 1, 2)}


def test_top_k_keeps_even_when_odd_false():
    coeffs = {(): 10.0, (0,): 5.0, (0, 1): 99.0}
    result = top_k_interactions(coeffs, k=10, odd=False)
    assert set(result.keys()) == {(), (0,), (0, 1)}


def test_top_k_sorts_by_absolute_magnitude():
    coeffs = {
        (0,): 0.1,
        (1,): -10.0,
        (2,): 5.0,
        (3,): -2.0,
    }
    result = top_k_interactions(coeffs, k=2, odd=True)
    # Top 2 by |coefficient|: |-10| and |5|
    assert set(result.keys()) == {(1,), (2,)}


def test_top_k_respects_limit():
    coeffs = {(i,): float(i) for i in range(20)}
    result = top_k_interactions(coeffs, k=3, odd=True)
    assert len(result) == 3


def test_top_k_returns_all_when_fewer_than_k():
    coeffs = {(0,): 1.0, (1,): 2.0}
    result = top_k_interactions(coeffs, k=10, odd=True)
    assert len(result) == 2


def test_top_k_empty_input():
    assert top_k_interactions({}, k=5, odd=True) == {}


# -----------------------------------------------------------------------------
# _tree_to_fourier — DFS recursion on hand-crafted trees
# -----------------------------------------------------------------------------


def test_tree_to_fourier_single_leaf():
    """A tree consisting of a single leaf is a constant function."""
    tree_info = {"tree_structure": {"leaf_value": 7.5}}
    result = _tree_to_fourier(tree_info)
    assert result == {(): 7.5}


def test_tree_to_fourier_one_split():
    """One split on feature 0 with leaves 4.0 and 2.0.

    Fourier coefficients (per the (l + r) / 2, (l - r) / 2 recursion):
        ()    -> (4 + 2) / 2 = 3.0
        (0,)  -> (4 - 2) / 2 = 1.0
    """
    tree_info = {
        "tree_structure": {
            "split_feature": 0,
            "left_child": {"leaf_value": 4.0},
            "right_child": {"leaf_value": 2.0},
        }
    }
    result = _tree_to_fourier(tree_info)
    assert result == {(): 3.0, (0,): 1.0}


def test_tree_to_fourier_two_level_split():
    """A 2-level tree splitting on feature 0 then feature 1.

    Recursion produces, for each split node:
        combined[T]                 = (left_val + right_val) / 2
        combined[T ∪ {feature_idx}] = (left_val - right_val) / 2

    With four leaves (LL=8, LR=2, RL=4, RR=2) and splits 0 then 1:

      level-1 (left subtree, split_feature=1):
        ()  = (8 + 2) / 2 = 5      (1,) = (8 - 2) / 2 = 3
      level-1 (right subtree, split_feature=1):
        ()  = (4 + 2) / 2 = 3      (1,) = (4 - 2) / 2 = 1

      root (split_feature=0):
        ()    = (5 + 3) / 2 = 4    (0,)   = (5 - 3) / 2 = 1
        (1,)  = (3 + 1) / 2 = 2    (0, 1) = (3 - 1) / 2 = 1
    """
    tree_info = {
        "tree_structure": {
            "split_feature": 0,
            "left_child": {
                "split_feature": 1,
                "left_child": {"leaf_value": 8.0},
                "right_child": {"leaf_value": 2.0},
            },
            "right_child": {
                "split_feature": 1,
                "left_child": {"leaf_value": 4.0},
                "right_child": {"leaf_value": 2.0},
            },
        }
    }
    result = _tree_to_fourier(tree_info)
    assert result[()] == pytest.approx(4.0)
    assert result[(0,)] == pytest.approx(1.0)
    assert result[(1,)] == pytest.approx(2.0)
    assert result[(0, 1)] == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# lgboost_to_fourier — end-to-end on a fitted LightGBM model
# -----------------------------------------------------------------------------


def _fit_lgbm_on_function(value_fn, n_features=4, n_samples=512, seed=42):
    """Fit a LightGBM regressor on (binary coalitions, value_fn(coalitions))."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_samples, n_features)).astype(bool)
    y = np.array([value_fn(row) for row in X], dtype=float)
    model = lgb.LGBMRegressor(
        n_estimators=20, max_depth=4, num_leaves=8, learning_rate=0.5,
        verbose=-1, n_jobs=1, random_state=seed,
    )
    model.fit(X, y)
    return model


def test_lgboost_to_fourier_constant_target():
    """Constant target → only () (empty interaction) should have non-zero coefficient."""
    model = _fit_lgbm_on_function(lambda x: 3.14)
    fourier = lgboost_to_fourier(model.booster_.dump_model())
    # Only (or mostly) the empty interaction should carry weight.
    assert () in fourier
    non_baseline = {k: v for k, v in fourier.items() if k != ()}
    if non_baseline:
        max_non_baseline = max(abs(v) for v in non_baseline.values())
        assert max_non_baseline < 1e-3, (
            f"Constant target should produce only baseline; got {non_baseline}"
        )


def test_lgboost_to_fourier_drops_zero_coefficients():
    """Output dict must not contain any zero-valued entries."""
    model = _fit_lgbm_on_function(lambda x: float(x[0]))
    fourier = lgboost_to_fourier(model.booster_.dump_model())
    assert all(v != 0.0 for v in fourier.values())


def test_lgboost_to_fourier_keys_are_sorted_tuples():
    """Interaction keys must be tuples of sorted feature indices."""
    model = _fit_lgbm_on_function(lambda x: float(x[0]) + float(x[1]) * float(x[2]))
    fourier = lgboost_to_fourier(model.booster_.dump_model())
    for interaction in fourier:
        assert isinstance(interaction, tuple)
        assert list(interaction) == sorted(interaction)


def test_lgboost_to_fourier_xor_picks_up_pairwise_interaction():
    """XOR of x0 and x1 — the (0, 1) interaction should be among the top by magnitude."""
    model = _fit_lgbm_on_function(lambda x: float(int(x[0]) ^ int(x[1])))
    fourier = lgboost_to_fourier(model.booster_.dump_model())
    # The (0, 1) coefficient should be present and non-trivial.
    assert (0, 1) in fourier
    assert abs(fourier[(0, 1)]) > 0.1


# -----------------------------------------------------------------------------
# Integration: lgboost_to_fourier + top_k_interactions
# -----------------------------------------------------------------------------


def test_pipeline_extracts_odd_singletons_from_linear_function():
    """For y = x0 + x2 + x3, (0,), (2,), (3,) should be among the top-3 odd interactions."""
    model = _fit_lgbm_on_function(
        lambda x: float(x[0]) + float(x[2]) + float(x[3]),
        n_features=4, n_samples=1024,
    )
    fourier = lgboost_to_fourier(model.booster_.dump_model())
    top = top_k_interactions(fourier, k=3, odd=True)
    expected_features = {(0,), (2,), (3,)}
    assert expected_features.issubset(set(top.keys())), (
        f"Expected singletons {expected_features} in top-3 odd, got {set(top.keys())}"
    )
