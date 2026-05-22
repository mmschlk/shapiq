"""Correctness tests for the fused multi-output interaction kernel.

The oracle is the *existing* scalar flatten path: for each output column ``j`` the
fused multi-output kernel must reproduce, bit-for-bit (up to float rounding), what
``preprocess_boolean_trees`` + ``compute_interactions_flatten`` produce when run on
the scalar value column ``values[:, j]``.
"""

from __future__ import annotations

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")

from shapiq.approximator.proxy._multioutput.tree import (  # noqa: E402
    convert_multioutput_xgboost,
)
from shapiq.tree.interventional.cext import (  # noqa: E402  # ty: ignore[unresolved-import]
    compute_interactions_flatten,
    compute_interactions_flatten_multi,
    preprocess_boolean_trees,
    preprocess_boolean_trees_multi,
)

N_FEATURES = 8
N_OUTPUTS = 5
N_SAMPLES = 200


def _fit_multioutput_tree() -> object:
    """Fit a small multi-output XGBoost regressor on synthetic 0/1 coalition data."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(N_SAMPLES, N_FEATURES)).astype(np.float64)
    # Targets depend on a few features + interactions so the trees actually split.
    base = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] - 0.3 * X[:, 3] + 0.7 * X[:, 4] * X[:, 5]
    coeffs = rng.normal(size=N_OUTPUTS)
    offsets = rng.normal(size=N_OUTPUTS)
    y = base[:, None] * coeffs[None, :] + offsets[None, :]
    model = xgboost.XGBRegressor(
        multi_strategy="multi_output_tree",
        max_depth=4,
        n_estimators=2,
        objective="reg:squarederror",
    )
    model.fit(X, y)
    return model


def _scalar_reference(trees: list, column: int, index: str, max_order: int) -> dict:
    """Run the existing scalar flatten path on a single output column."""
    values_list = [tree.values[:, column].astype(np.float32) for tree in trees]
    features_list = [tree.features.astype(np.int64) for tree in trees]
    children_left_list = [tree.children_left.astype(np.int64) for tree in trees]
    children_right_list = [tree.children_right.astype(np.int64) for tree in trees]

    (
        e_r_flatten,
        leaf_vals_flatten,
        e_size_flatten,
        r_size_flatten,
        feature_in_e,
        leaf_id,
    ) = preprocess_boolean_trees(
        values_list,
        features_list,
        children_left_list,
        children_right_list,
        N_FEATURES,
    )
    return compute_interactions_flatten(
        leaf_vals_flatten,
        e_r_flatten,
        e_size_flatten,
        r_size_flatten,
        feature_in_e,
        leaf_id,
        index,
        len(leaf_vals_flatten),
        N_FEATURES,
        len(e_r_flatten),
        max_order,
        0,  # verbose
        1.0,  # scaling_factor
        None,  # no custom weight table
    )


def _dense_result_size(n_features: int, max_order: int) -> int:
    from math import comb

    return sum(comb(n_features, k) for k in range(1, max_order + 1))


def _dict_to_dense(interactions: dict, n_features: int, max_order: int) -> np.ndarray:
    """Convert the sparse dict returned by compute_interactions_flatten to a dense vector."""
    dense = np.zeros(_dense_result_size(n_features, max_order), dtype=np.float64)
    # order-1: index i
    # order-2: n + compact upper-triangle offset
    # order-3: index3 layout
    base2 = n_features
    base3 = n_features + n_features * (n_features - 1) // 2
    for key, value in interactions.items():
        if len(key) == 1:
            dense[key[0]] = value
        elif len(key) == 2:
            i, j = sorted(key)
            offset = i * n_features - i * (i + 1) // 2 + (j - i - 1)
            dense[base2 + offset] = value
        elif len(key) == 3:
            i, j, k = sorted(key)
            offset = i + j * (j - 1) // 2 + k * (k - 1) * (k - 2) // 6
            dense[base3 + offset] = value
    return dense


@pytest.mark.parametrize(
    ("index", "max_order"),
    [("SV", 1), ("SII", 2), ("SII", 3)],
)
def test_fused_multi_matches_scalar(index: str, max_order: int) -> None:
    """The fused multi kernel must equal the scalar path for every output column."""
    model = _fit_multioutput_tree()
    trees = convert_multioutput_xgboost(model)
    n_outputs = trees[0].n_outputs
    assert n_outputs == N_OUTPUTS

    # --- fused multi path: one preprocess + one kernel call for all outputs ---
    values_list = [tree.values.astype(np.float32) for tree in trees]
    features_list = [tree.features.astype(np.int64) for tree in trees]
    children_left_list = [tree.children_left.astype(np.int64) for tree in trees]
    children_right_list = [tree.children_right.astype(np.int64) for tree in trees]

    (
        e_r_flatten,
        e_size_flatten,
        r_size_flatten,
        feature_in_e,
        leaf_id,
        leaf_values,
    ) = preprocess_boolean_trees_multi(
        values_list,
        features_list,
        children_left_list,
        children_right_list,
        N_FEATURES,
    )
    assert leaf_values.shape[1] == N_OUTPUTS

    fused = compute_interactions_flatten_multi(
        leaf_values,
        e_r_flatten,
        e_size_flatten,
        r_size_flatten,
        feature_in_e,
        leaf_id,
        index,
        len(e_r_flatten),
        N_FEATURES,
        max_order,
        0,  # verbose
        1.0,  # scaling_factor
        None,  # no custom weight table
    )
    assert fused.shape == (N_OUTPUTS, _dense_result_size(N_FEATURES, max_order))

    # --- per-output comparison against the scalar oracle ---
    for j in range(N_OUTPUTS):
        scalar = _scalar_reference(trees, j, index, max_order)
        scalar_dense = _dict_to_dense(scalar, N_FEATURES, max_order)
        np.testing.assert_allclose(
            fused[j],
            scalar_dense,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"mismatch for output column {j} (index={index}, max_order={max_order})",
        )
