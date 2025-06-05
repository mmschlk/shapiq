"""Tests that check if the explainer gets the correct interaction values on the California Housing dataset."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, get_args

import pytest

from shapiq.explainer.custom_types import ExplainerIndices
from shapiq.explainer.tabular import TabularExplainer
from shapiq.game_theory.indices import index_generalizes_bv, index_generalizes_sv
from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    from shapiq.utils.custom_types import IndexType


def _load_ground_truth_interaction_values(index: IndexType, order: int):
    """Load the ground truth interaction values for the California Housing dataset."""
    save_dir = (
        pathlib.Path(__file__).parent.parent / "data" / "interaction_values" / "california_housing"
    )
    all_files = list(
        save_dir.glob(
            f"iv_california_housing_imputer_9070456741283270540_index={index}_order={order}.pkl"
        )
    )
    assert len(all_files) == 1
    file_path = all_files[0]
    return InteractionValues.load(file_path)


@pytest.mark.parametrize("index", get_args(ExplainerIndices))
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.integration
def test_explainer_california_housing(
    index: IndexType,
    order: int,
    california_housing_train_test_explain: tuple[np.ndarray, ...],
    california_housing_rf_model: RandomForestRegressor,
):
    """Test the explainer on the California Housing dataset."""
    # prepare the expected index based on the order and index
    expected_index = index
    if order == 1:
        expected_index = "BV" if index_generalizes_bv(index) else expected_index
        expected_index = "SV" if index_generalizes_sv(index) else expected_index

    # skip tests for indices that are not possible
    if expected_index in ["BV", "SV"] and order > 1:
        return  # the test is not applicable for higher orders (should be skipped)

    x_train, y_train, x_test, y_test, x_explain = california_housing_train_test_explain
    n_features = x_train.shape[1]

    model = california_housing_rf_model
    explainer = TabularExplainer(
        model=model.predict,
        data=x_test,
        index=index,
        max_order=order,
        random_state=42,
        verbose=False,
        imputer="marginal",
        # imputer params
        sample_size=100,
        joint_marginal_distribution=True,
    )
    iv = explainer.explain(x_explain, budget=2**n_features)
    iv = iv.get_n_order(min_order=1, max_order=order)

    # load the ground truth interaction values
    gt_iv = _load_ground_truth_interaction_values(index, order)

    # check that interaction values are correct
    assert isinstance(iv, InteractionValues)
    assert iv.index == expected_index
    assert iv.index == gt_iv.index
    assert iv.max_order == order
    assert iv.min_order == 1

    # check that the interaction values are computed correctly
    interactions_dict = iv.dict_values
    gt_dict = gt_iv.dict_values
    for key in interactions_dict:
        assert key in gt_dict, f"Key {key} not found in ground truth interaction values."
        assert interactions_dict[key] == pytest.approx(gt_dict[key], abs=0.001), (
            f"Interaction value for key {key} does not match ground truth."
        )
        assert gt_dict[key] == pytest.approx(interactions_dict[key], abs=0.001), (
            f"Ground truth interaction value for key {key} does not match computed value."
        )

    if index not in ["BV", "FBII"]:
        assert gt_iv.baseline_value == pytest.approx(iv.baseline_value, abs=0.001), (
            "Baseline value does not match ground truth."
        )
