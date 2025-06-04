"""Tests that check if the explainer gets the correct interaction values on the California Housing dataset."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pytest

from shapiq.explainer.tabular import TabularExplainer
from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    from shapiq.utils.custom_types import IndexType


def _load_ground_truth_interaction_values(index: IndexType, order: int):
    """Load the ground truth interaction values for the California Housing dataset."""
    save_dir = pathlib.Path(__file__).parent.parent / "data" / "interaction_values"
    all_files = list(save_dir.glob(f"iv_california_housing_*_index={index}_order={order}.pkl"))
    assert len(all_files) == 1
    file_path = all_files[0]
    return InteractionValues.load(file_path)


@pytest.mark.parametrize(
    "index, order",
    [
        ("k-SII", 2),
        ("FSII", 2),
        ("FSII", 3),
        ("SV", 1),
    ],
)
@pytest.mark.integration
def test_explainer_california_housing(
    index,
    order,
    california_housing_train_test_explain,
    california_housing_rf_model,
    california_housing_imputer,
):
    """Test the explainer on the California Housing dataset."""
    gt_iv = _load_ground_truth_interaction_values(index, order)

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
    )
    iv = explainer.explain(x_explain, budget=2**n_features)
    iv = iv.get_n_order(min_order=1, max_order=order)

    interactions_dict = iv.dict_values
    gt_dict = gt_iv.dict_values

    for key in interactions_dict:
        assert key in gt_dict, f"Key {key} not found in ground truth interaction values."
        assert interactions_dict[key] == pytest.approx(gt_dict[key], rel=1e-3), (
            f"Interaction value for key {key} does not match ground truth."
        )
